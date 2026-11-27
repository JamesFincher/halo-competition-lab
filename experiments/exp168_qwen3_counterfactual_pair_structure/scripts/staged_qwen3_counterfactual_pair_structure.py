#!/usr/bin/env python3
"""Staged exp168: counterfactual-pair structure ablation for Qwen3-4B behavior SFT.

Inactive until exp163/exp167 establish the scaling-curve shape. At fixed 1,600
examples/200 optimizer updates, compares complete distributed pairs, singleton rows
from twice as many pairs, repeated singleton rows from the same pairs, and complete
pairs reordered adjacently. This separates counterfactual pair coverage, context
diversity, repetition, and batch adjacency without changing model/optimizer/budget.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, random, shutil, sys, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/edit_then_target_rl/scripts'))
sys.path.insert(0,str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import json_safe, write_json  # noqa: E402
from experiments.common.behavior_stats import benchmark_comparisons_against_reference, comparisons_against_reference  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark, run_transfer_benchmark  # noqa: E402

def load_exp152():
 spec=importlib.util.spec_from_file_location('exp152mod',ROOT/'experiments/exp152_qwen3_behavioral_scale_pivot/scripts/run_qwen3_behavioral_scale_pivot.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
exp152=load_exp152()
EXP_DIR=ROOT/'experiments/exp168_qwen3_counterfactual_pair_structure';RESULTS_DIR=EXP_DIR/'results';ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'

def smoke():return os.environ.get('HALO_SMOKE_TEST')=='1'
def append(rec):
 RESULTS_DIR.mkdir(parents=True,exist_ok=True)
 with ARM_JSONL.open('a') as f:f.write(json.dumps(json_safe(rec),sort_keys=True)+'\n')
def done():
 if not ARM_JSONL.exists():return set()
 out=set()
 for line in ARM_JSONL.read_text().splitlines():
  if line.strip():
   try:out.add(json.loads(line)['arm'])
   except Exception:pass
 return out

def load_train_rows_with_meta(path:Path)->list[dict]:
 rows=[]
 for line in path.read_text().splitlines():
  if not line.strip():continue
  raw=json.loads(line);msgs=raw['messages'];target=json.loads(msgs[2]['content'])
  rows.append({'messages':msgs[:2],'target':target,'assistant_text':msgs[2]['content'],'meta':raw.get('meta',{})})
 return rows

def pair_index(rows:list[dict])->tuple[list[str],dict[str,list[dict]]]:
 order=[];pairs={}
 for row in rows:
  pid=row.get('meta',{}).get('pair_id')
  if not pid:continue
  if pid not in pairs:pairs[pid]=[];order.append(pid)
  pairs[pid].append(row)
 bad=[pid for pid in order if len(pairs[pid])!=2]
 if bad:raise ValueError(f'incomplete counterfactual pairs: {bad[:5]}')
 return order,pairs

def pair_cell(pair:list[dict])->tuple[str,str]:
 tool=policy=None
 for row in pair:
  tgt=row['target']
  if tgt['action']=='call_tool':tool=tgt['tool']
  else:policy=tgt['policy_code']
 if tool is None or policy is None:raise ValueError('pair does not contain one allow and one refusal row')
 return tool,policy

def stratified_pairs(order:list[str],pairs:dict[str,list[dict]],want_pairs:int)->list[str]:
 groups={}
 for pid in order:groups.setdefault(pair_cell(pairs[pid]),[]).append(pid)
 total=len(order)
 if want_pairs>total:raise ValueError(f'requested {want_pairs} pairs > available {total}')
 raw={cell:len(ids)*want_pairs/total for cell,ids in groups.items()}
 alloc={cell:int(x) for cell,x in raw.items()}
 while sum(alloc.values())<want_pairs:
  candidates=[c for c in groups if alloc[c]<len(groups[c])]
  if not candidates:break
  cell=max(candidates,key=lambda c:((raw[c]-alloc[c]),len(groups[c])-alloc[c],str(c)))
  alloc[cell]+=1
 if sum(alloc.values())!=want_pairs:raise ValueError('could not allocate stratified pair subset')
 keep=[]
 for cell,ids in groups.items():keep.extend(ids[:alloc[cell]])
 keep_set=set(keep)
 return [pid for pid in order if pid in keep_set]

def rows_for_pairs(rows:list[dict],pair_ids:list[str])->list[dict]:
 keep=set(pair_ids)
 out=[r for r in rows if r.get('meta',{}).get('pair_id') in keep]
 if len(out)!=2*len(pair_ids):raise ValueError(f'expected {2*len(pair_ids)} complete-pair rows, got {len(out)}')
 return out

def singleton_rows(pairs:dict[str,list[dict]],pair_ids:list[str],rows:list[dict])->tuple[list[dict],dict]:
 """Choose exactly half allow/half refusal singleton rows with maximal within-cell balance."""
 if len(pair_ids)%2:raise ValueError('singleton pair count must be even for exact role balance')
 groups={}
 for pid in pair_ids:groups.setdefault(pair_cell(pairs[pid]),[]).append(pid)
 want_allow=len(pair_ids)//2
 raw={cell:len(ids)*want_allow/len(pair_ids) for cell,ids in groups.items()}
 alloc={cell:int(x) for cell,x in raw.items()}
 while sum(alloc.values())<want_allow:
  candidates=[c for c in groups if alloc[c]<len(groups[c])]
  cell=max(candidates,key=lambda c:((raw[c]-alloc[c]),len(groups[c])-alloc[c],str(c)))
  alloc[cell]+=1
 if sum(alloc.values())!=want_allow:raise ValueError('could not allocate exact singleton role balance')
 chosen_keys=set();role_counts={'allow':0,'refuse':0}
 for cell,ids in groups.items():
  for position,pid in enumerate(ids):
   want_role='allow' if position<alloc[cell] else 'refuse'
   matches=[]
   for row in pairs[pid]:
    role='allow' if row['target']['action']=='call_tool' else 'refuse'
    if role==want_role:matches.append(row)
   if len(matches)!=1:raise ValueError(f'role selection failed for {pid}')
   chosen_keys.add((pid,want_role));role_counts[want_role]+=1
 # Preserve the corpus's distributed row order rather than grouping by pair.
 out=[]
 for row in rows:
  pid=row.get('meta',{}).get('pair_id');role='allow' if row['target']['action']=='call_tool' else 'refuse'
  if (pid,role) in chosen_keys:out.append(row)
 if len(out)!=len(pair_ids):raise ValueError(f'expected {len(pair_ids)} singleton rows, got {len(out)}')
 if role_counts!={'allow':len(pair_ids)//2,'refuse':len(pair_ids)//2}:raise ValueError(f'imbalanced singleton roles: {role_counts}')
 return out,{'role_counts':role_counts,'within_cell_largest_remainder_role_balance':True}

def batch_adjacent_rows(pairs:dict[str,list[dict]],pair_ids:list[str])->list[dict]:
 out=[]
 for pid in pair_ids:out.extend(pairs[pid])
 return out

def integrity(rows:list[dict],label:str)->dict:
 pids=[r.get('meta',{}).get('pair_id') for r in rows]
 roles=['allow' if r['target']['action']=='call_tool' else 'refuse' for r in rows]
 pair_positions={}
 for i,pid in enumerate(pids):pair_positions.setdefault(pid,[]).append(i)
 adjacent=sum(1 for pos in pair_positions.values() if len(pos)==2 and pos[1]==pos[0]+1)
 return {'construction':label,'examples':len(rows),'unique_examples':len({json.dumps({'messages':r['messages'],'target':r['target']},sort_keys=True) for r in rows}),'unique_pairs':len(set(pids)),'allow_rows':roles.count('allow'),'refuse_rows':roles.count('refuse'),'adjacent_pair_count':adjacent}

def measure(model,tok,eval_rows,device,max_new_tokens,continuation_n):
 return {'behavior_scaled120':base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_new_tokens,use_cache=True,batch_size=8,include_rows=True),'continuation_benchmark':run_continuation_benchmark(model,tok,device,continuation_n,include_rows=True),'transfer_benchmark':run_transfer_benchmark(model,tok,device,continuation_n,include_rows=True)}

def memory_stats(device):
 if device.type!='cuda':return {'device':'cpu'}
 return {'device':torch.cuda.get_device_name(device),'max_allocated_gb':round(torch.cuda.max_memory_allocated(device)/1e9,2),'max_reserved_gb':round(torch.cuda.max_memory_reserved(device)/1e9,2)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--steps',type=int,default=200);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--max-len',type=int,default=384)
 ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--rank',type=int,default=16);ap.add_argument('--lora-alpha',type=int,default=32);ap.add_argument('--lora-dropout',type=float,default=0.05)
 ap.add_argument('--warmup',type=int,default=20);ap.add_argument('--checkpoint-interval',type=int,default=100);ap.add_argument('--max-new-tokens',type=int,default=72);ap.add_argument('--continuation-n',type=int,default=100);ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args();t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f'[Exp168] Qwen3 counterfactual-pair structure ablation on {device}',flush=True)
 tok=exp152.load_tokenizer(exp152.MODEL_NAME)
 all_rows=load_train_rows_with_meta(ROOT/'data/policy_train_scaled_v3_3200.jsonl');eval_rows=base_rl.load_rows(ROOT/'data/policy_eval_scaled_v3.jsonl')
 order,pairs=pair_index(all_rows);complete800=stratified_pairs(order,pairs,800);all1600=order[:1600]
 if len(all1600)!=1600:raise ValueError(f'expected 1600 pairs, found {len(all1600)}')
 complete_rows=rows_for_pairs(all_rows,complete800)
 singleton1600,singleton1600_meta=singleton_rows(pairs,all1600,all_rows)
 singleton800,singleton800_meta=singleton_rows(pairs,complete800,all_rows)
 repeated800=singleton800+singleton800
 adjacent_complete=batch_adjacent_rows(pairs,complete800)
 arm_data=[
  ('complete_pairs_800_distributed',complete_rows,integrity(complete_rows,'complete_pairs_800_distributed')),
  ('singleton_pairs_1600',singleton1600,{**integrity(singleton1600,'singleton_pairs_1600'),**singleton1600_meta}),
  ('singleton_pairs_800_repeated_twice',repeated800,{**integrity(repeated800,'singleton_pairs_800_repeated_twice'),**singleton800_meta,'repetitions_per_unique_row':2}),
  ('complete_pairs_800_batch_adjacent',adjacent_complete,integrity(adjacent_complete,'complete_pairs_800_batch_adjacent')),
 ]
 continuation_n=2 if smoke() else args.continuation_n
 ckpt_base=EXP_DIR/('checkpoints_smoke' if smoke() else 'checkpoints')
 if smoke() and ckpt_base.exists():shutil.rmtree(ckpt_base)
 finished=done()
 if 'baseline_zero_shot' not in finished:
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  arm_t0=time.time();model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  rec={'arm':'baseline_zero_shot','measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  baseline exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}%',flush=True);del model
  if device.type=='cuda':torch.cuda.empty_cache()
 for arm,train_rows,data_info in arm_data:
  if arm in finished:continue
  if len(train_rows)!=1600:raise ValueError(f'{arm} has {len(train_rows)} examples, expected 1600')
  arm_t0=time.time();print(f'[Exp168] arm {arm}',flush=True)
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  random.seed(args.seed);torch.manual_seed(args.seed)
  model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  ckpt_root=ckpt_base/arm;final_adapter=ckpt_root/'final_adapter';resume=final_adapter if final_adapter.exists() else exp152.latest_checkpoint(ckpt_root)[0]
  model=exp152.make_lora(model,exp152.ATTN_MLP_TARGETS,None,resume,args.rank,args.lora_alpha,args.lora_dropout)
  train_info=exp152.train_lora(model,tok,train_rows,[],device,ckpt_root,args.steps,args.batch_size,args.max_len,args.lr,args.warmup,args.checkpoint_interval,False)
  train_info['unique_rows']=data_info['unique_examples'];train_info['approx_epochs']=(args.steps*args.batch_size)/len(train_rows)
  rec={'arm':arm,'fixed_example_budget':1600,'fixed_update_budget':args.steps*args.batch_size,'data_integrity':data_info,'train_info':train_info,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  {arm}: exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}%',flush=True)
  del model
  if device.type=='cuda':torch.cuda.empty_cache()
 arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()] if ARM_JSONL.exists() else []
 reference='complete_pairs_800_distributed'
 write_json(RESULTS_DIR/'qwen3_counterfactual_pair_structure_summary.json',{'experiment':'exp168_qwen3_counterfactual_pair_structure','model':exp152.MODEL_NAME,'question':'At fixed 1600-example/200-update behavior SFT cost, how much of the scaled-corpus benefit depends on complete counterfactual pair coverage versus context diversity, repetition, or batch adjacency?','arms':arms,'behavior_row_statistics':{'versus_baseline_zero_shot':comparisons_against_reference(arms,'baseline_zero_shot'),'versus_complete_pairs_distributed':comparisons_against_reference(arms,reference)},'capability_row_statistics':{'continuation_versus_baseline_zero_shot':benchmark_comparisons_against_reference(arms,'baseline_zero_shot','continuation_benchmark'),'continuation_versus_complete_pairs_distributed':benchmark_comparisons_against_reference(arms,reference,'continuation_benchmark'),'transfer_versus_baseline_zero_shot':benchmark_comparisons_against_reference(arms,'baseline_zero_shot','transfer_benchmark'),'transfer_versus_complete_pairs_distributed':benchmark_comparisons_against_reference(arms,reference,'transfer_benchmark')},'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()})
 if smoke():shutil.rmtree(ckpt_base,ignore_errors=True)
 print(f'[DONE] exp168 arms={len(arms)} wall={round(time.time()-t0,2)}s',flush=True)
if __name__=='__main__':main()
