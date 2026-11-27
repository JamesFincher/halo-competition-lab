#!/usr/bin/env python3
"""Exp169: Qwen3-4B low-data scaling floor under fixed compute.

Exp163's 400-row arm saturated the entity-disjoint v3 behavioral eval, so the
informative curve is below 400 rows. This experiment uses nested complete
counterfactual pairs at 50/100/200/400 unique rows while holding the exp152 fixed
2400-example update budget constant. It therefore estimates the data floor and the
diversity-versus-repetition tradeoff at 48/24/12/6 approximate epochs. Unlike the
first exp163 run, every arm writes row-level behavioral and benchmark outcomes for
paired statistics and auditability.
"""
from __future__ import annotations

import argparse,importlib.util,json,os,random,shutil,sys,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/edit_then_target_rl/scripts'));sys.path.insert(0,str(ROOT))
import run_micro_targeted_rl as base_rl
from experiments.common.serialization_utils import json_safe,write_json
from experiments.common.behavior_stats import comparisons_against_reference,benchmark_comparisons_against_reference
from scripts.continuation_benchmark import run_continuation_benchmark,run_transfer_benchmark

def load_exp152():
 spec=importlib.util.spec_from_file_location('exp152mod',ROOT/'experiments/exp152_qwen3_behavioral_scale_pivot/scripts/run_qwen3_behavioral_scale_pivot.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
exp152=load_exp152()
EXP_DIR=ROOT/'experiments/exp169_qwen3_low_data_scaling_floor';RESULTS_DIR=EXP_DIR/'results';ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'
ROW_COUNTS=[50,100,200,400]

def smoke():return os.environ.get('HALO_SMOKE_TEST')=='1'
def append(rec):
 RESULTS_DIR.mkdir(parents=True,exist_ok=True)
 with ARM_JSONL.open('a',encoding='utf-8') as f:f.write(json.dumps(json_safe(rec),sort_keys=True)+'\n')
def done():
 if not ARM_JSONL.exists():return set()
 out=set()
 for line in ARM_JSONL.read_text(encoding='utf-8').splitlines():
  if line.strip():
   try:out.add(json.loads(line)['arm'])
   except Exception:pass
 return out

def load_train_rows_with_meta(path:Path)->list[dict]:
 rows=[]
 for line in path.read_text(encoding='utf-8').splitlines():
  if not line.strip():continue
  raw=json.loads(line);msgs=raw['messages'];target=json.loads(msgs[2]['content'])
  rows.append({'messages':msgs[:2],'target':target,'assistant_text':msgs[2]['content'],'meta':raw.get('meta',{})})
 return rows

def pair_metadata(rows:list[dict]):
 pair_meta={};first_order=[]
 for row in rows:
  pid=row.get('meta',{}).get('pair_id')
  if not pid:continue
  if pid not in pair_meta:
   pair_meta[pid]={'tool':None,'policy':None};first_order.append(pid)
  tgt=row['target']
  if tgt['action']=='call_tool':pair_meta[pid]['tool']=tgt['tool']
  else:pair_meta[pid]['policy']=tgt['policy_code']
 groups={}
 for pid in first_order:
  cell=(pair_meta[pid]['tool'],pair_meta[pid]['policy'])
  if None in cell:raise ValueError(f'incomplete pair metadata for {pid}')
  groups.setdefault(cell,[]).append(pid)
 return first_order,groups

def nested_complete_pair_subset(rows:list[dict],n_rows:int)->tuple[list[dict],dict]:
 """Stratified nested subset of complete pairs across tool x refusal-policy cells."""
 if n_rows%2:raise ValueError('row count must be even to preserve counterfactual pairs')
 want_pairs=n_rows//2;first_order,groups=pair_metadata(rows);total_pairs=len(first_order)
 if want_pairs>total_pairs:raise ValueError(f'requested {want_pairs} pairs > available {total_pairs}')
 raw={cell:len(ids)*want_pairs/total_pairs for cell,ids in groups.items()}
 alloc={cell:int(x) for cell,x in raw.items()}
 while sum(alloc.values())<want_pairs:
  best=max(groups,key=lambda c:((raw[c]-alloc[c]),len(groups[c])-alloc[c],str(c)))
  cell=best
  if alloc[cell]>=len(groups[cell]):
   candidates=[c for c in groups if alloc[c]<len(groups[c])]
   if not candidates:break
   cell=max(candidates,key=lambda c:len(groups[c])-alloc[c])
  alloc[cell]+=1
 if sum(alloc.values())!=want_pairs:raise ValueError(f'could not allocate {want_pairs} complete pairs across {len(groups)} cells')
 keep=set()
 for cell,ids in groups.items():keep.update(ids[:alloc[cell]])
 out=[r for r in rows if r.get('meta',{}).get('pair_id') in keep]
 pair_counts={pid:sum(1 for r in out if r.get('meta',{}).get('pair_id')==pid) for pid in keep}
 if len(out)!=n_rows or set(pair_counts.values())!={2}:raise ValueError(f'incomplete nested subset rows={len(out)} pair_sizes={sorted(set(pair_counts.values()))}')
 audit={'requested_rows':n_rows,'allocated_pairs':sum(alloc.values()),'cells_total':len(groups),'cells_represented':sum(1 for x in alloc.values() if x),'allocation_by_cell':{str(k):int(v) for k,v in sorted(alloc.items(),key=lambda x:str(x[0])) if v}}
 return out,audit

def measure(model,tok,eval_rows,device,max_new_tokens,continuation_n):
 return {
  'behavior_scaled120':base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_new_tokens,use_cache=True,batch_size=8,include_rows=True),
  'continuation_benchmark':run_continuation_benchmark(model,tok,device,continuation_n,include_rows=True),
  'transfer_benchmark':run_transfer_benchmark(model,tok,device,max(continuation_n//2,1),include_rows=True),
 }

def memory_stats(device):
 if device.type!='cuda':return {'device':'cpu'}
 return {'device':torch.cuda.get_device_name(device),'max_allocated_gb':round(torch.cuda.max_memory_allocated(device)/1e9,2),'max_reserved_gb':round(torch.cuda.max_memory_reserved(device)/1e9,2)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--steps',type=int,default=300);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--max-len',type=int,default=384)
 ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--rank',type=int,default=16);ap.add_argument('--lora-alpha',type=int,default=32);ap.add_argument('--lora-dropout',type=float,default=0.05)
 ap.add_argument('--warmup',type=int,default=20);ap.add_argument('--checkpoint-interval',type=int,default=100);ap.add_argument('--max-new-tokens',type=int,default=72);ap.add_argument('--continuation-n',type=int,default=50);ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args();t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f'[Exp169] Qwen3 low-data scaling floor on {device}',flush=True)
 tok=exp152.load_tokenizer(exp152.MODEL_NAME)
 all_rows=load_train_rows_with_meta(ROOT/'data/policy_train_scaled_v3_3200.jsonl');eval_rows=base_rl.load_rows(ROOT/'data/policy_eval_scaled_v3.jsonl')
 continuation_n=2 if smoke() else args.continuation_n
 ckpt_base=EXP_DIR/('checkpoints_smoke' if smoke() else 'checkpoints')
 if smoke() and ckpt_base.exists():shutil.rmtree(ckpt_base)
 finished=done()
 if 'baseline_zero_shot' not in finished:
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  arm_t0=time.time();model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  rec={'arm':'baseline_zero_shot','rows':0,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  baseline exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}%',flush=True);del model
  if device.type=='cuda':torch.cuda.empty_cache()
 subset_pairs={n:nested_complete_pair_subset(all_rows,n) for n in ROW_COUNTS}
 subsets={n:x[0] for n,x in subset_pairs.items()};audits={str(n):x[1] for n,x in subset_pairs.items()}
 for n in ROW_COUNTS:
  arm=f'rows_{n}'
  if arm in finished:continue
  arm_t0=time.time();print(f'[Exp169] arm {arm}',flush=True)
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  # Hold LoRA initialization fixed across scale arms; only nested data size/repetition varies.
  random.seed(args.seed);torch.manual_seed(args.seed)
  model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  ckpt_root=ckpt_base/arm;final_adapter=ckpt_root/'final_adapter';resume=exp152.select_resume_adapter(ckpt_root)
  model=exp152.make_lora(model,exp152.ATTN_MLP_TARGETS,None,resume,args.rank,args.lora_alpha,args.lora_dropout)
  train_rows=subsets[n]
  train_info=exp152.train_lora(model,tok,train_rows,[],device,ckpt_root,args.steps,args.batch_size,args.max_len,args.lr,args.warmup,args.checkpoint_interval,False)
  train_info['unique_rows']=len(train_rows);train_info['approx_epochs']=(args.steps*args.batch_size)/len(train_rows)
  rec={'arm':arm,'rows':n,'nested_complete_pairs':True,'pair_allocation_audit':audits[str(n)],'fixed_update_budget':args.steps*args.batch_size,'train_info':train_info,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  rows_{n}: exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}% epochs={train_info["approx_epochs"]:.2f}',flush=True)
  del model
  if device.type=='cuda':torch.cuda.empty_cache()
 arms=[json.loads(x) for x in ARM_JSONL.read_text(encoding='utf-8').splitlines() if x.strip()] if ARM_JSONL.exists() else []
 stats_arms=[dict(rec,measurement=rec.get('measurement',{})) for rec in arms]
 write_json(RESULTS_DIR/'qwen3_low_data_scaling_floor_summary.json',{
  'experiment':'exp169_qwen3_low_data_scaling_floor','model':exp152.MODEL_NAME,
  'question':'Where is the low-data floor for entity-disjoint behavior compliance when fixed compute forces high repetition?',
  'design':{'row_counts':ROW_COUNTS,'fixed_update_budget_examples':args.steps*args.batch_size,'nested_complete_pairs':True,'pair_allocation_audits':audits,'motivation':'exp163 log showed rows_400 already at the v3 exact-compliance ceiling'},
  'arms':arms,
  'behavior_row_statistics':{'versus_baseline_zero_shot':comparisons_against_reference(stats_arms,'baseline_zero_shot','behavior_scaled120'),'versus_rows_400':comparisons_against_reference(stats_arms,'rows_400','behavior_scaled120')},
  'continuation_row_statistics':{'versus_baseline_zero_shot':benchmark_comparisons_against_reference(stats_arms,'baseline_zero_shot','continuation_benchmark'),'versus_rows_400':benchmark_comparisons_against_reference(stats_arms,'rows_400','continuation_benchmark')},
  'transfer_row_statistics':{'versus_baseline_zero_shot':benchmark_comparisons_against_reference(stats_arms,'baseline_zero_shot','transfer_benchmark'),'versus_rows_400':benchmark_comparisons_against_reference(stats_arms,'rows_400','transfer_benchmark')},
  'wall_sec':round(time.time()-t0,2),'smoke_test':smoke(),
 })
 if smoke():shutil.rmtree(ckpt_base,ignore_errors=True)
 print(f'[DONE] exp169 arms={len(arms)} wall={round(time.time()-t0,2)}s',flush=True)
if __name__=='__main__':main()
