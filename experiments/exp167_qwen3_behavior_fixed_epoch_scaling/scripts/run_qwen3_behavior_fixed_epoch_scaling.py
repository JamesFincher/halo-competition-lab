#!/usr/bin/env python3
"""Exp167: Qwen3-4B fixed-epoch behavior data scaling curve.

Uses nested, complete-counterfactual-pair, tool/policy-stratified v3 subsets at
400/800/1600/3200 rows and trains exactly one pass over each subset. Unlike exp163's
fixed-compute diversity/repetition curve, update count scales with row count. Warmup
and cosine decay are proportional to each arm's step count. Together exp163+exp167
separate unique-data diversity at fixed compute from data+compute scaling at fixed epochs.
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
from experiments.common.behavior_stats import comparisons_against_reference  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark, run_transfer_benchmark  # noqa: E402

def load_exp152():
 spec=importlib.util.spec_from_file_location('exp152mod',ROOT/'experiments/exp152_qwen3_behavioral_scale_pivot/scripts/run_qwen3_behavioral_scale_pivot.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
exp152=load_exp152()
EXP_DIR=ROOT/'experiments/exp167_qwen3_behavior_fixed_epoch_scaling';RESULTS_DIR=EXP_DIR/'results';ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'
ROW_COUNTS=[400,800,1600,3200]

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


def load_train_rows_with_meta(path: Path) -> list[dict]:
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        raw=json.loads(line);msgs=raw['messages'];target=json.loads(msgs[2]['content'])
        rows.append({'messages':msgs[:2],'target':target,'assistant_text':msgs[2]['content'],'meta':raw.get('meta',{})})
    return rows

def nested_complete_pair_subset(rows:list[dict],n_rows:int)->list[dict]:
    """Stratified nested subset of complete pairs across tool x refusal-policy cells."""
    if n_rows%2:raise ValueError('row count must be even to preserve counterfactual pairs')
    want_pairs=n_rows//2
    pair_meta={};groups={};first_order=[]
    for row in rows:
        pid=row.get('meta',{}).get('pair_id')
        if not pid:continue
        if pid not in pair_meta:
            pair_meta[pid]={'tool':None,'policy':None};first_order.append(pid)
        tgt=row['target']
        if tgt['action']=='call_tool':pair_meta[pid]['tool']=tgt['tool']
        else:pair_meta[pid]['policy']=tgt['policy_code']
    for pid in first_order:
        cell=(pair_meta[pid]['tool'],pair_meta[pid]['policy'])
        if None in cell:raise ValueError(f'incomplete pair metadata for {pid}')
        groups.setdefault(cell,[]).append(pid)
    total_pairs=len(first_order)
    if want_pairs>total_pairs:raise ValueError(f'requested {want_pairs} pairs > available {total_pairs}')
    raw={cell:len(ids)*want_pairs/total_pairs for cell,ids in groups.items()}
    alloc={cell:int(x) for cell,x in raw.items()}
    while sum(alloc.values())<want_pairs:
        best=max(groups,key=lambda c:((raw[c]-alloc[c]),len(groups[c])-alloc[c],str(c)))
        if alloc[cell:=best]>=len(groups[best]):
            # Fall back to any underfilled cell if exact largest-remainder choices are exhausted.
            candidates=[c for c in groups if alloc[c]<len(groups[c])]
            if not candidates:break
            cell=max(candidates,key=lambda c:len(groups[c])-alloc[c])
        alloc[cell]+=1
    keep=set()
    for cell,ids in groups.items():
        keep.update(ids[:alloc[cell]])
    out=[r for r in rows if r.get('meta',{}).get('pair_id') in keep]
    pair_counts={pid:sum(1 for r in out if r.get('meta',{}).get('pair_id')==pid) for pid in keep}
    if len(out)!=n_rows or set(pair_counts.values())!={2}:raise ValueError(f'incomplete nested subset rows={len(out)} pair_sizes={sorted(set(pair_counts.values()))}')
    return out

def measure(model,tok,eval_rows,device,max_new_tokens,continuation_n):
 return {'behavior_scaled120':base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_new_tokens,use_cache=True,batch_size=4,include_rows=True),'continuation_benchmark':run_continuation_benchmark(model,tok,device,continuation_n),'transfer_benchmark':run_transfer_benchmark(model,tok,device,max(continuation_n//2,1))}

def memory_stats(device):
 if device.type!='cuda':return {'device':'cpu'}
 return {'device':torch.cuda.get_device_name(device),'max_allocated_gb':round(torch.cuda.max_memory_allocated(device)/1e9,2),'max_reserved_gb':round(torch.cuda.max_memory_reserved(device)/1e9,2)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--steps',type=int,default=0)  # 0 means exactly one epoch per arm; positive override is for smoke/debugging
 ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--max-len',type=int,default=384)
 ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--rank',type=int,default=16);ap.add_argument('--lora-alpha',type=int,default=32);ap.add_argument('--lora-dropout',type=float,default=0.05)
 ap.add_argument('--warmup',type=int,default=20);ap.add_argument('--checkpoint-interval',type=int,default=100);ap.add_argument('--max-new-tokens',type=int,default=72);ap.add_argument('--continuation-n',type=int,default=50);ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args();t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f'[Exp167] Qwen3 behavior fixed-epoch scaling on {device}',flush=True)
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
 subsets={n:nested_complete_pair_subset(all_rows,n) for n in ROW_COUNTS}
 for n in ROW_COUNTS:
  arm=f'rows_{n}'
  if arm in finished:continue
  arm_t0=time.time();print(f'[Exp167] arm {arm}',flush=True)
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  # Hold LoRA initialization fixed across scale arms; only nested data size/repetition varies.
  random.seed(args.seed);torch.manual_seed(args.seed)
  model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  ckpt_root=ckpt_base/arm;final_adapter=ckpt_root/'final_adapter';resume=final_adapter if final_adapter.exists() else exp152.latest_checkpoint(ckpt_root)[0]
  model=exp152.make_lora(model,exp152.ATTN_MLP_TARGETS,None,resume,args.rank,args.lora_alpha,args.lora_dropout)
  train_rows=subsets[n]
  if len(train_rows)%args.batch_size:raise ValueError(f'row count {len(train_rows)} not divisible by batch size {args.batch_size}')
  arm_steps=args.steps if args.steps>0 else len(train_rows)//args.batch_size
  arm_warmup=max(1,int(round(args.warmup*arm_steps/300)))
  train_info=exp152.train_lora(model,tok,train_rows,[],device,ckpt_root,arm_steps,args.batch_size,args.max_len,args.lr,arm_warmup,args.checkpoint_interval,False)
  train_info['unique_rows']=len(train_rows);train_info['approx_epochs']=(arm_steps*args.batch_size)/len(train_rows);train_info['warmup_steps']=arm_warmup;train_info['schedule']='warmup/cosine proportions matched to arm length'
  rec={'arm':arm,'rows':n,'nested_complete_pairs':True,'fixed_epoch_target':1.0,'planned_steps':arm_steps,'update_budget':arm_steps*args.batch_size,'train_info':train_info,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  rows_{n}: exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}% epochs={train_info["approx_epochs"]:.2f}',flush=True)
  del model
  if device.type=='cuda':torch.cuda.empty_cache()
 arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()] if ARM_JSONL.exists() else []
 write_json(RESULTS_DIR/'qwen3_behavior_fixed_epoch_summary.json',{'experiment':'exp167_qwen3_behavior_fixed_epoch_scaling','model':exp152.MODEL_NAME,'question':'How does entity-disjoint behavior compliance scale with 400/800/1600/3200 unique counterfactual rows when each arm trains exactly one epoch and compute scales with data count?','arms':arms,'behavior_row_statistics':{'versus_baseline_zero_shot':comparisons_against_reference(arms,'baseline_zero_shot'),'versus_rows_1600':comparisons_against_reference(arms,'rows_1600')},'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()})
 if smoke():shutil.rmtree(ckpt_base,ignore_errors=True)
 print(f'[DONE] exp167 arms={len(arms)} wall={round(time.time()-t0,2)}s',flush=True)
if __name__=='__main__':main()
