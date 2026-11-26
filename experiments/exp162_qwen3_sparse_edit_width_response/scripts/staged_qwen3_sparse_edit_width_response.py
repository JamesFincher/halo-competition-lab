#!/usr/bin/env python3
"""Staged exp162: Qwen3 sparse-edit width dose response.

No training. Isolates the number of edited down-projection columns (16/32/64/128)
because exp152's first 4B H22-style arm uses 64 while the 135M ultra-sparse sweet spot
was 16. This prevents overinterpreting a width-specific failure as a composition failure.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/edit_then_target_rl/scripts'))
sys.path.insert(0,str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import json_safe, write_json  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark  # noqa: E402

def load_module():
 spec=importlib.util.spec_from_file_location('exp152mod',ROOT/'experiments/exp152_qwen3_behavioral_scale_pivot/scripts/run_qwen3_behavioral_scale_pivot.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
exp152=load_module()
EXP_DIR=ROOT/'experiments/exp162_qwen3_sparse_edit_width_response';RESULTS_DIR=EXP_DIR/'results';ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'
WIDTHS=[16,32,64,128]

def smoke():return os.environ.get('HALO_SMOKE_TEST')=='1'
def append(rec):
 RESULTS_DIR.mkdir(parents=True,exist_ok=True)
 with ARM_JSONL.open('a') as f:f.write(json.dumps(json_safe(rec),sort_keys=True)+'\n')
def done():
 if not ARM_JSONL.exists():return set()
 return {json.loads(x)['arm'] for x in ARM_JSONL.read_text().splitlines() if x.strip()}

def measure(model,tok,eval_rows,device,max_new_tokens,continuation_n):
 return {'behavior_scaled120':base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_new_tokens,use_cache=True),'continuation_benchmark':run_continuation_benchmark(model,tok,device,continuation_n)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-new-tokens',type=int,default=72);ap.add_argument('--continuation-n',type=int,default=100);ap.add_argument('--strength',type=float,default=0.9);ap.add_argument('--n-pairs',type=int,default=12);ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
 torch.manual_seed(args.seed);t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f'[Exp162] Qwen3 sparse-edit width response on {device}',flush=True)
 tok=exp152.load_tokenizer(exp152.MODEL_NAME);train_rows=base_rl.load_rows(ROOT/'data/policy_train_scaled_v2.jsonl');eval_rows=base_rl.load_rows(ROOT/'data/policy_eval_scaled_v2.jsonl')
 continuation_n=2 if smoke() else args.continuation_n
 finished=done();arms=[]
 if 'baseline_zero_shot' not in finished:
  model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  rec={'arm':'baseline_zero_shot','width':0,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()}
  append(rec);print('  baseline done',flush=True);del model
  if device.type=='cuda':torch.cuda.empty_cache()
 for width in WIDTHS:
  arm=f'columns_{width}'
  if arm in finished:continue
  arm_t0=time.time();print(f'[Exp162] arm {arm}',flush=True)
  model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval();edit_layers,_=exp152.architecture_mapped_layers(model)
  edit_info=exp152.apply_validated_rank1_edit(model,tok,train_rows,device,edit_layers=edit_layers,top_columns=width,strength=args.strength,n_pairs=2 if smoke() else args.n_pairs)
  rec={'arm':arm,'width':width,'strength':args.strength,'edit_layers':edit_layers,'edit_info':edit_info,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  columns_{width}: exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}%',flush=True)
  del model
  if device.type=='cuda':torch.cuda.empty_cache()
 arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()]
 write_json(RESULTS_DIR/'qwen3_sparse_edit_width_summary.json',{'experiment':'exp162_qwen3_sparse_edit_width_response','model':exp152.MODEL_NAME,'question':'How does sparse-edit column width affect Qwen3-4B behavior/capability collateral at fixed strength and architecture-mapped layers?','arms':arms,'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()})
 print(f'[DONE] exp162 arms={len(arms)} wall={round(time.time()-t0,2)}s',flush=True)
if __name__=='__main__':main()
