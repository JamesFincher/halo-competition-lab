#!/usr/bin/env python3
"""Exp165: Qwen3-4B capability-replay source controls.

At the exp152 reference replay frequency, separates coherent benchmark-specific replay,
coherent benchmark-independent replay, answer-format statistics, and simple displacement
of behavior-training steps. A reduced behavior-only control matches the number of behavior
steps in replay arms. Behavior v2 exactness measures compliance cost; in-domain and
held-out transfer continuation benchmarks measure capability benefit.
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
EXP_DIR=ROOT/'experiments/exp165_qwen3_capability_replay_source_controls';RESULTS_DIR=EXP_DIR/'results';ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'
REPLAY_EVERY=4

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


def measure(model,tok,eval_rows,device,max_new_tokens,continuation_n):
 return {'behavior_scaled120':base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_new_tokens,use_cache=True,batch_size=4,include_rows=True),'continuation_benchmark':run_continuation_benchmark(model,tok,device,continuation_n,include_rows=True),'transfer_benchmark':run_transfer_benchmark(model,tok,device,continuation_n,include_rows=True)}

def memory_stats(device):
 if device.type!='cuda':return {'device':'cpu'}
 return {'device':torch.cuda.get_device_name(device),'max_allocated_gb':round(torch.cuda.max_memory_allocated(device)/1e9,2),'max_reserved_gb':round(torch.cuda.max_memory_reserved(device)/1e9,2)}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--steps',type=int,default=300);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--max-len',type=int,default=384)
 ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--rank',type=int,default=16);ap.add_argument('--lora-alpha',type=int,default=32);ap.add_argument('--lora-dropout',type=float,default=0.05)
 ap.add_argument('--warmup',type=int,default=20);ap.add_argument('--checkpoint-interval',type=int,default=100);ap.add_argument('--max-new-tokens',type=int,default=72);ap.add_argument('--continuation-n',type=int,default=100);ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args();t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f'[Exp165] Qwen3 capability-replay source controls on {device}',flush=True)
 tok=exp152.load_tokenizer(exp152.MODEL_NAME)
 behavior_rows=base_rl.load_rows(ROOT/'data/policy_train_scaled_v2.jsonl');capability_rows=[json.loads(x) for x in (ROOT/'data/capability_train_scaled.jsonl').read_text().splitlines() if x.strip()];eval_rows=base_rl.load_rows(ROOT/'data/policy_eval_scaled_v2.jsonl')
 continuation_n=2 if smoke() else args.continuation_n
 ckpt_base=EXP_DIR/('checkpoints_smoke' if smoke() else 'checkpoints')
 if smoke() and ckpt_base.exists():shutil.rmtree(ckpt_base)
 finished=done()
 if 'baseline_zero_shot' not in finished:
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  arm_t0=time.time();model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  rec={'arm':'baseline_zero_shot','replay_every':None,'replay_source':'zero_shot','measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  baseline exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}%',flush=True);del model
  if device.type=='cuda':torch.cuda.empty_cache()
 rng=random.Random(args.seed);rng.shuffle(behavior_rows);rng.shuffle(capability_rows)
 general_rows=[json.loads(x) for x in (ROOT/'data/capability_train_general_v1.jsonl').read_text().splitlines() if x.strip()]
 rng.shuffle(general_rows)
 # Rotate answers by one: preserves answer-token/format statistics while destroying
 # context-answer coherence. This separates interference/displacement from learning.
 shuffled_rows=[]
 for i,row in enumerate(capability_rows):
  bad=dict(row);bad['answer_text']=capability_rows[(i+1)%len(capability_rows)]['answer_text'];shuffled_rows.append(bad)
 matched_behavior_steps=max(1,int(args.steps*0.75))
 armspecs=[
  (f'behavior_only_{args.steps}_steps',None,args.steps),
  (f'behavior_only_{matched_behavior_steps}_steps',None,matched_behavior_steps),
  ('benchmark_replay_every4','benchmark',args.steps),
  ('general_corpus_replay_every4','general',args.steps),
  ('shuffled_benchmark_replay_every4','shuffled',args.steps),
 ]
 for arm,replay_source,arm_steps in armspecs:
  if arm in finished:continue
  arm_t0=time.time();print(f'[Exp165] arm {arm}',flush=True)
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  # Hold LoRA initialization fixed across source arms; only replay content/dose varies.
  random.seed(args.seed);torch.manual_seed(args.seed)
  model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
  ckpt_root=ckpt_base/arm;final_adapter=ckpt_root/'final_adapter';resume=final_adapter if final_adapter.exists() else exp152.latest_checkpoint(ckpt_root)[0]
  model=exp152.make_lora(model,exp152.ATTN_MLP_TARGETS,None,resume,args.rank,args.lora_alpha,args.lora_dropout)
  rows={'benchmark':capability_rows,'general':general_rows,'shuffled':shuffled_rows}.get(replay_source,[])
  use_replay=replay_source is not None
  train_info=exp152.train_lora(model,tok,behavior_rows,rows,device,ckpt_root,arm_steps,args.batch_size,args.max_len,args.lr,args.warmup,args.checkpoint_interval,use_replay,REPLAY_EVERY)
  cap_steps=sum(1 for step in range(arm_steps) if use_replay and step%REPLAY_EVERY==REPLAY_EVERY-1);behavior_steps=arm_steps-cap_steps
  train_info['unique_rows']=len(behavior_rows);train_info['replay_source']=replay_source;train_info['expected_capability_steps']=cap_steps;train_info['expected_behavior_steps']=behavior_steps;train_info['approx_behavior_epochs']=(behavior_steps*args.batch_size)/len(behavior_rows)
  rec={'arm':arm,'replay_every':REPLAY_EVERY if use_replay else 0,'replay_source':replay_source,'planned_steps':arm_steps,'behavior_rows':len(behavior_rows),'arm_update_budget':arm_steps*args.batch_size,'reference_update_budget':args.steps*args.batch_size,'train_info':train_info,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
  append(rec);print(f'  source_arm: exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}% cap_steps={train_info["expected_capability_steps"]}',flush=True)
  del model
  if device.type=='cuda':torch.cuda.empty_cache()
 arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()] if ARM_JSONL.exists() else []
 write_json(RESULTS_DIR/'qwen3_capability_replay_source_summary.json',{'experiment':'exp165_qwen3_capability_replay_source_controls','model':exp152.MODEL_NAME,'question':'Does capability replay alter behavior because of coherent benchmark-specific content, coherent benchmark-independent content, answer-format statistics, or simply displacement of behavior training steps?','arms':arms,'behavior_row_statistics':{'versus_baseline_zero_shot':comparisons_against_reference(arms,'baseline_zero_shot'),'versus_matched_behavior_only':comparisons_against_reference(arms,f'behavior_only_{matched_behavior_steps}_steps')},'capability_row_statistics':{'continuation_versus_matched_behavior_only':benchmark_comparisons_against_reference(arms,f'behavior_only_{matched_behavior_steps}_steps','continuation_benchmark'),'transfer_versus_matched_behavior_only':benchmark_comparisons_against_reference(arms,f'behavior_only_{matched_behavior_steps}_steps','transfer_benchmark')},'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()})
 if smoke():shutil.rmtree(ckpt_base,ignore_errors=True)
 print(f'[DONE] exp165 arms={len(arms)} wall={round(time.time()-t0,2)}s',flush=True)
if __name__=='__main__':main()
