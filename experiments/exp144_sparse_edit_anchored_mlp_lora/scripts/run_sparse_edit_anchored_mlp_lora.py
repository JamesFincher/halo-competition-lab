#!/usr/bin/env python3
"""Experiment 144 / H22: ultra-sparse edit + post-edit-anchored complementary MLP LoRA."""
from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
for x in ['experiments/edit_then_target_rl/scripts','experiments/exp50_surgically_anchored_vector_rl/scripts','experiments/exp57_halo_native_targeted_lora/scripts','experiments/common','scripts']: sys.path.insert(0,str(ROOT/x))
from serialization_utils import write_json
import run_micro_targeted_rl as base_rl
import run_surgically_anchored_vector_rl as vec_rl
import run_halo_native_targeted_lora as halo_lora
from validated_edit import apply_validated_rank1_edit
from real_benchmark import run_benchmark

SEEDS=[7,42,123]; CONTROL_SEED=42
COMPLEMENT=[24,25,26,27]; RANDOM=[16,17,18,19]; INCLUSIVE=[24,25,26,27,28,29]

def delta(a,b): return {k:round(a[k]-b[k],2) for k in a if k in b and isinstance(a[k],(int,float))}
def endpoint(model,tok,eval_rows,device,bench_n,max_tokens):
 ev=base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_tokens)
 bench=run_benchmark(model,tok,device,bench_n)
 return {'behavior_eval':ev,'benchmark':bench}
def train_lora(name,base,ref,tok,prompts,device,target,args,seed):
 m,stats=base_rl.run_grpo_arm(name,base,ref,tok,prompts,device,target,16,args.steps,args.group_size,args.lr,args.beta,args.step_log,seed,max_new_tokens=args.max_new_tokens)
 pre=endpoint(m,tok,eval_rows_global,device,args.bench_n,args.max_new_tokens)
 exported=m.merge_and_unload()
 post=endpoint(exported,tok,eval_rows_global,device,args.bench_n,args.max_new_tokens)
 return {'train_stats':stats,'adapter_eval':pre,'exported_eval':post,'target_regex':target}

def main():
 global eval_rows_global
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=8); ap.add_argument('--group-size',type=int,default=4); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--beta',type=float,default=.02); ap.add_argument('--bench-n',type=int,default=50); ap.add_argument('--max-new-tokens',type=int,default=64); ap.add_argument('--seed-count',type=int,default=3); ap.add_argument('--control-count',type=int,default=4); args=ap.parse_args(); args.steps=min(args.steps,10)
 t0=time.time(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=ROOT/'experiments/exp144_sparse_edit_anchored_mlp_lora/results'; out.mkdir(parents=True,exist_ok=True)
 print(f'[Exp 144/H22] sparse edit + anchored MLP LoRA on {device}',flush=True)
 train_rows=base_rl.load_rows(ROOT/'data/policy_train.jsonl'); eval_rows_global=base_rl.load_rows(ROOT/'data/policy_eval.jsonl'); prompts=vec_rl.make_prompt_stream(train_rows)
 sft,tok=vec_rl.load_sft(device,torch.float32); base_eval=endpoint(sft,tok,eval_rows_global,device,args.bench_n,args.max_new_tokens)
 edited=copy.deepcopy(sft); edit_info=apply_validated_rank1_edit(edited,tok,train_rows,device,edit_layers=vec_rl.EDIT_LAYERS,top_columns=16,strength=.9)
 canonical=endpoint(edited,tok,eval_rows_global,device,args.bench_n,args.max_new_tokens)
 print(f"  canonical sparse edit exact={canonical['behavior_eval']['exact_compliance_rate']}%",flush=True)
 res={'experiment':'exp144_sparse_edit_anchored_mlp_lora','hypothesis':'H22 ultra-sparse edit + post-edit anchored complementary MLP-only LoRA','config':vars(args),'seeds':SEEDS,'control_seed':CONTROL_SEED,'edit_operator':edit_info,'complement_layers':COMPLEMENT,'baseline_sft':base_eval,'canonical_edit_replication':canonical,'seeded_h22_runs':{},'controls':{},'acceptance_rule':'H22 must preserve/improve canonical behavior across seeds 7/42/123 in exported merged artifact without benchmark regression'}
 comp_re=halo_lora.layer_regex(COMPLEMENT,mlp_only=True); rand_re=halo_lora.layer_regex(RANDOM,mlp_only=True); incl_re=halo_lora.layer_regex(INCLUSIVE,mlp_only=True)
 with (out/'step_metrics.jsonl').open('w') as fh:
  args.step_log=fh
  for seed in SEEDS[:args.seed_count]:
   print(f'[H22 seed {seed}]',flush=True)
   ref=copy.deepcopy(edited).eval(); vec_rl.freeze_weights(ref)
   r=train_lora(f'h22_seed{seed}',edited,ref,tok,prompts,device,comp_re,args,seed)
   r['exported_behavior_delta_vs_canonical']=delta(r['exported_eval']['behavior_eval'],canonical['behavior_eval']); r['exported_benchmark_delta_vs_canonical']=delta(r['exported_eval']['benchmark'],canonical['benchmark']); res['seeded_h22_runs'][str(seed)]=r
   del r
   if device.type=='cuda': torch.cuda.empty_cache()
  # Controls at seed 42: isolate edit contribution, layer specificity, edited-layer inclusion, and vector lower bound.
  seed=CONTROL_SEED; ref_edit=copy.deepcopy(edited).eval(); vec_rl.freeze_weights(ref_edit); ref_sft=copy.deepcopy(sft).eval(); vec_rl.freeze_weights(ref_sft)
  specs=[('lora_only_no_edit_control',sft,ref_sft,comp_re),('random_layer_mlp_control',edited,ref_edit,rand_re),('edited_layer_inclusion_negative_control',edited,ref_edit,incl_re)]
  for name,base,ref,regex in specs[:min(args.control_count,3)]:
   print(f'[Control] {name}',flush=True); rr=train_lora(name,base,ref,tok,prompts,device,regex,args,seed); rr['exported_behavior_delta_vs_canonical']=delta(rr['exported_eval']['behavior_eval'],canonical['behavior_eval']); res['controls'][name]=rr; del rr
  if args.control_count>=4:
   print('[Control] vector_only_post_edit_lower_bound',flush=True)
   vm,vs=vec_rl.train_vector_arm('vector_only_post_edit_lower_bound',edited,ref_edit,tok,prompts,device,vec_rl.DISJOINT_LAYERS,args.steps,args.group_size,2e-2,args.beta,seed,max_new_tokens=args.max_new_tokens)
   vv=endpoint(vm,tok,eval_rows_global,device,args.bench_n,args.max_new_tokens); res['controls']['vector_only_post_edit_lower_bound']={'train_stats':vs,'nonexportable_session_control_eval':vv}
 res['wall_sec']=round(time.time()-t0,2); write_json(out/'sparse_edit_anchored_mlp_lora_summary.json',res); print(f"[Exp 144 DONE] {res['wall_sec']}s",flush=True)
if __name__=='__main__': main()
