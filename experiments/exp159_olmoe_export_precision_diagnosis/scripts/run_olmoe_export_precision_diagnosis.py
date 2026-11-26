#!/usr/bin/env python3
"""Exp159: diagnose whether exp157 export mismatch is algebra or bf16 amplification.

Reuses exp154 checkpoints and exp157 merge machinery. For each arm, compares pre-merge
adapter execution with merged-weight execution in bf16 and fp32, both at full-logit level
and isolated expert-layer level. If fp32 is near-exact while bf16 is not, the export
algebra is correct and the artifact is precision amplification; if fp32 also differs,
the merge implementation/orientation is wrong. No training is performed.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, shutil, sys, time
from pathlib import Path
import torch, transformers

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"experiments/edit_then_target_rl/scripts"))
sys.path.insert(0,str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import json_safe, write_json  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
exp154=load_module('exp154mod',ROOT/'experiments/exp154_olmoe_benchmark_selective_expert_adapters/scripts/run_olmoe_expert_adapters.py')
exp157=load_module('exp157mod',ROOT/'experiments/exp157_olmoe_expert_adapter_export_validation/scripts/run_olmoe_expert_adapter_export_validation.py')
EXP_DIR=ROOT/'experiments/exp159_olmoe_export_precision_diagnosis';RESULTS_DIR=EXP_DIR/'results';ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'
ARMS=exp157.ARMS

def smoke():return os.environ.get('HALO_SMOKE_TEST')=='1'
def append(rec):
 RESULTS_DIR.mkdir(parents=True,exist_ok=True)
 with ARM_JSONL.open('a') as f:f.write(json.dumps(json_safe(rec),sort_keys=True)+'\n')
def done():
 if not ARM_JSONL.exists():return set()
 return {json.loads(x)['arm'] for x in ARM_JSONL.read_text().splitlines() if x.strip()}

def load_model_dtype(tok,device,dtype):
 if smoke():return exp154.load_model(tok,device)
 return AutoModelForCausalLM.from_pretrained(exp154.MODEL_NAME,dtype=dtype,attn_implementation='sdpa').to(device)

def encode_probe(tok,rows,device,max_len):
 texts=[(r['context'].strip()+' '+r['answer_text'].strip())[:1200] for r in rows]
 enc=[tokenizer_out for tokenizer_out in [tok(t,add_special_tokens=False,truncation=True,max_length=max_len)['input_ids'] for t in texts]]
 width=max(len(x) for x in enc);pad=int(tok.pad_token_id)
 return {'input_ids':torch.tensor([x+[pad]*(width-len(x)) for x in enc],device=device),'attention_mask':torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in enc],device=device)}

@torch.no_grad()
def capture_probe(model,batch,layer_ids):
 captured={};handles=[]
 def make(li):
  def hook(_m,args,out):captured[li]=(tuple(x.detach().clone() for x in args),out.detach().clone())
  return hook
 for li in layer_ids:handles.append(model.model.layers[li].mlp.experts.register_forward_hook(make(li)))
 try:logits=model(**batch).logits.detach().float().cpu()
 finally:
  for h in handles:h.remove()
 return logits,captured

@torch.no_grad()
def compare_dtype(model,bank,batch,arm):
 layer_ids=sorted(set([0,len(model.model.layers)-1]))
 pre_logits,pre_capture=capture_probe(model,batch,layer_ids)
 merge_info=exp157.merge_bank_into_experts(model,bank)
 post_logits,post_capture=capture_probe(model,batch,layer_ids)
 layer_out={}
 for li in layer_ids:
  args,pre_out=pre_capture[li]
  post_direct=model.model.layers[li].mlp.experts(*args)
  diff=(pre_out-post_direct).abs()
  layer_out[str(li)]={'max_abs_diff':round(float(diff.max().item()),8),'mean_abs_diff':round(float(diff.mean().item()),10)}
 ld=(pre_logits-post_logits).abs()
 return {'merge_info':merge_info,'full_logits':{'max_abs_diff':round(float(ld.max().item()),8),'mean_abs_diff':round(float(ld.mean().item()),10)},'isolated_expert_outputs':layer_out}

def make_smoke_checkpoints(root,tok):
 if root.exists():shutil.rmtree(root)
 exp157.make_smoke_checkpoints(root,tok,torch.device('cpu'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--probe-rows',type=int,default=2);ap.add_argument('--max-len',type=int,default=256);args=ap.parse_args()
 t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 print(f'[Exp159] export precision diagnosis on {device}; transformers={transformers.__version__} torch={torch.__version__}',flush=True)
 tok=exp154.load_tokenizer();ckpt_base=ROOT/'experiments/exp154_olmoe_benchmark_selective_expert_adapters'/('checkpoints_smoke' if smoke() else 'checkpoints')
 if smoke():make_smoke_checkpoints(ckpt_base,tok)
 rows=[json.loads(x) for x in (ROOT/'data/capability_validation_scaled.jsonl').read_text().splitlines() if x.strip()][:(1 if smoke() else args.probe_rows)]
 batch=encode_probe(tok,rows,device,args.max_len)
 finished=done();missing=[]
 for arm in ARMS:
  if arm in finished:continue
  ckpt=ckpt_base/arm/'final_adapter.pt'
  if not ckpt.exists():missing.append(arm);continue
  state=torch.load(ckpt,map_location='cpu',weights_only=False)
  n_layers,n_experts,hidden,intermediate,rank=exp157.infer_bank_geometry(state);selections=exp157.infer_selections(state,n_layers)
  variants={}
  for label,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
   if device.type!='cuda' and label=='bf16':dtype=torch.float32
   model=load_model_dtype(tok,device,dtype);model.eval()
   bank=exp154.ExpertAdapterBank(n_layers,n_experts,hidden,intermediate,rank,selections,16.0,device,next(model.parameters()).dtype).to(device)
   bank.load_state_dict(state['bank']);exp154.install_adapters(model,bank)
   variants[label]=compare_dtype(model,bank,batch,arm)
   del model,bank
   if device.type=='cuda':torch.cuda.empty_cache()
  bf=variants['bf16']['full_logits']['max_abs_diff'];fp=variants['fp32']['full_logits']['max_abs_diff']
  diagnosis='precision_amplification' if fp < max(0.01,bf*0.1) else 'algebra_or_implementation_mismatch'
  rec={'arm':arm,'variants':variants,'diagnosis':diagnosis,'checkpoint_step':int(state.get('step',0)),'smoke_test':smoke(),'wall_sec':round(time.time()-t0,2)}
  append(rec);print(f'  {arm}: bf16={bf} fp32={fp} diagnosis={diagnosis}',flush=True)
 if missing:
  # A dependency-missing run is a failed diagnostic, not a completed null result:
  # valid JSON here would permanently satisfy the autoqueue completion scan.
  crash_dir=RESULTS_DIR/'crashes';crash_dir.mkdir(parents=True,exist_ok=True)
  (crash_dir/'dependency_missing.log').write_text('missing exp154 final_adapter.pt dependency arms: '+','.join(missing)+'\n')
  raise RuntimeError('missing exp154 dependency checkpoints: '+','.join(missing))
 arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()] if ARM_JSONL.exists() else []
 write_json(RESULTS_DIR/'olmoe_export_precision_summary.json',{'experiment':'exp159_olmoe_export_precision_diagnosis','question':'Is exp157 export mismatch algebraic or bf16 numerical amplification?','missing_dependency_arms':missing,'arms':arms,'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()})
 if smoke():shutil.rmtree(ckpt_base,ignore_errors=True)
 print(f'[DONE] exp159 arms={len(arms)} missing={len(missing)} wall={round(time.time()-t0,2)}s',flush=True)
if __name__=='__main__':main()
