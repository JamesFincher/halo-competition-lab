#!/usr/bin/env python3
"""Exp166: Qwen3-4B mixed-batch versus alternating capability replay.

At the exp152 reference 25% capability-example dose, tests whether the behavior cost of
capability replay is caused by homogeneous-batch alternation rather than capability
exposure itself. The alternating reference uses exp152's every-fourth-step replay. The
mixed arm places behavior and capability examples in every batch with exactly matched
behavior/capability example counts across the full run. Measurement is behavior v2 plus
in-domain and held-out transfer continuation benchmarks.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, os, random, shutil, sys, time
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
EXP_DIR=ROOT/'experiments/exp166_qwen3_mixed_batch_interleaving'
RESULTS_DIR=EXP_DIR/'results'
ARM_JSONL=RESULTS_DIR/'arm_results.jsonl'


def smoke():return os.environ.get('HALO_SMOKE_TEST')=='1'


def append(rec):
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    with ARM_JSONL.open('a') as f:f.write(json.dumps(json_safe(rec),sort_keys=True)+'\n')


def completed_arms():
    if not ARM_JSONL.exists():return set()
    out=set()
    for line in ARM_JSONL.read_text().splitlines():
        if line.strip():
            try:out.add(json.loads(line)['arm'])
            except Exception:pass
    return out


def alternating_example_sequences(steps,batch_size,replay_every,behavior_rows,capability_rows):
    """Example streams exactly as exp152-style homogeneous alternating replay consumes them."""
    behavior_seq=[];capability_seq=[]
    for step in range(steps):
        is_capability=step%replay_every==replay_every-1
        source=capability_rows if is_capability else behavior_rows
        batch=exp152.cycled_batch(source,step,batch_size)
        if is_capability:capability_seq.extend(batch)
        else:behavior_seq.extend(batch)
    return behavior_seq,capability_seq


def train_mixed_lora(model,tokenizer,behavior_rows,capability_rows,device,checkpoint_root,steps,batch_size,cap_per_batch,max_len,lr,warmup,checkpoint_interval):
    """Multitask mixed batches with exact behavior/capability example-count control."""
    final_adapter=checkpoint_root/'final_adapter'
    ckpt,resume_step=exp152.latest_checkpoint(checkpoint_root)
    start_step=steps if final_adapter.exists() else (resume_step if ckpt is not None else 0)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.0)
    if ckpt is not None and not final_adapter.exists() and (ckpt/'trainer_state.pt').exists():
        try:
            state=torch.load(ckpt/'trainer_state.pt',map_location='cpu',weights_only=False)
            optimizer.load_state_dict(state['optimizer'])
            for state_values in optimizer.state.values():
                for key,value in list(state_values.items()):
                    if torch.is_tensor(value):state_values[key]=value.to(device)
        except Exception as exc:
            print(f'  optimizer resume skipped: {type(exc).__name__}: {exc}',flush=True)
    if device.type=='cuda':
        model.gradient_checkpointing_enable();model.enable_input_require_grads()
    model.config.use_cache=False;model.train()
    log=[];train_t0=time.time();behavior_per_batch=batch_size-cap_per_batch
    for step in range(start_step,steps):
        raw_behavior=[behavior_rows[(step*behavior_per_batch+i)%len(behavior_rows)] for i in range(behavior_per_batch)]
        raw_capability=[capability_rows[(step*cap_per_batch+i)%len(capability_rows)] for i in range(cap_per_batch)]
        examples=[exp152.behavior_example(tokenizer,r,max_len) for r in raw_behavior]+[exp152.capability_example(tokenizer,r,max_len) for r in raw_capability]
        batch=exp152.collate(examples,int(tokenizer.pad_token_id),device)
        progress=min(1.0,(step+1)/max(steps,1))
        lr_now=lr*min(1.0,(step+1)/max(warmup,1))*(0.1+0.9*0.5*(1.0+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr_now
        optimizer.zero_grad(set_to_none=True)
        loss=model(**batch).loss
        loss.backward()
        grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1.0).item())
        optimizer.step()
        if step%10==0 or step==steps-1:
            rec={'step':step+1,'loss':round(float(loss.detach().item()),5),'lr':lr_now,'grad_norm':round(grad_norm,4),'batch_type':'mixed_behavior_capability'}
            log.append(rec);print(f'    step {step+1}/{steps}: loss={rec["loss"]:.3f} type=mixed grad={rec["grad_norm"]:.2f}',flush=True)
        if (step+1)%checkpoint_interval==0 and step+1<steps:
            exp152.save_checkpoint(model,optimizer,step+1,checkpoint_root/f'checkpoint-step-{step+1}')
    if steps>0:exp152.save_checkpoint(model,optimizer,steps,final_adapter)
    model.config.use_cache=True;model.eval()
    return {
        'requested_steps':steps,
        'resumed_from_step':start_step,
        'optimizer_steps_completed':steps,
        'train_wall_sec':round(time.time()-train_t0,2),
        'train_log_tail':log[-8:],
        'trainable_params':sum(p.numel() for p in params),
        'interleaving':'mixed_batch',
        'behavior_examples':steps*behavior_per_batch,
        'capability_examples':steps*cap_per_batch,
        'cap_per_batch':cap_per_batch,
        'behavior_per_batch':behavior_per_batch,
    }


def measure(model,tok,eval_rows,device,max_new_tokens,continuation_n):
    return {
        'behavior_scaled120':base_rl.evaluate_heldout(model,tok,eval_rows,device,max_new_tokens=max_new_tokens,use_cache=True,batch_size=4,include_rows=True),
        'continuation_benchmark':run_continuation_benchmark(model,tok,device,continuation_n,include_rows=True),
        'transfer_benchmark':run_transfer_benchmark(model,tok,device,continuation_n,include_rows=True),
    }


def memory_stats(device):
    if device.type!='cuda':return {'device':'cpu'}
    return {'device':torch.cuda.get_device_name(device),'max_allocated_gb':round(torch.cuda.max_memory_allocated(device)/1e9,2),'max_reserved_gb':round(torch.cuda.max_memory_reserved(device)/1e9,2)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=300);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--cap-per-batch',type=int,default=2);ap.add_argument('--replay-every',type=int,default=4)
    ap.add_argument('--max-len',type=int,default=384);ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--rank',type=int,default=16);ap.add_argument('--lora-alpha',type=int,default=32);ap.add_argument('--lora-dropout',type=float,default=0.05)
    ap.add_argument('--warmup',type=int,default=20);ap.add_argument('--checkpoint-interval',type=int,default=100);ap.add_argument('--max-new-tokens',type=int,default=72);ap.add_argument('--continuation-n',type=int,default=50);ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if smoke() and args.batch_size<=args.cap_per_batch:
        args.batch_size=2;args.cap_per_batch=1
    if not 0<args.cap_per_batch<args.batch_size:raise ValueError('cap-per-batch must be strictly inside batch size')
    t0=time.time();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Exp166] Qwen3 mixed-batch interleaving on {device}',flush=True)
    tok=exp152.load_tokenizer(exp152.MODEL_NAME)
    behavior_rows=base_rl.load_rows(ROOT/'data/policy_train_scaled_v2.jsonl')
    capability_rows=[json.loads(x) for x in (ROOT/'data/capability_train_scaled.jsonl').read_text().splitlines() if x.strip()]
    eval_rows=base_rl.load_rows(ROOT/'data/policy_eval_scaled_v2.jsonl')
    continuation_n=2 if smoke() else args.continuation_n
    ckpt_base=EXP_DIR/('checkpoints_smoke' if smoke() else 'checkpoints')
    if smoke() and ckpt_base.exists():shutil.rmtree(ckpt_base)
    finished=completed_arms()
    if 'baseline_zero_shot' not in finished:
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        arm_t0=time.time();model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
        rec={'arm':'baseline_zero_shot','interleaving':'zero_shot','measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
        append(rec);print(f'  baseline exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}%',flush=True);del model
        if device.type=='cuda':torch.cuda.empty_cache()
    rng=random.Random(args.seed);rng.shuffle(behavior_rows);rng.shuffle(capability_rows)
    alt_behavior_seq,alt_capability_seq=alternating_example_sequences(args.steps,args.batch_size,args.replay_every,behavior_rows,capability_rows)
    expected_mixed_behavior=args.steps*(args.batch_size-args.cap_per_batch)
    expected_mixed_capability=args.steps*args.cap_per_batch
    exact_example_match=(len(alt_behavior_seq)==expected_mixed_behavior and len(alt_capability_seq)==expected_mixed_capability)
    if not exact_example_match and not smoke():
        raise ValueError(f'example-count mismatch: alternating {len(alt_behavior_seq)}/{len(alt_capability_seq)} vs mixed {expected_mixed_behavior}/{expected_mixed_capability}')
    # In smoke or non-default exploratory settings, cycle the alternating streams. At the
    # real default dose they are exact, so only temporal interleaving varies across arms.
    mixed_behavior_rows=alt_behavior_seq or behavior_rows
    mixed_capability_rows=alt_capability_seq or capability_rows
    armspecs=[('alternating_reference_every4','alternating'),('mixed_batch_25pct','mixed')]
    for arm,interleaving in armspecs:
        if arm in finished:continue
        arm_t0=time.time();print(f'[Exp166] arm {arm}',flush=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        random.seed(args.seed);torch.manual_seed(args.seed)
        model=exp152.load_model(exp152.MODEL_NAME,tok,device);model.eval()
        ckpt_root=ckpt_base/arm;final_adapter=ckpt_root/'final_adapter';resume=final_adapter if final_adapter.exists() else exp152.latest_checkpoint(ckpt_root)[0]
        model=exp152.make_lora(model,exp152.ATTN_MLP_TARGETS,None,resume,args.rank,args.lora_alpha,args.lora_dropout)
        if interleaving=='alternating':
            train_info=exp152.train_lora(model,tok,behavior_rows,capability_rows,device,ckpt_root,args.steps,args.batch_size,args.max_len,args.lr,args.warmup,args.checkpoint_interval,True,args.replay_every)
            cap_steps=sum(1 for step in range(args.steps) if step%args.replay_every==args.replay_every-1)
            train_info['interleaving']='alternating_homogeneous_batches';train_info['capability_examples']=cap_steps*args.batch_size;train_info['behavior_examples']=(args.steps-cap_steps)*args.batch_size
        else:
            train_info=train_mixed_lora(model,tok,mixed_behavior_rows,mixed_capability_rows,device,ckpt_root,args.steps,args.batch_size,args.cap_per_batch,args.max_len,args.lr,args.warmup,args.checkpoint_interval)
        rec={'arm':arm,'interleaving':interleaving,'planned_steps':args.steps,'batch_size':args.batch_size,'cap_per_batch':args.cap_per_batch if interleaving=='mixed' else args.batch_size,'replay_every':args.replay_every,'exact_alternating_example_match':exact_example_match,'train_info':train_info,'measurement':measure(model,tok,eval_rows,device,args.max_new_tokens,continuation_n),'memory':memory_stats(device),'wall_sec':round(time.time()-arm_t0,2),'smoke_test':smoke()}
        append(rec);print(f'  {arm}: exact={rec["measurement"]["behavior_scaled120"]["exact_compliance_rate"]}% cap_examples={train_info["capability_examples"]}',flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()] if ARM_JSONL.exists() else []
    write_json(RESULTS_DIR/'qwen3_mixed_batch_interleaving_summary.json',{'experiment':'exp166_qwen3_mixed_batch_interleaving','model':exp152.MODEL_NAME,'question':'At matched 25% capability-example dose, does mixed-batch multitasking preserve behavior better than alternating homogeneous capability/behavior batches while retaining transfer capability?','arms':arms,'behavior_row_statistics':{'versus_baseline_zero_shot':comparisons_against_reference(arms,'baseline_zero_shot'),'versus_alternating_reference':comparisons_against_reference(arms,'alternating_reference_every4')},'capability_row_statistics':{'continuation_versus_alternating_reference':benchmark_comparisons_against_reference(arms,'alternating_reference_every4','continuation_benchmark'),'transfer_versus_alternating_reference':benchmark_comparisons_against_reference(arms,'alternating_reference_every4','transfer_benchmark')},'wall_sec':round(time.time()-t0,2),'smoke_test':smoke()})
    if smoke():shutil.rmtree(ckpt_base,ignore_errors=True)
    print(f'[DONE] exp166 arms={len(arms)} wall={round(time.time()-t0,2)}s',flush=True)


if __name__=='__main__':main()
