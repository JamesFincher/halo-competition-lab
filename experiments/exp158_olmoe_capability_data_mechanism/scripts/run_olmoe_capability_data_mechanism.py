#!/usr/bin/env python3
"""Exp158: OLMoE capability-data mechanism ablation.

Holds the exp154 random-expert adapter surface fixed and varies only the training
mixture: ARC-only, MMLU-only, HellaSwag-only, or shuffled-answer control. This tests
whether generic expert-LoRA gains are source-specific supervision or nonspecific
answer-only continuation/regularization. Base weights and routers are frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import transformers

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import json_safe, write_json  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark, run_transfer_benchmark  # noqa: E402

MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"
EXP_DIR = ROOT / "experiments/exp158_olmoe_capability_data_mechanism"
RESULTS_DIR = EXP_DIR / "results"
ARM_JSONL = RESULTS_DIR / "arm_results.jsonl"


def smoke() -> bool:
    return os.environ.get("HALO_SMOKE_TEST") == "1"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(tokenizer, device):
    if smoke():
        cfg = OlmoeConfig(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2,
            max_position_embeddings=384, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        model = OlmoeForCausalLM(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, attn_implementation="sdpa"
        )
    return model.to(device)


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ExpertAdapterBank(torch.nn.Module):
    """Per-expert down-projection LoRA deltas. Input I -> rank r -> output H."""

    def __init__(self, n_layers: int, n_experts: int, hidden: int, intermediate: int, rank: int, selections: dict[int, list[int]], alpha: float, device, dtype):
        super().__init__()
        self.n_layers, self.n_experts, self.rank, self.alpha = n_layers, n_experts, rank, alpha
        self.selections = {int(k): sorted(int(x) for x in v) for k, v in selections.items()}
        self.a = torch.nn.Parameter(torch.empty((n_layers, n_experts, rank, intermediate), device=device, dtype=dtype))
        self.b = torch.nn.Parameter(torch.zeros((n_layers, n_experts, hidden, rank), device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.a, a=5**0.5)
        with torch.no_grad():
            for layer in range(n_layers):
                keep = torch.zeros((n_experts, 1, 1), device=device, dtype=dtype)
                keep[self.selections[layer]] = 1.0
                self.register_buffer(f"keep_{layer}", keep)

    def delta(self, layer: int, expert: int, intermediate_activation: torch.Tensor) -> torch.Tensor:
        a = self.a[layer, expert]
        b = self.b[layer, expert]
        out = F.linear(F.linear(intermediate_activation, a), b) * (self.alpha / max(self.rank, 1))
        keep = getattr(self, f"keep_{layer}")[expert]
        return out * keep


def experts_forward_with_adapters(experts, hidden_states, top_k_index, top_k_weights, bank: ExpertAdapterBank, layer: int):
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=experts.num_experts + 1).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = int(expert_idx[0])
        if expert_idx == experts.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        inter = experts.act_fn(gate) * up
        base = F.linear(inter, experts.down_proj[expert_idx])
        adapted = base + bank.delta(layer, expert_idx, inter)
        adapted = adapted * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, adapted.to(final_hidden_states.dtype))
    return final_hidden_states


def install_adapters(model, bank: ExpertAdapterBank) -> None:
    for layer_idx, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        def forward(hidden_states, top_k_index, top_k_weights, layer_idx=layer_idx, experts=experts):
            return experts_forward_with_adapters(experts, hidden_states, top_k_index, top_k_weights, bank, layer_idx)
        experts.forward = forward


@torch.no_grad()
def cache_consistency(model, tokenizer, device) -> dict:
    prompt = base_rl.build_prompt(tokenizer, [{"role": "system", "content": "You are concise."}, {"role": "user", "content": "Say hello."}])
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    ids_c = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    ids_u = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=False)
    logits_c = model(**enc, use_cache=True).logits
    logits_u = model(**enc, use_cache=False).logits
    return {"generated_ids_equal": bool(torch.equal(ids_c.cpu(), ids_u.cpu())), "single_forward_max_abs_logit_diff": round(float((logits_c - logits_u).abs().max().item()), 8), "claim": "expert output adapters are token-local weight-like operations, so cached generation is valid"}


@torch.no_grad()
def profile_answer_experts(model, tokenizer, rows: list[dict], device, top_n: int, max_rows: int) -> dict[int, list[int]]:
    caps = [[] for _ in model.model.layers]
    hooks = []
    def mk(idx):
        def hook(_m, _i, output):
            caps[idx].append(output[2].detach().cpu())
        return hook
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(mk(i)))
    try:
        counts = [torch.zeros(int(model.config.num_experts), dtype=torch.float64) for _ in model.model.layers]
        for row in rows[:max_rows]:
            context = row["context"].strip() + " "
            text = context + row["answer_text"].strip()
            prefix_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=384, add_special_tokens=False).to(device)
            before = [len(x) for x in caps]
            _ = model(**enc)
            positions = torch.arange(int(enc["input_ids"].shape[1])) >= max(prefix_len - 1, 0)
            for li in range(len(counts)):
                if len(caps[li]) == before[li]:
                    continue
                idx = caps[li][-1][: int(enc["input_ids"].shape[1])][positions].reshape(-1)
                counts[li] += torch.bincount(idx, minlength=len(counts[li])).double()
        k = min(top_n, len(counts[0]))
        return {li: [int(x) for x in torch.topk(c, k=k).indices.tolist()] for li, c in enumerate(counts)}
    finally:
        for h in hooks:
            h.remove()


def capability_example(tokenizer, row: dict, max_len: int) -> dict:
    cids = tokenizer(row["context"].strip() + " ", add_special_tokens=False)["input_ids"]
    aids = tokenizer(row["answer_text"].strip(), add_special_tokens=False)["input_ids"]
    ids = (cids + aids)[:max_len]
    labels = ([-100] * len(cids) + aids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def collate(examples, pad_id, device):
    width = max(len(x["input_ids"]) for x in examples)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad_id] * (width - len(x["input_ids"])) for x in examples], device=device),
        "labels": torch.tensor([x["labels"] + [-100] * (width - len(x["input_ids"])) for x in examples], device=device),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (width - len(x["input_ids"])) for x in examples], device=device),
    }


def save_adapter_checkpoint(bank, optimizer, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"step": step, "bank": bank.state_dict(), "optimizer": optimizer.state_dict()}, tmp)
    tmp.replace(path)


def latest_adapter_checkpoint(root: Path) -> tuple[Path | None, int]:
    candidates = sorted(root.glob("checkpoint-step-*.pt"), key=lambda x: int(x.stem.rsplit("-", 1)[1]) if x.stem.rsplit("-", 1)[1].isdigit() else -1)
    if not candidates:
        return None, 0
    try:
        return candidates[-1], int(torch.load(candidates[-1], map_location="cpu", weights_only=False)["step"])
    except Exception:
        return None, 0


def train_bank(model, bank, tokenizer, rows, device, steps, batch_size, max_len, lr, checkpoint_root: Path, checkpoint_interval: int) -> dict:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in bank.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=0.0)
    final_ckpt = checkpoint_root / "final_adapter.pt"
    ckpt, resume_step = latest_adapter_checkpoint(checkpoint_root)
    start_step = 0
    load_path = final_ckpt if final_ckpt.exists() else ckpt
    if load_path is not None and load_path.exists():
        state = torch.load(load_path, map_location="cpu", weights_only=False)
        bank.load_state_dict(state["bank"])
        start_step = steps if final_ckpt.exists() else int(state.get("step", resume_step))
        if not final_ckpt.exists() and "optimizer" in state:
            try:
                opt.load_state_dict(state["optimizer"])
                for state_values in opt.state.values():
                    for key, value in list(state_values.items()):
                        if torch.is_tensor(value):
                            state_values[key] = value.to(device)
            except Exception as exc:
                print(f"    optimizer resume skipped: {type(exc).__name__}: {exc}", flush=True)
    model.config.use_cache = False
    model.train(); bank.train()
    log=[]; t0=time.time()
    for step in range(start_step, steps):
        batch_rows=[rows[(step*batch_size+i)%len(rows)] for i in range(batch_size)]
        batch=collate([capability_example(tokenizer,r,max_len) for r in batch_rows],int(tokenizer.pad_token_id),device)
        opt.zero_grad(set_to_none=True)
        loss=model(**batch).loss
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.0).item())
        opt.step()
        if (step + 1) % checkpoint_interval == 0 and step + 1 < steps:
            save_adapter_checkpoint(bank, opt, step + 1, checkpoint_root / f"checkpoint-step-{step + 1}.pt")
        if step%10==0 or step==steps-1:
            rec={"step":step+1,"loss":round(float(loss.detach().item()),5),"grad_norm":round(grad,6)}
            log.append(rec); print(f"    step {step+1}/{steps}: loss={rec['loss']:.3f} grad={grad:.2f}",flush=True)
    if steps > 0:
        save_adapter_checkpoint(bank, opt, steps, final_ckpt)
    model.config.use_cache=True; model.eval(); bank.eval()
    return {"steps":steps,"resumed_from_step":start_step,"checkpointed":True,"train_wall_sec":round(time.time()-t0,2),"train_log_tail":log[-8:],"allocated_adapter_params":sum(p.numel() for p in bank.parameters() if p.requires_grad),"effective_selected_adapter_params":sum((bank.rank * bank.a.shape[-1] + bank.b.shape[-2] * bank.rank) * len(bank.selections[li]) for li in range(bank.n_layers))}


def completed_arms():
    if not ARM_JSONL.exists(): return set()
    out=set()
    for line in ARM_JSONL.read_text().splitlines():
        if line.strip():
            try: out.add(json.loads(line)["arm"])
            except Exception: pass
    return out


def append_arm(record):
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    with ARM_JSONL.open("a") as f: f.write(json.dumps(json_safe(record),sort_keys=True)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=240)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--max-len",type=int,default=384)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--rank",type=int,default=8)
    ap.add_argument("--adapter-alpha",type=float,default=16.0)
    ap.add_argument("--top-experts-per-layer",type=int,default=8)
    ap.add_argument("--profile-rows",type=int,default=64)
    ap.add_argument("--continuation-n",type=int,default=100)
    ap.add_argument("--max-new-tokens",type=int,default=72)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--checkpoint-interval",type=int,default=100)
    args=ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    t0=time.time(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp158] OLMoE capability-data mechanism on {device}; transformers={transformers.__version__} torch={torch.__version__}",flush=True)
    tok=load_tokenizer(); probe_model=load_model(tok,device); probe_model.eval()
    cap_train=jsonl_rows(ROOT/"data/capability_train_scaled.jsonl")
    cap_val=jsonl_rows(ROOT/"data/capability_validation_scaled.jsonl")
    if smoke(): cap_train=cap_train[:64]; cap_val=cap_val[:2]
    rng=random.Random(args.seed); rng.shuffle(cap_train)
    selective=profile_answer_experts(probe_model,tok,cap_val,device,args.top_experts_per_layer,args.profile_rows)
    n_layers=len(probe_model.model.layers); n_exp=int(probe_model.config.num_experts)
    rng=random.Random(args.seed+1)
    random_sel={li:sorted(rng.sample(range(n_exp),min(args.top_experts_per_layer,n_exp))) for li in range(n_layers)}
    diffuse={li:list(range(n_exp)) for li in range(n_layers)}
    behavior_eval=base_rl.load_rows(ROOT/"data/policy_eval_scaled_v2.jsonl")
    baseline={
      "behavior_scaled120":base_rl.evaluate_heldout(probe_model,tok,behavior_eval,device,max_new_tokens=args.max_new_tokens,use_cache=True),
      "continuation_benchmark":run_continuation_benchmark(probe_model,tok,device,args.continuation_n),
      "transfer_benchmark":run_transfer_benchmark(probe_model,tok,device,args.continuation_n // 2),
      "cache_consistency":cache_consistency(probe_model,tok,device),
    }
    del probe_model
    if device.type=="cuda": torch.cuda.empty_cache()
    hidden=2048 if not smoke() else 32; intermediate=1024 if not smoke() else 16
    arm_specs=[]
    for name,source in [
      ("random_experts_arc_only","arc_easy_train"),
      ("random_experts_mmlu_only","mmlu_validation"),
      ("random_experts_hellaswag_only","hellaswag_train"),
      ("random_experts_shuffled_answer_control","shuffled_answer_control"),
    ]:
      arm_specs.append((name,random_sel,args.rank,source))
    done=completed_arms()
    ckpt_base=EXP_DIR/("checkpoints_smoke" if smoke() else "checkpoints")
    if smoke() and ckpt_base.exists(): shutil.rmtree(ckpt_base)
    for arm,selections,rank,source in arm_specs:
      if arm in done: continue
      arm_t0=time.time(); print(f"[Exp158] arm {arm}",flush=True)
      # Hold adapter initialization fixed across data arms so the training mixture is
      # the single intended causal variable; seed sensitivity needs a separate replication.
      random.seed(args.seed); torch.manual_seed(args.seed)
      model=load_model(tok,device); model.eval()
      dtype=next(model.parameters()).dtype
      bank=ExpertAdapterBank(n_layers,n_exp,hidden,intermediate,rank,selections,args.adapter_alpha,device,dtype).to(device)
      install_adapters(model,bank)
      source_names=("arc_easy_train","mmlu_validation","hellaswag_train")
      source_counts={src:sum(1 for r in cap_train if r.get("source")==src) for src in source_names}
      if source == "shuffled_answer_control":
        per_source=max(1,min(source_counts.values())//len(source_names))
        grouped={src:[r for r in cap_train if r.get("source")==src][:per_source] for src in source_names}
        balanced=[]
        for i in range(per_source):
          for src in source_names:
            if i < len(grouped[src]): balanced.append(grouped[src][i])
        train_rows=[]
        for i,r in enumerate(balanced):
          rr=dict(r); rr["answer_text"]=balanced[(i+997)%len(balanced)]["answer_text"]; train_rows.append(rr)
      else:
        train_rows=[r for r in cap_train if r.get("source")==source]
        train_rows=train_rows[:min(source_counts.values())]
      train_info=train_bank(model,bank,tok,train_rows,device,args.steps,args.batch_size,args.max_len,args.lr,ckpt_base / arm,args.checkpoint_interval)
      train_info["training_source"]=source; train_info["training_rows"]=len(train_rows)
      behavior=base_rl.evaluate_heldout(model,tok,behavior_eval,device,max_new_tokens=args.max_new_tokens,use_cache=True)
      continuation=run_continuation_benchmark(model,tok,device,args.continuation_n)
      transfer=run_transfer_benchmark(model,tok,device,args.continuation_n // 2)
      cache=cache_consistency(model,tok,device)
      rec={"arm":arm,"selections":selections if smoke() else {str(k):v for k,v in selections.items()},"rank_per_selected_expert":rank,"parameter_matching":"selective/random use top_n*rank; diffuse uses num_experts*1 so total adapter parameters match when top_n*rank=num_experts","train_info":train_info,"behavior_scaled120":behavior,"continuation_benchmark":continuation,"transfer_benchmark":transfer,"baseline":baseline,"cache_consistency_after_training":cache,"wall_sec":round(time.time()-arm_t0,2),"smoke_test":smoke()}
      append_arm(rec)
      del model,bank
      if device.type=="cuda": torch.cuda.empty_cache()
    arms=[json.loads(x) for x in ARM_JSONL.read_text().splitlines() if x.strip()]
    write_json(RESULTS_DIR/"olmoe_capability_data_mechanism_summary.json",{"experiment":"exp158_olmoe_capability_data_mechanism","model":MODEL_NAME,"question":"Does generic random-expert capability improvement come from source-specific supervision or answer-only continuation structure rather than expert selectivity, with adapter initialization held fixed across data arms?","baseline":baseline,"arms":arms,"wall_sec":round(time.time()-t0,2),"smoke_test":smoke()})
    if smoke(): shutil.rmtree(ckpt_base,ignore_errors=True)
    print(f"[DONE] exp158 arms={len(arms)} wall={round(time.time()-t0,2)}s",flush=True)

if __name__=="__main__": main()
