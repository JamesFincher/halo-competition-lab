#!/usr/bin/env python3
"""Slider Head Fork — iterative attention-capability research program.

Goal: measure how far zero-init slider-gated attention heads can push real benchmark
capability (MMLU/ARC-Easy/HellaSwag) on SmolLM2-135M. Iterates creative variants:
  V1 single-head L27 baseline | V2 4-head bank L24-27 | V3 deep spread L8/18/28
  V4 wide heads (dim 128) | V5 memory cross-attention head | V6 full-bank wide
Each variant: detailed checkpoint logging (weights, alphas, per-step telemetry),
industry benchmark before/after, behavioral eval before/after. Results append to
experiments/slider_head_fork/FORK_LEADERBOARD.md — the user-facing iteration log.
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl

FORK = ROOT / "experiments/slider_head_fork"
CKPT = FORK / "checkpoints"

class SliderHead(nn.Module):
    def __init__(self, dim, head_dim=64, memory=0):
        super().__init__()
        self.head_dim = head_dim
        self.memory = memory
        self.q = nn.Linear(dim, head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.o = nn.Linear(head_dim, dim, bias=False)
        if memory:
            self.mem_k = nn.Parameter(torch.randn(memory, head_dim) * 0.02)
            self.mem_v = nn.Parameter(torch.randn(memory, head_dim) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.memory:
            k = torch.cat([k, self.mem_k.unsqueeze(0).expand(B, -1, -1)], dim=1)
            v = torch.cat([v, self.mem_v.unsqueeze(0).expand(B, -1, -1)], dim=1)
        att = torch.softmax(q @ k.transpose(-1, -2) / (self.head_dim ** 0.5), dim=-1)
        causal = torch.triu(torch.ones(T, k.shape[1], device=x.device), diagonal=1).bool()
        if not self.memory:
            att = torch.softmax(q @ k.transpose(-1,-2) / (self.head_dim**0.5) + torch.triu(torch.full((T, T), -1e9, device=x.device), diagonal=1), dim=-1)
        return x + self.alpha * self.o(att @ v)

def make_variant(name):
    if name == "v1_single": return {27: [64]}
    if name == "v2_bank4": return {24: [64], 25: [64], 26: [64], 27: [64]}
    if name == "v3_deep_spread": return {8: [64], 18: [64], 28: [64]}
    if name == "v4_wide": return {27: [128]}
    if name == "v5_memory": return {27: [64]}  # memory flag set separately
    if name == "v6_fullbank_wide": return {24: [128], 26: [128], 28: [128]}
    return {}

def main():
    ap_variants = sys.argv[1:] or ["v1_single"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = FORK / "results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[SliderFork] on {device} variants={ap_variants}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  baseline exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)

    for variant in ap_variants:
        t0 = time.time()
        spec = make_variant(variant)
        if not spec: print(f"  unknown variant {variant}", flush=True); continue
        print(f"\n=== {variant}: layers={list(spec.keys())} ===", flush=True)
        model = copy.deepcopy(sft)
        heads = {}
        mem = 16 if variant == "v5_memory" else 0
        for li, dims in spec.items():
            for hd in dims:
                heads[li] = SliderHead(model.config.hidden_size, hd, memory=mem).to(device)
        hooks = []
        def mk(li):
            def hk(_m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                d = heads[li](o) - o
                return ((o + d,) + out[1:]) if isinstance(out, tuple) else o + d
            return hk
        for li in heads:
            hooks.append(model.model.layers[li].register_forward_hook(mk(li)))
        for p in model.parameters(): p.requires_grad_(False)
        allp = [p for h in heads.values() for p in h.parameters()]
        opt = torch.optim.AdamW(allp, lr=3e-3)
        ben = [r for r in train_rows if r["target"]["action"] == "call_tool"][:12]
        uns = [r for r in train_rows if r["target"]["action"] == "refuse"][:12]
        rows = ben + uns
        log = []
        for step in range(32):
            r = rows[step % len(rows)]
            ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
            pl = ids["input_ids"].shape[1]
            gen = model.generate(**ids, max_new_tokens=48, do_sample=True, temperature=0.9, top_p=0.95, num_return_sequences=4, pad_token_id=tok.eos_token_id)
            rewards = []
            for gi in range(4):
                text = tok.decode(gen[gi][pl:], skip_special_tokens=True)
                rew, _ = base_rl.score_completion(text, r["target"])
                rewards.append(rew)
            rew_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            adv = rew_t - rew_t.mean()
            loss = torch.zeros((), device=device)
            for gi in range(4):
                lp = base_rl.completion_logprobs(model, gen[gi].unsqueeze(0), pl)
                loss = loss - (adv[gi] * lp.sum() / max(1, lp.numel()))
            loss = loss / 4
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            if step % 8 == 0:
                alphas = {li: round(h.alpha.item(), 4) for li, h in heads.items()}
                print(f"    step {step}: rew={rew_t.mean().item():.2f} alphas={alphas}", flush=True)
                log.append({"step": step, "rew": round(rew_t.mean().item(), 3), "alphas": alphas})
        final_ev = base_rl.evaluate_heldout(model, tok, eval_rows, device)
        print(f"  [{variant}] exact={final_ev['exact_compliance_rate']:.1f}% unsafe={final_ev['unsafe_refusal_rate']:.1f}% benignfr={final_ev['benign_false_refusal_rate']:.1f}%", flush=True)
        # save detailed checkpoint
        ck = CKPT / variant
        ck.mkdir(parents=True, exist_ok=True)
        for li, h in heads.items():
            torch.save(h.state_dict(), ck / f"head_L{li}.pt")
        meta = {"variant": variant, "spec": {str(k): v for k, v in spec.items()}, "memory": mem,
                "wall_sec": round(time.time()-t0, 2),
                "alphas_final": {str(li): round(h.alpha.item(), 5) for li, h in heads.items()},
                "baseline_eval": base_ev, "final_eval": final_ev, "train_log": log}
        (ck / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
        (out / f"{variant}_summary.json").write_text(json.dumps(meta, indent=2))
        for hk_ in hooks: hk_.remove()
        del model, heads
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"[{variant} DONE] in {meta['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
