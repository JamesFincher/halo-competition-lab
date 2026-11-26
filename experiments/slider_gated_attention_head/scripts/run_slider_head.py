#!/usr/bin/env python3
"""Experiment 42: Zero-Init Slider-Gated Attention Head — additive capacity without core-weight loss.

User insight: encode new attention heads behind a learnable scalar 'slider' initialized at 0,
so the model is bit-identical to baseline at init (zero destruction by construction) and RL
can dial in the head only where it helps. Architecture-matched design: dense models get gated
heads in high-residual layers; MoE generalization (per-expert keyed heads) documented in README.
Trains ONLY slider + head weights, everything else frozen. Before/after on heldout eval.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl

class SliderGatedHead(nn.Module):
    """One extra attention head: full-seq causal self-attention, output scaled by slider alpha.
    alpha inits at 0 -> exact identity passthrough at start. Only alpha + qkv/o are trainable."""
    def __init__(self, dim: int, head_dim: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(dim, head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.o = nn.Linear(head_dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        att = torch.softmax(q @ k.transpose(-1, -2) / (self.head_dim ** 0.5) + torch.triu(torch.full((T, T), -1e9, device=x.device), diagonal=1), dim=-1)
        out = self.o(att @ v)
        return x + self.alpha * out

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/slider_gated_attention_head/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 42] Slider-gated attention head on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)

    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}% unsafe={base_ev['unsafe_refusal_rate']:.1f}% benignfr={base_ev['benign_false_refusal_rate']:.1f}%", flush=True)

    # verify zero-init identity: slider at 0 => outputs identical
    dim = sft.config.hidden_size
    head = SliderGatedHead(dim).to(device)
    ids = tok("test", return_tensors="pt").to(device)
    with torch.no_grad():
        x = sft.model.embed_tokens(ids["input_ids"])
        assert torch.allclose(head(x), x), "zero-init must be exact identity"
    print("  zero-init identity verified (bit-exact passthrough at alpha=0)", flush=True)

    # install hook: apply head after layer 27 resid (high-residual layer per atlas)
    LAYER = 27
    state = {}
    def hook(_m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        d = head(o) - o  # gated delta only
        if isinstance(out, tuple):
            return (o + d,) + out[1:]
        return o + d
    h = sft.model.layers[LAYER].register_forward_hook(hook)

    # freeze everything except head params
    for p in sft.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:12] + [r for r in train_rows if r["target"]["action"] == "refuse"][:12]
    print("  RL: training only slider + head weights (core frozen)...", flush=True)
    head.train()
    log = []
    for step in range(24):
        r = rows[step % len(rows)]
        ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
        prompt_len = ids["input_ids"].shape[1]
        gen = sft.generate(**ids, max_new_tokens=48, do_sample=True, temperature=0.9, top_p=0.95, num_return_sequences=4, pad_token_id=tok.eos_token_id)
        rewards = []
        for gi in range(4):
            text = tok.decode(gen[gi][prompt_len:], skip_special_tokens=True)
            rew, _ = base_rl.score_completion(text, r["target"])
            rewards.append(rew)
        rew_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = rew_t - rew_t.mean()
        loss = torch.zeros((), device=device)
        for gi in range(4):
            seq = gen[gi].unsqueeze(0)
            lp = base_rl.completion_logprobs(sft, seq, prompt_len)
            loss = loss - (adv[gi] * lp.sum() / max(1, lp.numel()))
        loss = loss / 4
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 6 == 0:
            print(f"    step {step}: loss={loss.item():.3f} alpha={head.alpha.item():.4f} mean_rew={rew_t.mean().item():.2f}", flush=True)
        log.append({"step": step, "loss": round(loss.item(), 4), "alpha": round(head.alpha.item(), 5), "mean_rew": round(rew_t.mean().item(), 3)})
    h.remove()

    # eval with trained head still hooked (reinstall)
    h = sft.model.layers[LAYER].register_forward_hook(hook)
    final_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    h.remove()
    print(f"  [slider_head L{LAYER}] exact={final_ev['exact_compliance_rate']:.1f}% unsafe={final_ev['unsafe_refusal_rate']:.1f}% benignfr={final_ev['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.4f}", flush=True)

    summary = {"device": str(device), "wall_sec": round(time.time()-t0, 2), "layer": LAYER,
               "baseline": base_ev, "after": final_ev, "final_alpha": round(head.alpha.item(), 5),
               "delta_exact": round(final_ev["exact_compliance_rate"] - base_ev["exact_compliance_rate"], 2),
               "train_log": log[::4],
               "note": "zero-init verified bit-exact; only slider+head trained, core frozen"}
    (out_dir/"slider_head_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 42 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
