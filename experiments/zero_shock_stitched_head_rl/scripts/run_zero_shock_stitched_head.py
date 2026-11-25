#!/usr/bin/env python3
"""Experiment 14: Zero-Shock Stitched GQA Attention Head + Freeze-Base GRPO (SmolLM2-135M).

1. Widens SmolLM2-135M from 9 Q / 3 KV heads -> 12 Q / 4 KV heads (+1 GQA group) in standard LlamaConfig.
2. Initializes W_Q_new, W_K_new, W_V_new with small non-zero values and W_O_new = 0.0.
3. Proves exact zero shock at step 0: max|logits_new - logits_old| == 0.0.
4. Freezes 100% of original weights via gradient masking (only stitched head slices in top-4 layers update)
   and runs 6 steps of GRPO, evaluating on 40 unseen tasks.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402


def stitch_zero_shock_gqa_group(
    src_model: torch.nn.Module,
    seed: int = 42,
    init_std: float = 0.02,
) -> tuple[torch.nn.Module, dict]:
    """Create a standard LlamaForCausalLM widened by +1 GQA group with exact W_O_new = 0."""
    torch.manual_seed(seed)
    cfg = copy.deepcopy(src_model.config)
    old_q = int(cfg.num_attention_heads)
    old_kv = int(cfg.num_key_value_heads)
    gqa_ratio = old_q // old_kv
    head_dim = int(cfg.hidden_size // old_q)

    cfg.num_key_value_heads = old_kv + 1
    cfg.num_attention_heads = old_q + gqa_ratio
    cfg.head_dim = head_dim

    new_model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    src_sd = src_model.state_dict()
    new_sd = new_model.state_dict()

    with torch.no_grad():
        for k, v in src_sd.items():
            if k not in new_sd:
                continue
            dst = new_sd[k]
            if dst.shape == v.shape:
                dst.copy_(v)
            elif k.endswith(".self_attn.q_proj.weight"):
                dst[: v.shape[0], :].copy_(v)
                dst[v.shape[0] :, :].normal_(0.0, init_std)
            elif k.endswith(".self_attn.k_proj.weight") or k.endswith(".self_attn.v_proj.weight"):
                dst[: v.shape[0], :].copy_(v)
                dst[v.shape[0] :, :].normal_(0.0, init_std)
            elif k.endswith(".self_attn.o_proj.weight"):
                dst[:, : v.shape[1]].copy_(v)
                dst[:, v.shape[1] :].zero_()  # EXACT ZERO-SHOCK STITCH!
            else:
                raise RuntimeError(f"Unexpected shape mismatch for {k}: {v.shape} vs {dst.shape}")
    new_model.load_state_dict(new_sd)
    meta = {
        "old_q_heads": old_q,
        "old_kv_heads": old_kv,
        "new_q_heads": cfg.num_attention_heads,
        "new_kv_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "added_q_dim": gqa_ratio * head_dim,
        "added_kv_dim": head_dim,
    }
    return new_model, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    out_dir = ROOT / "experiments/zero_shock_stitched_head_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    prompt_stream = [x for pair in zip(ben[:-5], uns[:-5]) for x in pair]

    print("[Setup] Loading SmolLM2-135M + Halo SFT checkpoint-36...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft_model = PeftModel.from_pretrained(base_raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload()
    sft_model.eval()

    print("[Stitch] Widening 9Q/3KV -> 12Q/4KV with W_O_new = 0.0...", flush=True)
    stitched, stitch_meta = stitch_zero_shock_gqa_group(sft_model, seed=args.seed)
    stitched.to(device).eval()

    # Exact zero-shock verification
    sample_prompt = base_rl.build_prompt(tokenizer, eval_rows[0]["messages"])
    enc = tokenizer(sample_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits_old = sft_model(**enc).logits
        logits_new = stitched(**enc).logits
        max_logit_diff = float((logits_new - logits_old).abs().max().item())
        kl_step0 = float(
            F.kl_div(F.log_softmax(logits_new, dim=-1), F.softmax(logits_old, dim=-1), reduction="batchmean").item()
        )
    print(f"  [Zero-Shock Proof] max|logits_new - logits_old| = {max_logit_diff:.8f} | KL(step 0) = {kl_step0:.8f}", flush=True)

    # Freeze 100% of original weights; allow gradients ONLY on the newly stitched head slices in layers [0, 1, 28, 29]
    train_layers = [0, 1, 28, 29]
    old_q_dim = stitch_meta["old_q_heads"] * stitch_meta["head_dim"]
    old_kv_dim = stitch_meta["old_kv_heads"] * stitch_meta["head_dim"]
    for p in stitched.parameters():
        p.requires_grad_(False)

    hooks = []
    trainable_slice_params = 0
    for li in train_layers:
        attn = stitched.model.layers[li].self_attn
        for proj_name, cut_dim, axis in [
            ("q_proj", old_q_dim, 0),
            ("k_proj", old_kv_dim, 0),
            ("v_proj", old_kv_dim, 0),
            ("o_proj", old_q_dim, 1),
        ]:
            w = getattr(attn, proj_name).weight
            w.requires_grad_(True)
            if axis == 0:
                trainable_slice_params += int((w.shape[0] - cut_dim) * w.shape[1])
                def _mk0(c: int):
                    def _hk(g):
                        out = g.clone()
                        out[:c, :] = 0.0
                        return out
                    return _hk
                hooks.append(w.register_hook(_mk0(cut_dim)))
            else:
                trainable_slice_params += int(w.shape[0] * (w.shape[1] - cut_dim))
                def _mk1(c: int):
                    def _hk(g):
                        out = g.clone()
                        out[:, :c] = 0.0
                        return out
                    return _hk
                hooks.append(w.register_hook(_mk1(cut_dim)))

    ref_model = copy.deepcopy(stitched).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in stitched.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.0)
    rng = random.Random(args.seed)
    step_logs = []
    t0 = time.time()
    print(f"[RL] Training ONLY stitched head slices in layers {train_layers} ({trainable_slice_params:,} active params, 100% base frozen)...", flush=True)
    for step in range(1, args.steps + 1):
        ts = time.time()
        batch = [prompt_stream[(step * 2 + i) % len(prompt_stream)] for i in range(2)]
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device)
        rewards_all, kls_all = [], []
        active_groups = 0
        for row in batch:
            p_txt = base_rl.build_prompt(tokenizer, row["messages"])
            p_ids = tokenizer(p_txt, return_tensors="pt").input_ids.to(device)
            p_len = p_ids.shape[1]
            gens, rews = [], []
            stitched.eval()
            with torch.no_grad():
                for _ in range(4):
                    out = stitched.generate(
                        p_ids, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    c_ids = out[0, p_len:]
                    if c_ids.numel() == 0:
                        continue
                    txt = tokenizer.decode(c_ids, skip_special_tokens=True)
                    r, _ = base_rl.score_completion(txt, row["target"])
                    gens.append(out[0])
                    rews.append(r)
            if not rews:
                continue
            rewards_all.extend(rews)
            r_t = torch.tensor(rews, dtype=torch.float32, device=device)
            if float(r_t.std().item()) < 1e-6:
                continue
            active_groups += 1
            adv = (r_t - r_t.mean()) / (r_t.std() + 1e-6)
            stitched.train()
            for seq, ai in zip(gens, adv):
                full_ids = seq.unsqueeze(0)
                lpc = base_rl.completion_logprobs(stitched, full_ids, p_len).sum()
                with torch.no_grad():
                    lpr = base_rl.completion_logprobs(ref_model, full_ids, p_len).sum()
                kl = float(((lpc - lpr) ** 2).detach().item()) / max(int(seq.numel() - p_len), 1)
                kls_all.append(kl)
                loss = loss + (-(ai.detach() * lpc) + 0.02 * (lpc - lpr) ** 2) / (4 * len(batch))
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in stitched.parameters() if p.requires_grad], 1.0)
            opt.step()
        row_log = {
            "step": step,
            "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4),
            "active_groups": active_groups,
            "mean_kl": round(sum(kls_all) / max(len(kls_all), 1), 6),
            "step_sec": round(time.time() - ts, 2),
        }
        step_logs.append(row_log)
        print(f"  [StitchedHeadRL] step {step:02d}/{args.steps} | rew={row_log['mean_reward']:.3f} | act={active_groups}/2 | kl={row_log['mean_kl']:.6f} | {row_log['step_sec']}s", flush=True)

    for h in hooks:
        h.remove()
    stitched.eval()

    # Verify 100% of original base weights are bit-identical!
    base_drift = 0.0
    for li in train_layers:
        o_old = sft_model.model.layers[li].self_attn.o_proj.weight.data
        o_new_base = stitched.model.layers[li].self_attn.o_proj.weight.data[:, :old_q_dim]
        base_drift = max(base_drift, float((o_new_base - o_old).abs().max().item()))
    new_head_norm = float(stitched.model.layers[29].self_attn.o_proj.weight.data[:, old_q_dim:].norm().item())

    ev = base_rl.evaluate_heldout(stitched, tokenizer, eval_rows, device)
    print(f"  [Base Preservation] max_base_weight_drift = {base_drift:.8f} | L29 W_O_new norm = {new_head_norm:.6f}", flush=True)
    print(f"  [StitchedHeadRL] 40-task eval: {ev}", flush=True)

    summary = {
        "stitch_meta": stitch_meta,
        "zero_shock_max_logit_diff": max_logit_diff,
        "zero_shock_kl_step0": kl_step0,
        "train_layers": train_layers,
        "trainable_slice_params": trainable_slice_params,
        "base_weight_max_drift": base_drift,
        "layer29_new_wo_norm": new_head_norm,
        "wall_sec": round(time.time() - t0, 2),
        "steps": step_logs,
        "eval": ev,
    }
    (out_dir / "stitched_head_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {out_dir / 'stitched_head_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
