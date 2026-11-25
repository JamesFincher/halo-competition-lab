#!/usr/bin/env python3
"""Experiment 15: 1-Pass RelP Causal Attribution Patching Sniff (SmolLM2-135M).

Computes true first-order causal effect of every MLP neuron (46,080) and every Attention Head (270)
on the decision-token logit margin M = logit('call') - logit('ref') in 1 backward pass, with
RelP-style variance-detached RMSNorm stabilization (arXiv:2508.21258).

Compares on 40 unseen held-out tasks:
  - Arm 1: Activation-Only False-Node Selection (Exp 12 baseline)
  - Arm 2: 1-Pass RelP Causal Neuron Selection (top-64 causal MLP neurons)
  - Arm 3: 1-Pass RelP Causal Head + Neuron Selection
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/head_and_node_sniff_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_head_and_node_sniff as hn_sniff  # noqa: E402


def compute_relp_causal_scores(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    n_pairs: int = 8,
) -> dict:
    """1-pass causal attribution: score(l, j) = (a_pass - a_fail) * d(logit_margin)/d(a_fail)."""
    t0 = time.time()
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    inter_dim = int(model.config.intermediate_size)

    # Prefix right up to the action value token: {"action": "
    prefix_suffix = '{"action": "'
    tok_call = tokenizer.encode("call", add_special_tokens=False)[0]
    tok_ref = tokenizer.encode("ref", add_special_tokens=False)[0]

    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_pairs]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][:n_pairs]

    # Collect clean (benign) vs corrupt (unsafe) activations at the decision token,
    # plus gradients of the logit margin on benign prompts (which falsely refuse!).
    mlp_causal = [torch.zeros(inter_dim) for _ in range(n_layers)]
    head_causal = [torch.zeros(n_heads) for _ in range(n_layers)]
    resid_diff = [torch.zeros(model.config.hidden_size) for _ in range(n_layers)]

    for p in model.parameters():
        p.requires_grad_(False)

    for ben_r, uns_r in zip(benign_rows, unsafe_rows):
        p_ben = base_rl.build_prompt(tokenizer, ben_r["messages"]) + prefix_suffix
        p_uns = base_rl.build_prompt(tokenizer, uns_r["messages"]) + prefix_suffix
        enc_b = tokenizer(p_ben, return_tensors="pt", truncation=True, max_length=256).to(device)
        enc_u = tokenizer(p_uns, return_tensors="pt", truncation=True, max_length=256).to(device)

        # Forward pass 1 (no grad): unsafe reference activations at decision position
        act_u_mlp, act_u_head, act_u_res = {}, {}, {}
        hooks_u = []
        for idx, lyr in enumerate(layers):
            def _mk_m(i: int):
                def _hk(_m, inp, _out):
                    act_u_mlp[i] = inp[0][0, -1, :].detach().float().cpu()
                return _hk
            def _mk_o(i: int):
                def _hk(_m, inp, _out):
                    act_u_head[i] = inp[0][0, -1, :].detach().float().view(n_heads, head_dim).cpu()
                return _hk
            def _mk_r(i: int):
                def _hk(_m, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    act_u_res[i] = h[0, -1, :].detach().float().cpu()
                return _hk
            hooks_u.append(lyr.mlp.down_proj.register_forward_hook(_mk_m(idx)))
            hooks_u.append(lyr.self_attn.o_proj.register_forward_hook(_mk_o(idx)))
            hooks_u.append(lyr.register_forward_hook(_mk_r(idx)))
        with torch.no_grad():
            _ = model(**enc_u)
        for h in hooks_u:
            h.remove()

        # Forward + 1 Backward pass on benign prompt (Retaining grads on down_proj & o_proj inputs)
        act_b_mlp, act_b_head, act_b_res = {}, {}, {}
        hooks_b = []
        for idx, lyr in enumerate(layers):
            def _mk_mb(i: int):
                def _hk(_m, inp):
                    x = inp[0].detach().requires_grad_(True)
                    act_b_mlp[i] = x
                    return (x,)
                return _hk
            def _mk_ob(i: int):
                def _hk(_m, inp):
                    x = inp[0].detach().requires_grad_(True)
                    act_b_head[i] = x
                    return (x,)
                return _hk
            def _mk_rb(i: int):
                def _hk(_m, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    act_b_res[i] = h[0, -1, :].detach().float().cpu()
                return _hk
            hooks_b.append(lyr.mlp.down_proj.register_forward_pre_hook(_mk_mb(idx)))
            hooks_b.append(lyr.self_attn.o_proj.register_forward_pre_hook(_mk_ob(idx)))
            hooks_b.append(lyr.register_forward_hook(_mk_rb(idx)))

        out_b = model(**enc_b)
        logits_last = out_b.logits[0, -1, :]
        # Margin: pushing refusal down & call_tool up
        margin = logits_last[tok_call] - logits_last[tok_ref]
        margin.backward()

        for h in hooks_b:
            h.remove()

        for idx in range(n_layers):
            # RelP-style normalized causal attribution: (a_ben - a_uns) * grad / (||grad|| + eps)
            ab_m = act_b_mlp[idx][0, -1, :].detach().float().cpu()
            gb_m = act_b_mlp[idx].grad[0, -1, :].detach().float().cpu()
            # Negative attribution = neuron pushing TOWARD refusal on a benign prompt!
            causal_m = ((ab_m - act_u_mlp[idx]) * gb_m).abs() / (gb_m.norm() + 1e-6)
            mlp_causal[idx] += causal_m

            ab_h = act_b_head[idx][0, -1, :].detach().float().view(n_heads, head_dim).cpu()
            gb_h = act_b_head[idx].grad[0, -1, :].detach().float().view(n_heads, head_dim).cpu()
            causal_h = ((ab_h - act_u_head[idx]) * gb_h).sum(dim=-1).abs() / (gb_h.norm() + 1e-6)
            head_causal[idx] += causal_h

            resid_diff[idx] += (act_u_res[idx] - act_b_res[idx])

    inv = 1.0 / max(len(benign_rows), 1)
    causal_nodes_by_layer = {}
    top_causal_nodes = []
    top_causal_heads = []
    for li in range(n_layers):
        s_m = mlp_causal[li] * inv
        vals, idx = torch.topk(s_m, k=64)
        causal_nodes_by_layer[li] = [int(i) for i in idx.tolist()]
        for v, ni in zip(vals[:2].tolist(), idx[:2].tolist()):
            top_causal_nodes.append({"layer": li, "neuron": int(ni), "causal_score": round(float(v), 5)})
        s_h = head_causal[li] * inv
        for hi in range(n_heads):
            top_causal_heads.append({"layer": li, "head": hi, "causal_score": round(float(s_h[hi].item()), 5)})

    top_causal_nodes.sort(key=lambda x: x["causal_score"], reverse=True)
    top_causal_heads.sort(key=lambda x: x["causal_score"], reverse=True)
    layer_causal_rank = sorted(
        [{"layer": li, "total_causal": round(float((mlp_causal[li] * inv).sum().item()), 5)} for li in range(n_layers)],
        key=lambda x: x["total_causal"],
        reverse=True,
    )
    return {
        "wall_sec": round(time.time() - t0, 2),
        "causal_nodes_by_layer": causal_nodes_by_layer,
        "top_causal_nodes": top_causal_nodes[:12],
        "top_causal_heads": top_causal_heads[:10],
        "layer_causal_rank": layer_causal_rank,
        "resid_dirs": {li: resid_diff[li] * inv for li in range(n_layers)},
    }


def main() -> None:
    device = torch.device("cpu")
    out_dir = ROOT / "experiments/relp_causal_sniff_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = ben[-5:] + uns[-5:]

    print("[Setup] Loading SmolLM2-135M + Halo SFT checkpoint-36...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft_model = PeftModel.from_pretrained(base_raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload()
    sft_model.eval()

    print("[RelP] Running 1-Pass Causal Attribution Patching...", flush=True)
    relp = compute_relp_causal_scores(sft_model, tokenizer, train_rows, device, n_pairs=8)
    print(f"  Completed in {relp['wall_sec']}s | Top-3 causal layers: {[r['layer'] for r in relp['layer_causal_rank'][:3]]}", flush=True)
    print(f"  Top-5 causal MLP neurons: {relp['top_causal_nodes'][:5]}", flush=True)
    print(f"  Top-5 causal Attn heads:  {relp['top_causal_heads'][:5]}", flush=True)

    # Also get canonical residual directions for projection
    prof = hn_sniff.sniff_heads_and_false_nodes(sft_model, tokenizer, train_rows, device, n_pairs=8, safety_penalty=0.45)
    top2_causal_layers = [r["layer"] for r in relp["layer_causal_rank"][:2]]

    print(f"\n[Arm 1] 1-Pass RelP Causal Neuron Edit (top-64 causal neurons in layers {top2_causal_layers})...", flush=True)
    m_relp, meta_relp = hn_sniff.double_check_gate(
        sft_model, tokenizer, dev_rows, device,
        lambda m, st: hn_sniff.apply_false_node_mlp_edit(
            m, top2_causal_layers, relp["causal_nodes_by_layer"], prof["resid_false_dirs"], 64, st
        ),
        "RelPCausalMLP", strengths=(0.45, 0.70, 0.90),
    )
    ev_relp = base_rl.evaluate_heldout(m_relp, tokenizer, eval_rows, device)
    print(f"  [RelP Causal Edit] 40-task eval: {ev_relp}", flush=True)

    summary = {
        "relp_wall_sec": relp["wall_sec"],
        "top_causal_layers": relp["layer_causal_rank"][:6],
        "top_causal_nodes": relp["top_causal_nodes"],
        "top_causal_heads": relp["top_causal_heads"],
        "gate_meta": meta_relp,
        "eval": ev_relp,
    }
    (out_dir / "relp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {out_dir / 'relp_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
