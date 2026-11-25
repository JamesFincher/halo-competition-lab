#!/usr/bin/env python3
"""Experiment 12: Head & MLP False-Node Seek, Sniff & Double-Check RL (SmolLM2-135M).

Identifies:
  1. Individual False-Trigger Attention Heads (across all 270 Q-heads via o_proj input hooks; 0 KV touched)
  2. Individual MLP False Nodes (across all 46,080 hidden neurons via down_proj input hooks)
Applies a Sequential Double-Check Gate (verify on dev slice before moving on), then benchmarks 4 arms:
  - Arm A: Attention-Head Double-Check Edit Only (0 RL steps, o_proj head slices only)
  - Arm B: MLP False-Node Double-Check Edit Only (0 RL steps, down_proj false-node columns only)
  - Arm C: Combined Head Double-Check -> MLP False-Node Double-Check (0 RL steps)
  - Arm D: Combined Head + False-Node Double-Check -> Disjoint Re-Anchored RL (6 steps)
"""

from __future__ import annotations

import argparse
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
import run_micro_targeted_rl as base_rl  # noqa: E402


@torch.no_grad()
def sniff_heads_and_false_nodes(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    n_pairs: int = 8,
    safety_penalty: float = 0.45,
) -> dict:
    """Profile all attention heads (o_proj inputs) and all MLP neurons (down_proj inputs)
    across token-aligned benign-pass vs false-refusal vs true-unsafe contrasts."""
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    inter_dim = int(model.config.intermediate_size)

    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_pairs]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][:n_pairs]
    canonical_refusal = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'

    pass_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in benign_rows]
    fail_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + canonical_refusal for r in benign_rows]
    unsafe_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in unsafe_rows]

    def _collect(texts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        head_acc = [torch.zeros(n_heads, head_dim) for _ in range(n_layers)]
        mlp_acc = [torch.zeros(inter_dim) for _ in range(n_layers)]
        resid_acc = [torch.zeros(model.config.hidden_size) for _ in range(n_layers)]

        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
            cap_h, cap_m, cap_r = {}, {}, {}
            hooks = []
            for idx, lyr in enumerate(layers):
                def _mk_o(i: int):
                    def _hk(_m, inp, _out):
                        x = inp[0][0, -24:, :].detach().float().mean(dim=0).cpu()
                        cap_h[i] = x.view(n_heads, head_dim)
                    return _hk
                def _mk_d(i: int):
                    def _hk(_m, inp, _out):
                        cap_m[i] = inp[0][0, -24:, :].detach().float().mean(dim=0).cpu()
                    return _hk
                def _mk_l(i: int):
                    def _hk(_m, _inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        cap_r[i] = h[0, -24:, :].detach().float().mean(dim=0).cpu()
                    return _hk
                hooks.append(lyr.self_attn.o_proj.register_forward_hook(_mk_o(idx)))
                hooks.append(lyr.mlp.down_proj.register_forward_hook(_mk_d(idx)))
                hooks.append(lyr.register_forward_hook(_mk_l(idx)))
            try:
                _ = model(**enc)
            finally:
                for h in hooks:
                    h.remove()
            for idx in range(n_layers):
                head_acc[idx] += cap_h[idx]
                mlp_acc[idx] += cap_m[idx]
                resid_acc[idx] += cap_r[idx]
        inv = 1.0 / max(len(texts), 1)
        return [x * inv for x in head_acc], [x * inv for x in mlp_acc], [x * inv for x in resid_acc]

    h_pass, m_pass, r_pass = _collect(pass_texts)
    h_fail, m_fail, r_fail = _collect(fail_texts)
    h_uns, m_uns, r_uns = _collect(unsafe_texts)

    # 1. Rank individual Attention Heads (270 total)
    head_records = []
    head_false_dirs: dict[tuple[int, int], torch.Tensor] = {}
    for li in range(n_layers):
        for hi in range(n_heads):
            df = h_fail[li][hi] - h_pass[li][hi]
            ds = h_uns[li][hi] - h_pass[li][hi]
            fn = float(df.norm().item())
            sn = float(ds.norm().item())
            false_score = fn - safety_penalty * sn
            head_false_dirs[(li, hi)] = df
            head_records.append(
                {
                    "layer": li,
                    "head": hi,
                    "false_l2": round(fn, 4),
                    "safety_l2": round(sn, 4),
                    "false_head_score": round(false_score, 4),
                }
            )
    head_records.sort(key=lambda x: x["false_head_score"], reverse=True)

    # 2. Rank individual MLP False Nodes (46,080 total)
    false_nodes_by_layer: dict[int, list[int]] = {}
    top_node_records = []
    resid_false_dirs = {li: (r_fail[li] - r_pass[li]) for li in range(n_layers)}
    for li in range(n_layers):
        df = (m_fail[li] - m_pass[li]).abs()
        ds = (m_uns[li] - m_pass[li]).abs()
        # False nodes: strong false-refusal spike minus real-safety spike
        score = df - safety_penalty * ds
        top_vals, top_idx = torch.topk(score, k=64)
        false_nodes_by_layer[li] = [int(i) for i in top_idx.tolist()]
        for v, idx in zip(top_vals[:3].tolist(), top_idx[:3].tolist()):
            top_node_records.append(
                {
                    "layer": li,
                    "neuron": int(idx),
                    "false_delta": round(float(df[idx].item()), 4),
                    "safety_delta": round(float(ds[idx].item()), 4),
                    "false_node_score": round(float(v), 4),
                }
            )
    top_node_records.sort(key=lambda x: x["false_node_score"], reverse=True)
    layer_rank = sorted(
        [{"layer": li, "resid_l2": round(float(resid_false_dirs[li].norm().item()), 4)} for li in range(n_layers)],
        key=lambda x: x["resid_l2"],
        reverse=True,
    )
    return {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "top_false_heads": head_records[:16],
        "head_false_dirs": head_false_dirs,
        "top_false_nodes": top_node_records[:20],
        "false_nodes_by_layer": false_nodes_by_layer,
        "resid_false_dirs": resid_false_dirs,
        "layer_rank": layer_rank,
    }


def apply_head_qo_edit(
    model: torch.nn.Module,
    target_heads: list[tuple[int, int]],
    resid_false_dirs: dict[int, torch.Tensor],
    head_dim: int,
    strength: float,
) -> dict:
    """Surgical edit on ONLY selected attention head output slices o_proj[:, s0:s1] (0 KV touched!)."""
    edited = 0
    total = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for li, hi in target_heads:
            o_proj = model.model.layers[li].self_attn.o_proj
            w = o_proj.weight.data.float()
            s0, s1 = hi * head_dim, (hi + 1) * head_dim
            u = resid_false_dirs[li].to(w.device, dtype=torch.float32)
            u = u / (u.norm() + 1e-8)
            proj = torch.matmul(u, w[:, s0:s1])
            w[:, s0:s1] -= strength * torch.outer(u, proj)
            o_proj.weight.data.copy_(w.to(o_proj.weight.dtype))
            edited += int(w.shape[0] * head_dim)
    return {"edited_heads": target_heads, "edited_weights": edited, "edited_pct": round(100.0 * edited / max(total, 1), 5), "kv_heads_touched": 0}


def apply_false_node_mlp_edit(
    model: torch.nn.Module,
    edit_layers: list[int],
    false_nodes_by_layer: dict[int, list[int]],
    resid_false_dirs: dict[int, torch.Tensor],
    top_nodes: int,
    strength: float,
) -> dict:
    """Surgical edit on ONLY sniffed MLP false-node columns in down_proj."""
    edited = 0
    total = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for li in edit_layers:
            down = model.model.layers[li].mlp.down_proj
            w = down.weight.data.float()
            idx = torch.tensor(false_nodes_by_layer[li][:top_nodes], dtype=torch.long, device=w.device)
            u = resid_false_dirs[li].to(w.device, dtype=torch.float32)
            u = u / (u.norm() + 1e-8)
            proj = torch.matmul(u, w[:, idx])
            w[:, idx] -= strength * torch.outer(u, proj)
            down.weight.data.copy_(w.to(down.weight.dtype))
            edited += int(w.shape[0] * idx.numel())
    return {"edit_layers": edit_layers, "top_nodes_per_layer": top_nodes, "edited_weights": edited, "edited_pct": round(100.0 * edited / max(total, 1), 5)}


def double_check_gate(model, tokenizer, dev_rows, device, cand_builder, label: str, strengths=(0.4, 0.7, 1.0)):
    """Immediate Double-Check before moving on: keep adjustment ONLY if dev reward improves without safety collapse."""
    base_ev = base_rl.evaluate_heldout(model, tokenizer, dev_rows, device)
    best_m, best_ev = model, base_ev
    best_meta = {"label": label, "accepted": False, "dev_before": base_ev["mean_reward"]}
    for st in strengths:
        cand = copy.deepcopy(model).to(device)
        info = cand_builder(cand, st)
        ev = base_rl.evaluate_heldout(cand, tokenizer, dev_rows, device)
        safety_ok = ev["unsafe_refusal_rate"] >= base_ev["unsafe_refusal_rate"] - 21.0
        print(
            f"    [DoubleCheck:{label}] st={st} | dev_rew={base_ev['mean_reward']:.3f}->{ev['mean_reward']:.3f} | "
            f"unsafe={ev['unsafe_refusal_rate']}% | benign_fr={ev['benign_false_refusal_rate']}% | ok={safety_ok}",
            flush=True,
        )
        if safety_ok and ev["mean_reward"] > best_ev["mean_reward"] + 1e-4:
            best_m, best_ev = cand, ev
            best_meta = {"label": label, "accepted": True, "strength": st, "dev_before": base_ev["mean_reward"], "dev_after": ev["mean_reward"], **info}
    return best_m, best_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    out_dir = ROOT / "experiments/head_and_node_sniff_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = ben[-5:] + uns[-5:]
    prompt_stream = [x for pair in zip(ben[:-5], uns[:-5]) for x in pair]

    print("[Setup] Loading SmolLM2-135M + Halo SFT checkpoint-36...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft_model = PeftModel.from_pretrained(base_raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload()
    sft_model.eval()

    print("[Sniff] Profiling all 270 attention heads & 46,080 MLP neurons for false nodes...", flush=True)
    prof = sniff_heads_and_false_nodes(sft_model, tokenizer, train_rows, device, n_pairs=8, safety_penalty=0.45)
    top_heads = [(h["layer"], h["head"]) for h in prof["top_false_heads"][:8]]
    top2_layers = [r["layer"] for r in prof["layer_rank"][:2]]
    comp6_layers = sorted([r["layer"] for r in prof["layer_rank"][2:8]])
    print(f"  Top-8 False Attention Heads: {top_heads}", flush=True)
    print(f"  Top-5 False MLP Neurons: {prof['top_false_nodes'][:5]}", flush=True)

    # Arm A: Attention-Head Double-Check Only (0 KV touched, 0 RL steps)
    print("\n[Arm A] Attention-Head Double-Check Edit Only (o_proj slices only, 0 KV touched)...", flush=True)
    arm_a_m, arm_a_meta = double_check_gate(
        sft_model, tokenizer, dev_rows, device,
        lambda m, st: apply_head_qo_edit(m, top_heads, prof["resid_false_dirs"], prof["head_dim"], st),
        "HeadQO", strengths=(0.5, 0.85, 1.0),
    )
    ev_a = base_rl.evaluate_heldout(arm_a_m, tokenizer, eval_rows, device)
    print(f"  [Arm A] 40-task eval: {ev_a}", flush=True)

    # Arm B: MLP False-Node Double-Check Only (0 RL steps)
    print("\n[Arm B] MLP False-Node Double-Check Edit Only (top-64 false neurons in top-2 layers)...", flush=True)
    arm_b_m, arm_b_meta = double_check_gate(
        sft_model, tokenizer, dev_rows, device,
        lambda m, st: apply_false_node_mlp_edit(m, top2_layers, prof["false_nodes_by_layer"], prof["resid_false_dirs"], 64, st),
        "MLPFalseNode", strengths=(0.45, 0.70, 0.90),
    )
    ev_b = base_rl.evaluate_heldout(arm_b_m, tokenizer, eval_rows, device)
    print(f"  [Arm B] 40-task eval: {ev_b}", flush=True)

    # Arm C: Sequential Head Double-Check -> MLP False-Node Double-Check (0 RL steps)
    print("\n[Arm C] Combined Head Double-Check -> MLP False-Node Double-Check (0 RL steps)...", flush=True)
    arm_c_m, arm_c_node_meta = double_check_gate(
        arm_a_m, tokenizer, dev_rows, device,
        lambda m, st: apply_false_node_mlp_edit(m, top2_layers, prof["false_nodes_by_layer"], prof["resid_false_dirs"], 64, st),
        "HeadThenMLPNode", strengths=(0.45, 0.70, 0.90),
    )
    ev_c = base_rl.evaluate_heldout(arm_c_m, tokenizer, eval_rows, device)
    print(f"  [Arm C] 40-task eval: {ev_c}", flush=True)

    # Arm D: Combined Head + False-Node Double-Check -> Disjoint Re-Anchored RL (6 steps)
    print("\n[Arm D] Combined Head + False-Node Double-Check -> Disjoint Re-Anchored RL (6 steps)...", flush=True)
    ref_c = copy.deepcopy(arm_c_m).to(device)
    ref_c.eval()
    comp_regex = f".*\\.layers\\.({'|'.join(map(str, comp6_layers))})\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as fh:
        arm_d_m, arm_d_stats = base_rl.run_grpo_arm(
            "armD_head_node_double_check_rl", arm_c_m, ref_c, tokenizer, prompt_stream, device,
            comp_regex, 16, steps=args.steps, group_size=4, lr=7.5e-5, beta=0.02, step_log_fh=fh, seed=args.seed,
        )
    ev_d = base_rl.evaluate_heldout(arm_d_m, tokenizer, eval_rows, device)
    print(f"  [Arm D] 40-task eval: {ev_d}", flush=True)

    summary = {
        "top_false_heads": prof["top_false_heads"][:10],
        "top_false_mlp_nodes": prof["top_false_nodes"][:10],
        "comp6_rl_layers": comp6_layers,
        "arms": {
            "armA_head_double_check_only": {"meta": arm_a_meta, "eval": ev_a},
            "armB_mlp_false_node_only": {"meta": arm_b_meta, "eval": ev_b},
            "armC_head_plus_false_node_only": {"head_meta": arm_a_meta, "node_meta": arm_c_node_meta, "eval": ev_c},
            "armD_head_node_double_check_rl": {"rl_stats": arm_d_stats, "eval": ev_d},
        },
    }
    (out_dir / "head_node_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved {out_dir / 'head_node_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
