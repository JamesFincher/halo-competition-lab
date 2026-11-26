#!/usr/bin/env python3
"""Experiment 97: Independent-alpha initialization-scale ablation for the v2 bank.

The original fork used exact-zero alphas. Because exact zero cannot be made
"smaller," this experiment interprets the iteration-2 hypothesis as alpha init
scale: sample per-head alphas uniformly in ±scale before training. Scale is the
only variable; architecture, seed, signs/pattern, training budget, and evaluation
are fixed. Scale 0 is the exact-zero reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from methodical_slider import IndependentLayerSliderBank, head_parameter_count, install_independent_layer_bank, ordered_fork_rows, probe_layers_geometry, train_independent_layer_bank  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402

LAYERS = [24, 25, 26, 27]
WIDTH = 64
INIT_SCALES = [0.0, 0.001, 0.003]


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp97_independent_alpha_init_scale/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 97] Independent-alpha init-scale ablation on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = ordered_fork_rows(train_rows, 12)
    probe_rows = rows[:4] + rows[-4:]
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline benchmark] {baseline_bench}", flush=True)
    try:
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, LAYERS)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    fixed_signs = {li: (1.0 if idx % 2 == 0 else -1.0) for idx, li in enumerate(LAYERS)}
    results = {
        "experiment": "exp97_independent_alpha_init_scale",
        "controlled_variable": "initial independent-alpha scale only (0/0.001/0.003; fixed alternating signs)",
        "interpretation_note": "original v2 alphas were exact-zero; this tests whether small nonzero per-head alpha initialization is protective or harmful",
        "fixed_conditions": {
            "layers": LAYERS,
            "head_dim": WIDTH,
            "seed": args.seed,
            "fixed_alpha_signs": fixed_signs,
            "training": "slider_head_fork-style joint grouped REINFORCE, no KL/BC",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "transformerlens_probe_by_layer": tl_probe,
        "arms": {},
    }
    for scale in INIT_SCALES:
        arm = f"init_{str(scale).replace('.', 'p')}"
        print(f"[Run] {arm}: fixed v2 architecture, alpha init scale only", flush=True)
        torch.manual_seed(args.seed)
        bank = IndependentLayerSliderBank(sft.config.hidden_size, WIDTH, LAYERS).to(device)
        init_alphas = {}
        with torch.no_grad():
            for li, head in bank.layer_heads.items():
                val = fixed_signs[int(li)] * scale
                head.alpha.fill_(val)
                init_alphas[li] = val
        pre_handles = install_independent_layer_bank(sft, bank)
        pre_behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        for pre_handle in pre_handles:
            pre_handle.remove()
        handles, trace = train_independent_layer_bank(
            sft, tokenizer, rows, device, bank,
            steps=args.steps, group_size=args.group_size, lr=args.lr,
        )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        for handle in handles:
            handle.remove()
        final_alphas = {str(k): round(v, 6) for k, v in bank.alphas().items()}
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alphas={final_alphas} bench={bench}", flush=True)
        results["arms"][arm] = {
            "alpha_init_scale": scale,
            "initial_alphas": init_alphas,
            "pretraining_behavior_eval": pre_behavior,
            "parameter_count": head_parameter_count(bank),
            "final_alphas": final_alphas,
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": metric_delta(bench, baseline_bench),
            "trace": trace,
        }
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "independent_alpha_init_scale_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 97 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
