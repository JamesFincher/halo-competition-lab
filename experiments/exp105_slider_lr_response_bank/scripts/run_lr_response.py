#!/usr/bin/env python3
"""Experiment 105: learning-rate dose-response for the collapsing v2 multi-layer bank.

Fork v2/v3/v6 collapsed with negative alpha drift. This experiment fixes the v2
architecture and varies only AdamW learning rate to test optimization instability.
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
from methodical_slider import IndependentLayerSliderBank, head_parameter_count, ordered_fork_rows, probe_layers_geometry, train_independent_layer_bank  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402

LAYERS = [24, 25, 26, 27]
WIDTH = 64
LRS = [3e-4, 1e-3, 3e-3]


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp105_slider_lr_response_bank/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 105] Slider learning-rate dose for v2 bank on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = ordered_fork_rows(train_rows, 12)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline benchmark] {baseline_bench}", flush=True)
    probe_rows = rows[:4] + rows[-4:]
    try:
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, LAYERS)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp105_slider_lr_response_bank",
        "controlled_variable": "AdamW learning rate only (3e-4/1e-3/3e-3)",
        "fixed_conditions": {
            "layers": LAYERS,
            "head_dim": WIDTH,
            "independent_alpha_per_layer": True,
            "seed": args.seed,
            "training": "slider_head_fork-style joint grouped REINFORCE, no KL/BC",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "transformerlens_probe_by_layer": tl_probe,
        "arms": {},
    }
    for lr in LRS:
        arm = f"lr_{str(lr).replace('.', 'p')}"
        print(f"[Run] {arm}: fixed v2 L24-27 width-64 independent-alpha bank", flush=True)
        torch.manual_seed(args.seed)
        bank = IndependentLayerSliderBank(sft.config.hidden_size, WIDTH, LAYERS).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity_diff = max(float((head(probe) - probe).abs().max().item()) for head in bank.layer_heads.values())
        if identity_diff != 0.0:
            raise RuntimeError(f"zero-init identity failed: {identity_diff}")
        handles, trace = train_independent_layer_bank(
            sft, tokenizer, rows, device, bank,
            steps=args.steps, group_size=args.group_size, lr=lr,
        )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        for handle in handles:
            handle.remove()
        final_alphas = {str(k): round(v, 6) for k, v in bank.alphas().items()}
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alphas={final_alphas} bench={bench}", flush=True)
        results["arms"][arm] = {
            "learning_rate": lr,
            "parameter_count": head_parameter_count(bank),
            "zero_init_identity_max_diff": identity_diff,
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
    path = out_dir / "slider_lr_response_bank_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 105 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
