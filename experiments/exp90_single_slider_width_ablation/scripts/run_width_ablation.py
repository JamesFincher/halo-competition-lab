#!/usr/bin/env python3
"""Experiment 90: Single-variable slider-head width ablation at layer 27.

Fork iteration 1 showed single dim64 survived while single dim128 collapsed. This
experiment isolates head width only: same layer, prompt stream, seed, optimizer,
training budget, initialization family, and evaluation for 16/32/64/128.
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
from methodical_slider import PlainSliderHead, head_parameter_count, ordered_fork_rows, probe_layer27_geometry, train_plain_slider  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402

LAYER = 27
WIDTHS = [16, 32, 64, 128]


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
    out_dir = ROOT / "experiments/exp90_single_slider_width_ablation/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 90] Single-slider width ablation on {device}", flush=True)

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
        tl_probe = probe_layer27_geometry(sft, tokenizer, probe_rows, device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp90_single_slider_width_ablation",
        "controlled_variable": "slider head width only (16/32/64/128)",
        "fixed_conditions": {
            "layer": LAYER,
            "head_count": 1,
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas",
            "training": "slider_head_fork-style grouped REINFORCE, no KL/BC",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "transformerlens_probe": tl_probe,
        "arms": {},
    }
    for width in WIDTHS:
        arm = f"width_{width}"
        print(f"[Run] {arm}: one L27 head, all other variables fixed", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, width).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity_diff = float((head(probe) - probe).abs().max().item())
        if identity_diff != 0.0:
            raise RuntimeError(f"zero-init identity failed for width {width}: {identity_diff}")
        handle, trace = train_plain_slider(
            sft, tokenizer, rows, device, head, LAYER,
            steps=args.steps, group_size=args.group_size, lr=args.lr,
        )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        handle.remove()
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.5f} bench={bench}", flush=True)
        results["arms"][arm] = {
            "head_dim": width,
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity_diff,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": metric_delta(bench, baseline_bench),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "single_slider_width_ablation_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 90 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
