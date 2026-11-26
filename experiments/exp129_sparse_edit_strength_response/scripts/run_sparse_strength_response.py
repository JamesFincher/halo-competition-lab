#!/usr/bin/env python3
"""Experiment 129: Strength response at the sparse validated-edit operating point.

Exp95 mapped strength at 64 columns per layer, but exp85/exp98 indicate the
useful operating point is much sparser. This experiment holds the edit fixed at
16 columns per layer (32 total) and varies only projection strength:
0.50/0.70/0.90/1.20.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/whitened_subspace_rfm_edit/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import dumps_safe, write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from run_whitened_subspace_edit import apply_rank_k_subspace_edit  # noqa: E402
from validated_edit import build_validated_rank1_subspaces  # noqa: E402
from methodical_slider import ordered_fork_rows, probe_layers_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402

EDIT_LAYERS = [29, 28]
TOP_COLUMNS = 16
STRENGTHS = [0.50, 0.70, 0.90, 1.20]


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=12)
    parser.add_argument("--bench-n", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp129_sparse_edit_strength_response/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 129] Sparse validated-edit strength response on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    probe_rows = ordered_fork_rows(train_rows, 12)[:4] + ordered_fork_rows(train_rows, 12)[-4:]
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline benchmark] {baseline_bench}", flush=True)
    try:
        baseline_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, EDIT_LAYERS)
    except Exception as exc:
        baseline_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [baseline TransformerLens probe failed; continuing] {baseline_probe['error']}", flush=True)

    subspaces, singular_values = build_validated_rank1_subspaces(
        sft, tokenizer, train_rows, device, edit_layers=EDIT_LAYERS, n_pairs=args.n_pairs
    )
    results = {
        "experiment": "exp129_sparse_edit_strength_response",
        "controlled_variable": "projection strength only (0.50/0.70/0.90/1.20) at 16 columns per layer",
        "fixed_conditions": {
            "operator": "validated raw-rank1 residual-direction projection",
            "edit_layers": EDIT_LAYERS,
            "top_columns_per_layer": TOP_COLUMNS,
            "total_columns_two_layers": TOP_COLUMNS * len(EDIT_LAYERS),
            "direction_measurement": "same fixed subspaces measured once on the unedited base model",
            "n_pairs": args.n_pairs,
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_transformerlens_probe": baseline_probe,
        "top_singular_values": singular_values,
        "arms": {},
    }
    for strength in STRENGTHS:
        arm = f"strength_{str(strength).replace('.', 'p')}"
        print(f"[Run] {arm}: fixed L29/L28 16-columns-per-layer validated edit; strength only", flush=True)
        edited = copy.deepcopy(sft).to(device).eval()
        edit_info = apply_rank_k_subspace_edit(edited, EDIT_LAYERS, subspaces, TOP_COLUMNS, strength)
        behavior = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
        bench = run_benchmark(edited, tokenizer, device, args.bench_n)
        try:
            probe = probe_layers_geometry(edited, tokenizer, probe_rows, device, EDIT_LAYERS)
        except Exception as exc:
            probe = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{arm} TransformerLens probe failed; continuing] {probe['error']}", flush=True)
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench}", flush=True)
        results["arms"][arm] = {
            "top_columns_per_layer": TOP_COLUMNS,
            "total_columns_two_layers": TOP_COLUMNS * len(EDIT_LAYERS),
            "strength": strength,
            "edit_info": edit_info,
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": metric_delta(bench, baseline_bench),
            "transformerlens_probe": probe,
        }
        del edited
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "sparse_edit_strength_response_summary.json"
    write_json(path, results)
    print(f"[Exp 129 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
