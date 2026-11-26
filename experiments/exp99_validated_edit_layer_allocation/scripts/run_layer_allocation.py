#!/usr/bin/env python3
"""Experiment 99: Strict late-layer allocation ablation for the validated edit.

Total edited-column budget is fixed at 64 so layer choice is not confounded by
edit count. Arms compare L29-only, L28-only, L27-only, and the validated L29/L28
split at 32 columns per layer, all at strength 0.90.
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

PROBE_LAYERS = [27, 28, 29]
STRENGTH = 0.90
TOTAL_COLUMNS = 64
ALLOCATIONS = {
    "l29_only_64": {29: 64},
    "l28_only_64": {28: 64},
    "l27_only_64": {27: 64},
    "l29_l28_split_32_each": {29: 32, 28: 32},
}


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=12)
    parser.add_argument("--bench-n", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp99_validated_edit_layer_allocation/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 99] Strict validated-edit layer allocation on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    probe_source = ordered_fork_rows(train_rows, 12)
    probe_rows = probe_source[:4] + probe_source[-4:]
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline benchmark] {baseline_bench}", flush=True)
    try:
        baseline_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, PROBE_LAYERS)
    except Exception as exc:
        baseline_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [baseline TransformerLens probe failed; continuing] {baseline_probe['error']}", flush=True)

    subspaces, singular_values = build_validated_rank1_subspaces(
        sft, tokenizer, train_rows, device, edit_layers=PROBE_LAYERS, n_pairs=args.n_pairs
    )
    results = {
        "experiment": "exp99_validated_edit_layer_allocation",
        "controlled_variable": "late-layer allocation only, with total edited-column budget fixed at 64",
        "fixed_conditions": {
            "operator": "validated raw-rank1 residual-direction projection",
            "strength": STRENGTH,
            "total_edited_columns": TOTAL_COLUMNS,
            "direction_measurement": "layer-specific fixed subspaces measured once on the unedited base model",
            "n_pairs": args.n_pairs,
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_transformerlens_probe": baseline_probe,
        "top_singular_values": singular_values,
        "arms": {},
    }
    for arm, allocation in ALLOCATIONS.items():
        print(f"[Run] {arm}: fixed strength-0.90 total-64 validated edit", flush=True)
        edited = copy.deepcopy(sft).to(device).eval()
        edit_parts = []
        for layer, count in allocation.items():
            edit_parts.append(apply_rank_k_subspace_edit(edited, [layer], subspaces, count, STRENGTH))
        behavior = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
        bench = run_benchmark(edited, tokenizer, device, args.bench_n)
        try:
            probe = probe_layers_geometry(edited, tokenizer, probe_rows, device, PROBE_LAYERS)
        except Exception as exc:
            probe = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{arm} TransformerLens probe failed; continuing] {probe['error']}", flush=True)
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench}", flush=True)
        results["arms"][arm] = {
            "allocation": allocation,
            "strength": STRENGTH,
            "total_columns": sum(allocation.values()),
            "edit_info_parts": edit_parts,
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
    path = out_dir / "validated_edit_layer_allocation_summary.json"
    write_json(path, results)
    print(f"[Exp 99 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
