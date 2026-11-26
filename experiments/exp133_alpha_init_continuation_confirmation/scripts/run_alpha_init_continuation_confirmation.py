#!/usr/bin/env python3
"""Experiment 133: Alpha-init continuation-benchmark confirmation.

Exp97 reported the first positive legacy-benchmark movement: independent-alpha
init scale 0.001 reached ARC-Easy 44 versus baseline 38 at n=50. That is only a
three-item difference, so this experiment reruns the exact same controlled
init-scale ablation and adds higher-resolution continuation-only scoring at
n=150. The controlled variable remains initial alpha scale only; the scientific
question is whether the apparent capability gain survives a sharper, larger
measurement or was small-sample measurement noise.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from methodical_slider import IndependentLayerSliderBank, head_parameter_count, install_independent_layer_bank, ordered_fork_rows, probe_layers_geometry, train_independent_layer_bank  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYERS = [24, 25, 26, 27]
WIDTH = 64
INIT_SCALES = [0.0, 0.001, 0.003]


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def continuation_delta(after: dict, before: dict) -> dict:
    out = {}
    for bench, metrics in after.items():
        if bench not in before or not isinstance(metrics, dict) or not isinstance(before[bench], dict):
            continue
        out[bench] = {
            k: round(metrics[k] - before[bench][k], 2)
            for k in ("sum_acc", "token_norm_acc", "byte_norm_acc")
            if isinstance(metrics.get(k), (int, float)) and isinstance(before[bench].get(k), (int, float))
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--continuation-n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp133_alpha_init_continuation_confirmation/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 133] Alpha-init continuation confirmation on {device}", flush=True)

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
    print(f"  [baseline legacy benchmark] {baseline_bench}", flush=True)
    baseline_continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
    print(f"  [baseline continuation benchmark] {baseline_continuation}", flush=True)
    try:
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, LAYERS)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    fixed_signs = {li: (1.0 if idx % 2 == 0 else -1.0) for idx, li in enumerate(LAYERS)}
    results = {
        "experiment": "exp133_alpha_init_continuation_confirmation",
        "controlled_variable": "initial independent-alpha scale only (0/0.001/0.003; fixed alternating signs)",
        "confirmation_question": "Does exp97's ARC-Easy +6 legacy movement at init 0.001 survive continuation-only scoring at n=150?",
        "primary_endpoint": "arc_easy continuation sum/token/byte accuracy delta versus baseline",
        "fixed_conditions": {
            "layers": LAYERS,
            "head_dim": WIDTH,
            "seed": args.seed,
            "fixed_alpha_signs": fixed_signs,
            "training": "slider_head_fork-style joint grouped REINFORCE, no KL/BC",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
            "exp97_replication_note": "same architecture/init signs/steps/optimizer as exp97; measurement resolution is the only intentional addition",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_continuation_benchmark": baseline_continuation,
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
        continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
        for handle in handles:
            handle.remove()
        final_alphas = {str(k): round(v, 6) for k, v in bank.alphas().items()}
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alphas={final_alphas} bench={bench} continuation={continuation}", flush=True)
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
            "continuation_benchmark": continuation,
            "continuation_delta_vs_baseline": continuation_delta(continuation, baseline_continuation),
            "trace": trace,
        }
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "alpha_init_continuation_confirmation_summary.json"
    write_json(path, results)
    print(f"[Exp 133 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
