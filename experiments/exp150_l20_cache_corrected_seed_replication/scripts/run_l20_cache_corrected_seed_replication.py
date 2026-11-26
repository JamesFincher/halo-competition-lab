#!/usr/bin/env python3
"""Experiment 150: cache-corrected L20 width-64 seed replication.

Exp146 showed that the original L20 collapse was a KV-cache artifact and that the
corrected L20 arm reached 72.5/90/0 at seed 42. This experiment isolates seed
robustness only: same layer, width, prompt stream, optimizer, objective, budget,
and cache-corrected generation/evaluation across seeds 7/42/123.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
for x in [
    "experiments/edit_then_target_rl/scripts",
    "experiments/exp50_surgically_anchored_vector_rl/scripts",
    "experiments/exp146_cache_mode_inversion_diagnostic/scripts",
    "experiments/common",
    "scripts",
]:
    sys.path.insert(0, str(ROOT / x))
from serialization_utils import write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_cache_mode_inversion_diagnostic as inv  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, ordered_fork_rows, probe_layers_geometry, train_plain_slider  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

SEEDS = [7, 42, 123]
LAYER = 20
WIDTH = 64


def continuation_delta(after: dict, before: dict) -> dict:
    out = {}
    for bench, metrics in after.items():
        if bench in before and isinstance(metrics, dict) and isinstance(before[bench], dict):
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
    parser.add_argument("--continuation-n", type=int, default=100)
    parser.add_argument("--diagnostic-n", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed-count", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp150_l20_cache_corrected_seed_replication/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 150] cache-corrected L20 seed replication on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = ordered_fork_rows(train_rows, 12)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, max_new_tokens=args.max_new_tokens, use_cache=False)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    baseline_cont = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
    baseline_diag = inv.distribution_diagnostic(sft, tokenizer, device, n=args.diagnostic_n)
    try:
        tl_probe = probe_layers_geometry(sft, tokenizer, rows[:4] + rows[-4:], device, [LAYER])
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp150_l20_cache_corrected_seed_replication",
        "controlled_variable": "random seed only for cache-corrected L20 width-64 slider training",
        "scientific_question": "Is the strong exp146 cache-corrected L20 result (72.5/90/0 at seed 42) seed-robust?",
        "reference_seed42_result": {"exact_compliance_rate": 72.5, "unsafe_refusal_rate": 90.0, "benign_false_refusal_rate": 0.0},
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": WIDTH,
            "head_count": 1,
            "optimizer": "AdamW lr=3e-3 default betas",
            "steps": args.steps,
            "group_size": args.group_size,
            "training": "slider_head_fork-style grouped REINFORCE, no KL/BC",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
            "generation_cache_mode": "use_cache=False for training reward generation and behavioral evaluation",
        },
        "config": vars(args),
        "baseline": {
            "behavior_eval": baseline_eval,
            "benchmark": baseline_bench,
            "continuation_benchmark": baseline_cont,
            "distribution_diagnostic": baseline_diag,
        },
        "transformerlens_probe": tl_probe,
        "seeds": {},
    }

    for seed in SEEDS[: args.seed_count]:
        arm = f"seed_{seed}"
        print(f"[Run] {arm}: L20 width64 cache-corrected", flush=True)
        torch.manual_seed(seed)
        head = PlainSliderHead(sft.config.hidden_size, WIDTH).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity = float((head(probe) - probe).abs().max().item())
        if identity != 0.0:
            raise RuntimeError(f"zero-init identity failed for {arm}: {identity}")
        handle, trace = train_plain_slider(
            sft,
            tokenizer,
            rows,
            device,
            head,
            LAYER,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            max_new_tokens=args.max_new_tokens,
            generation_use_cache=False,
        )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, max_new_tokens=args.max_new_tokens, use_cache=False)
        sft.eval()
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
        diag = inv.distribution_diagnostic(sft, tokenizer, device, n=args.diagnostic_n)
        handle.remove()
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.5f}", flush=True)
        results["seeds"][str(seed)] = {
            "insert_layer": LAYER,
            "head_dim": WIDTH,
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": inv.metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": inv.metric_delta(bench, baseline_bench),
            "continuation_benchmark": continuation,
            "continuation_delta_vs_baseline": continuation_delta(continuation, baseline_cont),
            "distribution_diagnostic": diag,
            "diagnostic_delta_vs_baseline": inv.metric_delta(diag, baseline_diag),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "l20_cache_corrected_seed_replication_summary.json"
    write_json(path, results)
    print(f"[Exp 150 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
