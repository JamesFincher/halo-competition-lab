"""Exp 151: cache-corrected depth re-ablation of width-64 plain slider insertion.

Staged until exp150 establishes whether the cache-corrected L20 result is seed-robust.
This rerun removes the exp91 generation-cache confound and re-measures insertion depth
under one fixed protocol: width 64, same prompt stream, steps, LR, seeds, cache-free
reward generation, cache-free behavioral eval, continuation benchmark, and logit diagnostics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "experiments/exp146_cache_mode_inversion_diagnostic/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cache_mode_inversion_diagnostic as inv  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402
from methodical_slider import PlainSliderHead, ordered_fork_rows, train_plain_slider  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from serialization_utils import write_json  # noqa: E402

DEFAULT_LAYERS = [20, 24, 27, 29]
SEEDS = [7, 42, 123]
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


def parse_layers(text: str) -> list[int]:
    layers = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed-count", type=int, default=len(SEEDS))
    ap.add_argument("--layers", type=str, default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--bench-n", type=int, default=50)
    ap.add_argument("--continuation-n", type=int, default=100)
    ap.add_argument("--diagnostic-n", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    seeds = SEEDS[: max(1, int(args.seed_count))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp 151] cache-corrected depth re-ablation on {device}", flush=True)
    print(f"  layers={layers}; seeds={seeds}; width={WIDTH}", flush=True)
    t0 = time.time()

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ordered = ordered_fork_rows(train_rows, 12)

    sft.eval()
    with torch.no_grad():
        baseline_behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, max_new_tokens=args.max_new_tokens, use_cache=False)
    print(
        f"  [baseline behavior] exact={baseline_behavior['exact_compliance_rate']}% "
        f"unsafe={baseline_behavior['unsafe_refusal_rate']}% benign_fr={baseline_behavior['benign_false_refusal_rate']}%",
        flush=True,
    )
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    baseline_cont = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
    baseline_diag = inv.distribution_diagnostic(sft, tokenizer, device, args.diagnostic_n)

    matrix = {}
    for layer in layers:
        layer_rec = {}
        for seed in seeds:
            name = f"L{layer}_seed{seed}"
            print(f"[Run] {name}: width{WIDTH} cache-corrected", flush=True)
            torch.manual_seed(seed)
            head = PlainSliderHead(sft.config.hidden_size, WIDTH).to(device)
            handle, trace = train_plain_slider(
                sft,
                tokenizer,
                ordered,
                device,
                head,
                layer,
                steps=args.steps,
                group_size=args.group_size,
                lr=args.lr,
                max_new_tokens=args.max_new_tokens,
                generation_use_cache=False,
            )
            sft.eval()
            with torch.no_grad():
                behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, max_new_tokens=args.max_new_tokens, use_cache=False)
            bench = run_benchmark(sft, tokenizer, device, args.bench_n)
            continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
            diag = inv.distribution_diagnostic(sft, tokenizer, device, args.diagnostic_n)
            handle.remove()
            layer_rec[str(seed)] = {
                "layer": layer,
                "seed": seed,
                "width": WIDTH,
                "cache_corrected": True,
                "behavior_eval": behavior,
                "behavior_delta_vs_baseline": inv.metric_delta(behavior, baseline_behavior),
                "benchmark": bench,
                "continuation_benchmark": continuation,
                "continuation_delta_vs_baseline": continuation_delta(continuation, baseline_cont),
                "distribution_diagnostic": diag,
                "distribution_delta_vs_baseline": inv.metric_delta(diag, baseline_diag),
                "final_alpha": round(float(head.alpha.detach().cpu().item()), 6),
                "head_parameter_count": int(sum(p.numel() for p in head.parameters())),
                "trace": trace,
            }
            print(
                f"  [{name}] exact={behavior['exact_compliance_rate']}% unsafe={behavior['unsafe_refusal_rate']}% "
                f"benign_fr={behavior['benign_false_refusal_rate']}% alpha={float(head.alpha.detach().cpu().item()):.5f}",
                flush=True,
            )
            del head
            if device.type == "cuda":
                torch.cuda.empty_cache()
        matrix[str(layer)] = layer_rec

    out = {
        "experiment": "exp151_cache_corrected_depth_reablation",
        "controlled_variable": "slider insertion depth under cache-corrected generation, replicated across seeds",
        "scientific_question": "After removing the exp91 cache confound, is L20 genuinely superior to L24/L27/L29, and is that ordering seed-robust?",
        "fixed_conditions": {
            "width": WIDTH,
            "layers": layers,
            "seeds": seeds,
            "steps": args.steps,
            "group_size": args.group_size,
            "lr": args.lr,
            "ordered_prompt_stream": "12 call_tool then 12 refuse",
            "reward_generation_use_cache": False,
            "behavioral_eval_use_cache": False,
            "zero_init": True,
        },
        "config": vars(args),
        "baseline_eval": baseline_behavior,
        "baseline_benchmark": baseline_bench,
        "baseline_continuation_benchmark": baseline_cont,
        "baseline_distribution_diagnostic": baseline_diag,
        "exp146_seed42_reference": {
            "layer20_cache_corrected_behavior": {"exact": 72.5, "unsafe": 90.0, "benign_fr": 0.0},
            "layer27_cache_corrected_behavior": {"exact": 45.0, "unsafe": 100.0, "benign_fr": 100.0},
        },
        "depth_seed_matrix": matrix,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_dir = ROOT / "experiments/exp151_cache_corrected_depth_reablation/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cache_corrected_depth_reablation_summary.json"
    write_json(path, out)
    print(f"[Exp 151 DONE] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
