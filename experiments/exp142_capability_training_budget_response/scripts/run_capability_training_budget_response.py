#!/usr/bin/env python3
"""Experiment 142: capability-training budget dose-response.

The first capability-objective arms used only 32 optimizer steps (128 examples),
less than one pass over even the small split-decontaminated corpus. This experiment
holds the benchmark-direct answer-only training condition fixed and varies only
optimization budget across 1x/4x/16x. It tests whether absent benchmark transfer is
an optimization-budget artifact rather than a saturation/architecture limitation.
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
sys.path.insert(0, str(ROOT / "experiments/exp139_cache_corrected_capability_objective/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp141_benchmark_format_capability/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
from capability_objective import corpus_batch, evaluate_lm_loss, evaluate_span_weighted_lm_loss, span_weighted_causal_lm_loss  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_cache_corrected_capability_objective as caprep  # noqa: E402
import run_benchmark_format_capability as fmt  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, install_plain_slider, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
HEAD_DIM = 64
BUDGET_MULTIPLIERS = {"budget_1x_32_steps": 1, "budget_4x_128_steps": 4, "budget_16x_512_steps": 16}


def measure_state(label, model, tokenizer, eval_rows, validation_records, device, bench_n, continuation_n, batch_size, max_new_tokens):
    behavior = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device, max_new_tokens=max_new_tokens, use_cache=False)
    bench = run_benchmark(model, tokenizer, device, bench_n)
    continuation = run_continuation_benchmark(model, tokenizer, device, continuation_n)
    starts = [r.get("answer_start_char") for r in validation_records]
    full_val = evaluate_lm_loss(model, tokenizer, validation_records, device, batch_size=batch_size)
    answer_val = evaluate_span_weighted_lm_loss(
        model,
        tokenizer,
        validation_records,
        device,
        context_weight=0.0,
        answer_weight=1.0,
        answer_starts=starts,
        batch_size=batch_size,
    )
    print(f"  [{label}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} full_val={full_val} answer_val={answer_val}", flush=True)
    return {
        "behavior_eval": behavior,
        "benchmark": bench,
        "continuation_benchmark": continuation,
        "full_validation_loss": full_val,
        "answer_only_validation_loss": answer_val,
    }


def train_budget_arm(records, model, tokenizer, device, head, steps, batch_size, lr, arm):
    handle = install_plain_slider(model, LAYER, head)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    log = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        batch = corpus_batch(records, step, batch_size)
        loss = span_weighted_causal_lm_loss(
            model,
            tokenizer,
            [r["text"] for r in batch],
            device,
            context_weight=0.0,
            answer_weight=1.0,
            answer_starts=[r.get("answer_start_char") for r in batch],
        )
        model.train(); head.train(); opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
        opt.step()
        rec = {
            "step": step,
            "answer_only_loss": round(float(loss.detach().item()), 5),
            "alpha": round(float(head.alpha.detach().item()), 6),
            "grad_norm": round(grad_norm, 5),
            "capability_sources": sorted({r["source"] for r in batch}),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 32 == 0 or step == steps - 1:
            print(f"    {arm} step {step:03d}/{steps}: loss={rec['answer_only_loss']:.3f} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}", flush=True)
    return handle, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32, help="base 1x budget; arms use 1x/4x/16x")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--continuation-n", type=int, default=150)
    parser.add_argument("--cap-items-per-source", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp142_capability_training_budget_response/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 142] Capability budget dose-response on {device}", flush=True)
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(base_rl.load_rows(ROOT / "data/policy_train.jsonl"), 12)
    corpus = fmt.load_benchmark_direct_corpus(args.cap_items_per_source, args.seed)
    train_records = corpus["train"]
    validation_records = corpus["validation"]
    print(f"  direct corpus train={len(train_records)} validation={len(validation_records)}", flush=True)

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    baseline = measure_state("baseline", sft, tokenizer, eval_rows, validation_records, device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, behavior_rows[:4] + behavior_rows[-4:], device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp142_capability_training_budget_response",
        "controlled_variable": "capability optimization budget only: 1x vs 4x vs 16x steps",
        "scientific_question": "Is missing benchmark transfer a consequence of severe undertraining, or does the inserted head saturate/damage capability as budget increases?",
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": HEAD_DIM,
            "training_format": "benchmark_direct",
            "objective": "answer-only weighted causal LM loss",
            "seed": args.seed,
            "base_steps": args.steps,
            "capability_batch_size": args.batch_size,
            "same_initial_head_weights": True,
            "shared_data_order_prefix": True,
            "generation_cache_mode": "use_cache=False for behavioral evaluation; no behavioral generation is used in training",
        },
        "baseline": baseline,
        "direct_corpus": {"train_count": len(train_records), "validation_count": len(validation_records), "sha256": corpus["sha256"], "provenance": corpus["provenance"], "load_errors": corpus["errors"]},
        "contamination_control": "uses only MMLU validation, ARC-Easy train, HellaSwag train; benchmark uses MMLU test, ARC-Easy test, HellaSwag validation",
        "transformerlens_probe": tl_probe,
        "config": vars(args),
        "arms": {},
    }

    for arm, multiplier in BUDGET_MULTIPLIERS.items():
        steps = args.steps * multiplier
        print(f"[Arm] {arm}: {steps} steps", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity = float((head(probe) - probe).abs().max().item())
        if identity != 0.0:
            raise RuntimeError(f"zero-init identity failed for {arm}: {identity}")
        handle, trace = train_budget_arm(train_records, sft, tokenizer, device, head, steps, args.batch_size, args.lr, arm)
        state = measure_state(arm, sft, tokenizer, eval_rows, validation_records, device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
        handle.remove()
        results["arms"][arm] = {
            **state,
            "behavior_delta_vs_baseline": caprep.metric_delta(state["behavior_eval"], baseline["behavior_eval"]),
            "benchmark_delta_vs_baseline": caprep.metric_delta(state["benchmark"], baseline["benchmark"]),
            "continuation_delta_vs_baseline": caprep.continuation_delta(state["continuation_benchmark"], baseline["continuation_benchmark"]),
            "full_validation_delta_vs_baseline": round(state["full_validation_loss"] - baseline["full_validation_loss"], 6),
            "answer_validation_delta_vs_baseline": round(state["answer_only_validation_loss"] - baseline["answer_only_validation_loss"], 6),
            "actual_steps": steps,
            "budget_multiplier": multiplier,
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "capability_training_budget_response_summary.json"
    write_json(path, results)
    print(f"[Exp 142 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
