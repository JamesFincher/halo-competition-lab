#!/usr/bin/env python3
"""Experiment 140: answer-focused capability-objective weighting.

Exp132 showed that full-text capability training lowers validation loss but does
not produce robust continuation-benchmark gains. This experiment isolates the
objective-format variable: identical zero-init width-64 layer-27 slider heads are
trained with uniform full-text loss, 4x answer-upweighted loss, or answer-only
loss. The answer span is the final `Answer: `/`Continuation: ` payload, matching
the continuation-only benchmark measurement much more closely than uniform loss.
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
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
from capability_objective import corpus_batch, evaluate_lm_loss, evaluate_span_weighted_lm_loss, load_benchmark_style_train_corpus, span_weighted_causal_lm_loss  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_cache_corrected_capability_objective as caprep  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, install_plain_slider, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
HEAD_DIM = 64
ARMS = {
    "full_text_uniform_reference": (1.0, 1.0),
    "answer_upweighted_4x": (1.0, 4.0),
    "answer_only": (0.0, 1.0),
}


def measure_state(label, model, tokenizer, eval_rows, corpus_validation, device, bench_n, continuation_n, batch_size, max_new_tokens):
    behavior = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device, max_new_tokens=max_new_tokens, use_cache=False)
    bench = run_benchmark(model, tokenizer, device, bench_n)
    continuation = run_continuation_benchmark(model, tokenizer, device, continuation_n)
    full_val = evaluate_lm_loss(model, tokenizer, corpus_validation, device, batch_size=batch_size)
    answer_val = evaluate_span_weighted_lm_loss(model, tokenizer, corpus_validation, device, context_weight=0.0, answer_weight=1.0, batch_size=batch_size)
    print(
        f"  [{label}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} full_val={full_val} answer_val={answer_val}",
        flush=True,
    )
    return {
        "behavior_eval": behavior,
        "benchmark": bench,
        "continuation_benchmark": continuation,
        "capability_validation_loss": full_val,
        "answer_only_validation_loss": answer_val,
    }


def train_weighted_capability_arm(
    arm: str,
    weights: tuple[float, float],
    model,
    tokenizer,
    capability_rows: list[dict],
    device: torch.device,
    head: PlainSliderHead,
    steps: int,
    batch_size: int,
    lr: float,
) -> tuple[object, list[dict]]:
    context_weight, answer_weight = weights
    handle = install_plain_slider(model, LAYER, head)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        batch = corpus_batch(capability_rows, step, batch_size)
        loss = span_weighted_causal_lm_loss(
            model,
            tokenizer,
            [r["text"] for r in batch],
            device,
            context_weight=context_weight,
            answer_weight=answer_weight,
        )
        model.train()
        head.train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
        opt.step()
        rec = {
            "step": step,
            "objective": arm,
            "context_weight": context_weight,
            "answer_weight": answer_weight,
            "capability_sources": sorted({r["source"] for r in batch}),
            "span_weighted_loss": round(float(loss.detach().item()), 5),
            "alpha": round(float(head.alpha.detach().item()), 6),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(f"    {arm} step {step:02d}: loss={rec['span_weighted_loss']:.3f} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}", flush=True)
    return handle, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--continuation-n", type=int, default=150)
    parser.add_argument("--cap-items-per-source", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp140_answer_focused_capability_objective/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 140] Answer-focused capability objective on {device}", flush=True)

    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(base_rl.load_rows(ROOT / "data/policy_train.jsonl"), 12)
    corpus = load_benchmark_style_train_corpus(args.cap_items_per_source, args.seed)
    print(f"  capability corpus: train={len(corpus['train'])} validation={len(corpus['validation'])} sha256={corpus['sha256'][:12]} errors={list(corpus['errors'])}", flush=True)

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline = measure_state("baseline", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, behavior_rows[:4] + behavior_rows[-4:], device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp140_answer_focused_capability_objective",
        "controlled_variable": "relative context-vs-answer token weighting in the capability training loss",
        "scientific_question": "Does aligning capability training with answer-continuation scoring improve benchmark generalization relative to uniform full-text causal loss?",
        "motivation_from_exp132": "full-text capability-only training reduced validation loss by 0.268 but continuation-benchmark deltas were mixed/negative; objective-format mismatch is the first controlled follow-up",
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": HEAD_DIM,
            "head_count": 1,
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas/weight decay",
            "steps": args.steps,
            "capability_batch_size": args.batch_size,
            "zero_init": "alpha initialized exactly zero",
            "generation_cache_mode": "use_cache=False for behavioral evaluation; no behavioral generation is used in training",
        },
        "arms": {},
        "baseline": baseline,
        "capability_corpus": {
            "train_count": len(corpus["train"]),
            "validation_count": len(corpus["validation"]),
            "sha256": corpus["sha256"],
            "provenance": corpus["provenance"],
            "load_errors": corpus["errors"],
            "contamination_control": "uses only MMLU validation, ARC-Easy train, HellaSwag train; benchmark uses MMLU test, ARC-Easy test, HellaSwag validation",
        },
        "transformerlens_probe": tl_probe,
        "config": vars(args),
    }

    for arm, weights in ARMS.items():
        print(f"[Arm] {arm}: context_weight={weights[0]} answer_weight={weights[1]}", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity = float((head(probe) - probe).abs().max().item())
        if identity != 0.0:
            raise RuntimeError(f"zero-init identity failed for {arm}: {identity}")
        handle, trace = train_weighted_capability_arm(
            arm,
            weights,
            sft,
            tokenizer,
            corpus["train"],
            device,
            head,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        state = measure_state(arm, sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
        handle.remove()
        results["arms"][arm] = {
            **state,
            "behavior_delta_vs_baseline": caprep.metric_delta(state["behavior_eval"], baseline["behavior_eval"]),
            "benchmark_delta_vs_baseline": caprep.metric_delta(state["benchmark"], baseline["benchmark"]),
            "continuation_delta_vs_baseline": caprep.continuation_delta(state["continuation_benchmark"], baseline["continuation_benchmark"]),
            "capability_validation_loss_delta_vs_baseline": round(state["capability_validation_loss"] - baseline["capability_validation_loss"], 6),
            "answer_only_validation_loss_delta_vs_baseline": round(state["answer_only_validation_loss"] - baseline["answer_only_validation_loss"], 6),
            "context_weight": weights[0],
            "answer_weight": weights[1],
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "answer_focused_capability_objective_summary.json"
    write_json(path, results)
    print(f"[Exp 140 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
