#!/usr/bin/env python3
"""Experiment 135: Capability/behavior head-blend interpolation.

Iteration-5 creative blending test and direct compositionality probe. Two identical
zero-init width-64 slider heads are trained separately at the same safe insertion
point (layer 27): one receives pure capability next-token loss, the other receives
pure behavioral grouped REINFORCE. Both heads are then frozen and their output
deltas are merged behind a learned scalar mixture coefficient. Only that scalar is
optimized in the final stage, using the same hybrid capability-primary objective as
exp132.

This preserves the validated freeze-then-merge principle: complex heads are learned
one at a time in isolation, then composition is learned with almost no free
parameters. The scientific question is whether capability and behavioral head
functions are additive/blendable in the residual stream, or whether they interfere.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp132_capability_objective_slider_ablation/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
from capability_objective import causal_lm_loss, corpus_batch, evaluate_lm_loss, load_benchmark_style_train_corpus  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_capability_objective_slider as capobj  # noqa: E402
from methodical_slider import PlainSliderHead, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
HEAD_DIM = 64


class DualHeadBlend(nn.Module):
    """Parallel composition of two trained slider heads behind one scalar mixture."""

    def __init__(self, capability_head: PlainSliderHead, behavior_head: PlainSliderHead):
        super().__init__()
        self.capability_head = capability_head
        self.behavior_head = behavior_head
        self.mix_logit = nn.Parameter(torch.zeros(1))

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        capability_weight = torch.sigmoid(self.mix_logit)
        return capability_weight, 1.0 - capability_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cap_delta = self.capability_head(x) - x
        behavior_delta = self.behavior_head(x) - x
        cap_w, behavior_w = self.weights()
        return x + cap_w * cap_delta + behavior_w * behavior_delta


def install_layer_module(model: torch.nn.Module, layer: int, module: nn.Module):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        shifted = module(hidden)
        return (shifted,) + tuple(out[1:]) if isinstance(out, tuple) else shifted

    return model.model.layers[layer].register_forward_hook(hook)


def zero_init_diff(module: nn.Module, dim: int, device: torch.device) -> float:
    with torch.no_grad():
        probe = torch.randn(1, 5, dim, device=device)
        return float((module(probe) - probe).abs().max().item())


def measure_state(
    label: str,
    model,
    tokenizer,
    eval_rows: list[dict],
    corpus_validation: list[dict],
    device: torch.device,
    bench_n: int,
    continuation_n: int,
    batch_size: int,
) -> dict:
    behavior = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device, use_cache=False)
    bench = run_benchmark(model, tokenizer, device, bench_n)
    continuation = run_continuation_benchmark(model, tokenizer, device, continuation_n)
    cap_val_loss = evaluate_lm_loss(model, tokenizer, corpus_validation, device, batch_size=batch_size)
    print(
        f"  [{label}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} cap_val={cap_val_loss}",
        flush=True,
    )
    return {
        "behavior_eval": behavior,
        "benchmark": bench,
        "continuation_benchmark": continuation,
        "capability_validation_loss": cap_val_loss,
    }


def train_mixture_coefficient(
    model,
    tokenizer,
    behavior_rows: list[dict],
    capability_rows: list[dict],
    device: torch.device,
    blend: DualHeadBlend,
    layer: int,
    steps: int,
    group_size: int,
    batch_size: int,
    lr: float,
    behavior_weight: float,
    max_new_tokens: int,
) -> tuple[object, list[dict]]:
    handle = install_layer_module(model, layer, blend)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in blend.capability_head.parameters():
        p.requires_grad_(False)
    for p in blend.behavior_head.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([blend.mix_logit], lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        batch = corpus_batch(capability_rows, step, batch_size)
        cap_loss = causal_lm_loss(model, tokenizer, [r["text"] for r in batch], device)
        row = behavior_rows[step % len(behavior_rows)]
        behavior_loss, behavior_metrics = capobj.behavior_grouped_loss(model, tokenizer, row, device, group_size, max_new_tokens)
        total_loss = cap_loss + behavior_weight * behavior_loss
        model.train()
        blend.train()
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_([blend.mix_logit], 1.0).item())
        opt.step()
        cap_w, behavior_w = blend.weights()
        rec = {
            "step": step,
            "capability_loss": round(float(cap_loss.detach().item()), 5),
            **behavior_metrics,
            "weighted_behavior_loss": round(float((behavior_weight * behavior_loss).detach().item()), 5),
            "prompt_action": row["target"]["action"],
            "mix_logit": round(float(blend.mix_logit.detach().item()), 6),
            "capability_weight": round(float(cap_w.detach().item()), 6),
            "behavior_weight": round(float(behavior_w.detach().item()), 6),
            "total_loss": round(float(total_loss.detach().item()), 5),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(
                f"    step {step:02d}: cap={rec['capability_loss']:.3f} rew={rec['mean_reward']:.2f} cap_w={rec['capability_weight']:.3f} grad={grad_norm:.5f}",
                flush=True,
            )
    return handle, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--continuation-n", type=int, default=100)
    parser.add_argument("--cap-items-per-source", type=int, default=64)
    parser.add_argument("--behavior-weight", type=float, default=10.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp135_capability_behavior_head_blend/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 135] Capability/behavior head-blend interpolation on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(train_rows, 12)
    corpus = load_benchmark_style_train_corpus(args.cap_items_per_source, args.seed)
    print(
        f"  capability corpus: train={len(corpus['train'])} validation={len(corpus['validation'])} sha256={corpus['sha256'][:12]} errors={list(corpus['errors'])}",
        flush=True,
    )

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, use_cache=False)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline legacy benchmark] {baseline_bench}", flush=True)
    baseline_continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
    print(f"  [baseline continuation benchmark] {baseline_continuation}", flush=True)
    baseline_cap_loss = evaluate_lm_loss(sft, tokenizer, corpus["validation"], device, batch_size=args.batch_size)
    print(f"  [baseline heldout capability loss] {baseline_cap_loss}", flush=True)
    probe_rows = behavior_rows[:4] + behavior_rows[-4:]
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, probe_rows, device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp135_capability_behavior_head_blend",
        "controlled_variable": "composition stage only within one architecture family: separately trained capability head, separately trained behavior head, fixed equal blend, learned-scalar blend",
        "scientific_question": "Are capability-trained and behavior-trained slider-head functions additive/blendable in the residual stream, or do they interfere?",
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": HEAD_DIM,
            "head_count": 2,
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas/weight decay",
            "steps_per_stage": args.steps,
            "capability_batch_size": args.batch_size,
            "behavior_group_size": args.group_size,
            "behavior_prompt_stream": "12 benign then 12 unsafe, fixed order",
            "blend_training_loss": "capability_loss + behavior_weight * behavior_loss",
            "freeze_then_merge": "capability and behavior heads frozen before mixture coefficient training",
            "generation_cache_mode": "use_cache=False for inserted-head generation and behavioral evaluation",
        },
        "objective_scaling": {"capability_weight": 1.0, "behavior_weight": args.behavior_weight},
        "capability_corpus": {
            "train_count": len(corpus["train"]),
            "validation_count": len(corpus["validation"]),
            "sha256": corpus["sha256"],
            "provenance": corpus["provenance"],
            "load_errors": corpus["errors"],
            "contamination_control": "uses only MMLU validation, ARC-Easy train, HellaSwag train; benchmark uses MMLU test, ARC-Easy test, HellaSwag validation",
        },
        "config": vars(args),
        "baseline": {
            "behavior_eval": baseline_eval,
            "benchmark": baseline_bench,
            "continuation_benchmark": baseline_continuation,
            "capability_validation_loss": baseline_cap_loss,
        },
        "transformerlens_probe": tl_probe,
        "stages": {},
    }

    def add_deltas(state: dict) -> dict:
        return {
            **state,
            "behavior_delta_vs_baseline": capobj.metric_delta(state["behavior_eval"], baseline_eval),
            "benchmark_delta_vs_baseline": capobj.metric_delta(state["benchmark"], baseline_bench),
            "continuation_delta_vs_baseline": capobj.continuation_delta(state["continuation_benchmark"], baseline_continuation),
            "capability_validation_loss_delta_vs_baseline": round(state["capability_validation_loss"] - baseline_cap_loss, 6),
        }

    print("[Stage] capability_alone: pure capability objective", flush=True)
    torch.manual_seed(args.seed)
    capability_head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
    cap_identity = zero_init_diff(capability_head, sft.config.hidden_size, device)
    if cap_identity != 0.0:
        raise RuntimeError(f"zero-init identity failed for capability head: {cap_identity}")
    cap_handle, cap_trace = capobj.train_objective_arm(
        "capability_only",
        sft,
        tokenizer,
        behavior_rows,
        corpus["train"],
        device,
        capability_head,
        LAYER,
        steps=args.steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        lr=args.lr,
        behavior_weight=args.behavior_weight,
        max_new_tokens=args.max_new_tokens,
    )
    cap_state = measure_state("capability_alone", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
    cap_handle.remove()
    results["stages"]["capability_alone"] = {
        **add_deltas(cap_state),
        "head_alpha": round(float(capability_head.alpha.detach().item()), 6),
        "zero_init_identity_max_diff": cap_identity,
        "trace": cap_trace,
    }

    print("[Stage] behavior_alone: pure behavioral objective", flush=True)
    torch.manual_seed(args.seed)
    behavior_head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
    behavior_identity = zero_init_diff(behavior_head, sft.config.hidden_size, device)
    if behavior_identity != 0.0:
        raise RuntimeError(f"zero-init identity failed for behavior head: {behavior_identity}")
    behavior_handle, behavior_trace = capobj.train_objective_arm(
        "behavior_only",
        sft,
        tokenizer,
        behavior_rows,
        corpus["train"],
        device,
        behavior_head,
        LAYER,
        steps=args.steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        lr=args.lr,
        behavior_weight=args.behavior_weight,
        max_new_tokens=args.max_new_tokens,
    )
    behavior_state = measure_state("behavior_alone", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
    behavior_handle.remove()
    results["stages"]["behavior_alone"] = {
        **add_deltas(behavior_state),
        "head_alpha": round(float(behavior_head.alpha.detach().item()), 6),
        "zero_init_identity_max_diff": behavior_identity,
        "trace": behavior_trace,
    }

    print("[Stage] initial_equal_blend: frozen heads, coefficient fixed at 0.5", flush=True)
    blend = DualHeadBlend(capability_head, behavior_head).to(device)
    initial_handle = install_layer_module(sft, LAYER, blend)
    initial_state = measure_state("initial_equal_blend", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
    initial_handle.remove()
    results["stages"]["initial_equal_blend"] = {
        **add_deltas(initial_state),
        "capability_weight": 0.5,
        "behavior_weight": 0.5,
    }

    print("[Stage] learned_blend: frozen heads, scalar mixture coefficient only", flush=True)
    torch.manual_seed(args.seed + 1)
    blend_handle, blend_trace = train_mixture_coefficient(
        sft,
        tokenizer,
        behavior_rows,
        corpus["train"],
        device,
        blend,
        LAYER,
        steps=args.steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        lr=args.lr,
        behavior_weight=args.behavior_weight,
        max_new_tokens=args.max_new_tokens,
    )
    learned_state = measure_state("learned_blend", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
    blend_handle.remove()
    cap_w, behavior_w = blend.weights()
    results["stages"]["learned_blend"] = {
        **add_deltas(learned_state),
        "final_mix_logit": round(float(blend.mix_logit.detach().item()), 6),
        "final_capability_weight": round(float(cap_w.detach().item()), 6),
        "final_behavior_weight": round(float(behavior_w.detach().item()), 6),
        "capability_head_alpha": round(float(capability_head.alpha.detach().item()), 6),
        "behavior_head_alpha": round(float(behavior_head.alpha.detach().item()), 6),
        "trace": blend_trace,
    }

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "capability_behavior_head_blend_summary.json"
    write_json(path, results)
    print(f"[Exp 135 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
