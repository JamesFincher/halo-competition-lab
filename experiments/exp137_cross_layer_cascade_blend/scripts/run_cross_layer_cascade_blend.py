#!/usr/bin/env python3
"""Experiment 137: Cross-layer cascade slider-head composition.

Iteration-5 vertical layering test. A width-64 zero-init slider head at layer 24
is first trained bottom-up with the same hybrid capability-primary objective used
in the blending family. Its weights are then frozen and copied into two matched
upper-head arms at layer 27:

1. parallel control: the upper head reads only its local layer-27 residual state;
2. cascade: the upper head's attention input additionally receives the frozen
   lower head's output delta through a zero-initialized trainable scalar gate.

The upper head itself is zero-init in both arms, and only upper-head parameters
plus the scalar cascade gate train in stage 2. The controlled variable is therefore
the direct cross-layer input connection, not lower-head training, width, objective,
seed, insertion layers, or optimization budget.
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
from methodical_slider import PlainSliderHead, head_parameter_count, ordered_fork_rows, probe_layers_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LOWER_LAYER = 24
UPPER_LAYER = 27
HEAD_DIM = 64
ARMS = ["parallel_upper_control", "cross_layer_cascade"]


class CrossLayerCascadePair(nn.Module):
    """Lower/upper slider pair with an optional gated lower-to-upper input path."""

    def __init__(self, dim: int, head_dim: int, cascade_enabled: bool):
        super().__init__()
        self.lower = PlainSliderHead(dim, head_dim)
        self.upper = PlainSliderHead(dim, head_dim)
        self.cascade_enabled = bool(cascade_enabled)
        self.skip_scale = nn.Parameter(torch.zeros(1))
        self._lower_output: torch.Tensor | None = None

    def lower_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lower(x)
        self._lower_output = (out - x).detach()
        return out

    def upper_forward(self, x: torch.Tensor) -> torch.Tensor:
        upper_input = x
        if self.cascade_enabled and self._lower_output is not None:
            lower_output = self._lower_output.to(device=x.device, dtype=x.dtype)
            if lower_output.shape == x.shape:
                upper_input = x + self.skip_scale * lower_output
        delta = self.upper.attention_delta(upper_input)
        return x + self.upper.alpha * delta


def hook_output(module: nn.Module, layer: int, fn):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        shifted = fn(hidden)
        return (shifted,) + tuple(out[1:]) if isinstance(out, tuple) else shifted

    return module.model.layers[layer].register_forward_hook(hook)


def install_lower(model, pair: CrossLayerCascadePair):
    return hook_output(model, LOWER_LAYER, pair.lower_forward)


def install_pair(model, pair: CrossLayerCascadePair) -> list[object]:
    return [install_lower(model, pair), hook_output(model, UPPER_LAYER, pair.upper_forward)]


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


def train_hybrid_stage(
    label: str,
    model,
    tokenizer,
    behavior_rows: list[dict],
    capability_rows: list[dict],
    device: torch.device,
    train_params: list[torch.nn.Parameter],
    pair: CrossLayerCascadePair,
    steps: int,
    group_size: int,
    batch_size: int,
    lr: float,
    behavior_weight: float,
    max_new_tokens: int,
) -> list[dict]:
    opt = torch.optim.AdamW(train_params, lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        batch = corpus_batch(capability_rows, step, batch_size)
        cap_loss = causal_lm_loss(model, tokenizer, [r["text"] for r in batch], device)
        row = behavior_rows[step % len(behavior_rows)]
        behavior_loss, behavior_metrics = capobj.behavior_grouped_loss(model, tokenizer, row, device, group_size, max_new_tokens)
        total_loss = cap_loss + behavior_weight * behavior_loss
        model.train()
        pair.train()
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(train_params, 1.0).item())
        opt.step()
        rec = {
            "step": step,
            "stage": label,
            "capability_loss": round(float(cap_loss.detach().item()), 5),
            **behavior_metrics,
            "weighted_behavior_loss": round(float((behavior_weight * behavior_loss).detach().item()), 5),
            "prompt_action": row["target"]["action"],
            "lower_alpha": round(float(pair.lower.alpha.detach().item()), 6),
            "upper_alpha": round(float(pair.upper.alpha.detach().item()), 6),
            "skip_scale": round(float(pair.skip_scale.detach().item()), 6),
            "total_loss": round(float(total_loss.detach().item()), 5),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(
                f"    {label} step {step:02d}: cap={rec['capability_loss']:.3f} rew={rec['mean_reward']:.2f} lower_a={rec['lower_alpha']:.4f} upper_a={rec['upper_alpha']:.4f} skip={rec['skip_scale']:.4f} grad={grad_norm:.3f}",
                flush=True,
            )
    return log


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
    out_dir = ROOT / "experiments/exp137_cross_layer_cascade_blend/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 137] Cross-layer cascade slider composition on {device}", flush=True)

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
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, [LOWER_LAYER, UPPER_LAYER])
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp137_cross_layer_cascade_blend",
        "controlled_variable": "direct lower-to-upper cross-layer input connection only",
        "scientific_question": "Does explicit vertical composition outperform parallel insertion when lower-head weights, upper-head initialization, objective, seed, and budget are held fixed?",
        "fixed_conditions": {
            "lower_layer": LOWER_LAYER,
            "upper_layer": UPPER_LAYER,
            "head_dim": HEAD_DIM,
            "objective": "hybrid capability-primary",
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas/weight decay",
            "steps_per_stage": args.steps,
            "capability_batch_size": args.batch_size,
            "behavior_group_size": args.group_size,
            "behavior_prompt_stream": "12 benign then 12 unsafe, fixed order",
            "training_recipe": "train lower alone, freeze/copy identical lower weights into both arms, then train upper sequentially",
            "zero_init": "lower and upper alphas initialized exactly zero; cascade skip scale initialized exactly zero",
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

    print("[Stage] shared_lower_bottom_up: train one lower L24 head hybrid", flush=True)
    torch.manual_seed(args.seed)
    shared_pair = CrossLayerCascadePair(sft.config.hidden_size, HEAD_DIM, cascade_enabled=True).to(device)
    lower_identity = zero_init_diff(shared_pair.lower, sft.config.hidden_size, device)
    upper_identity = zero_init_diff(shared_pair.upper, sft.config.hidden_size, device)
    if lower_identity != 0.0 or upper_identity != 0.0:
        raise RuntimeError(f"zero-init identity failed: lower={lower_identity} upper={upper_identity}")
    lower_handle = install_lower(sft, shared_pair)
    lower_trace = train_hybrid_stage(
        "shared_lower_bottom_up",
        sft,
        tokenizer,
        behavior_rows,
        corpus["train"],
        device,
        list(shared_pair.lower.parameters()),
        shared_pair,
        steps=args.steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        lr=args.lr,
        behavior_weight=args.behavior_weight,
        max_new_tokens=args.max_new_tokens,
    )
    lower_state = measure_state("shared_lower_alone", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
    lower_handle.remove()
    lower_weights = {k: v.detach().cpu().clone() for k, v in shared_pair.lower.state_dict().items()}
    results["stages"]["shared_lower_alone"] = {
        **add_deltas(lower_state),
        "lower_alpha": round(float(shared_pair.lower.alpha.detach().item()), 6),
        "zero_init_identity_max_diff": lower_identity,
        "trace": lower_trace,
    }
    del shared_pair
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for arm in ARMS:
        cascade = arm == "cross_layer_cascade"
        print(f"[Arm] {arm}: same frozen lower head, connection only", flush=True)
        torch.manual_seed(args.seed + 100)
        pair = CrossLayerCascadePair(sft.config.hidden_size, HEAD_DIM, cascade_enabled=cascade).to(device)
        pair.lower.load_state_dict(lower_weights)
        for p in pair.lower.parameters():
            p.requires_grad_(False)
        if not cascade:
            pair.skip_scale.requires_grad_(False)
        handles = install_pair(sft, pair)
        train_params = list(pair.upper.parameters()) + ([pair.skip_scale] if cascade else [])
        upper_trace = train_hybrid_stage(
            arm,
            sft,
            tokenizer,
            behavior_rows,
            corpus["train"],
            device,
            train_params,
            pair,
            steps=args.steps,
            group_size=args.group_size,
            batch_size=args.batch_size,
            lr=args.lr,
            behavior_weight=args.behavior_weight,
            max_new_tokens=args.max_new_tokens,
        )
        arm_state = measure_state(arm, sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
        for handle in handles:
            handle.remove()
        results["stages"][arm] = {
            **add_deltas(arm_state),
            "parameter_count": head_parameter_count(pair),
            "lower_alpha_fixed": round(float(pair.lower.alpha.detach().item()), 6),
            "upper_alpha": round(float(pair.upper.alpha.detach().item()), 6),
            "final_skip_scale": round(float(pair.skip_scale.detach().item()), 8),
            "cascade_enabled": cascade,
            "zero_init_identity_max_diff": upper_identity,
            "trace": upper_trace,
        }
        del pair
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "cross_layer_cascade_blend_summary.json"
    write_json(path, results)
    print(f"[Exp 137 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
