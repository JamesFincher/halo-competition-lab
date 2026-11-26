#!/usr/bin/env python3
"""Experiment 138: sign-constrained subtractive slider-head gate.

A naive negative-alpha attention head is not a meaningful subtractive mechanism
when q/k/v/o train jointly, because those matrices can simply learn the opposite
sign. This experiment makes subtraction identifiable: first train one width-64
zero-init feature head at layer 27 with the hybrid capability-primary objective,
freeze its attention weights, copy them into two sign-constrained gate wrappers,
and retrain only a positive or negative softplus gate under identical data, seed,
budget, and objective. The controlled variable is deployment sign, not feature
weights or optimization budget.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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

LAYER = 27
HEAD_DIM = 64
GATE_ARMS = ["positive_gate_control", "negative_subtractive_gate"]


class SignedFeatureHead(nn.Module):
    """Frozen attention feature extractor behind a sign-constrained softplus gate."""

    def __init__(self, dim: int, head_dim: int, sign: float, initial_magnitude: float):
        super().__init__()
        self.core = PlainSliderHead(dim, head_dim)
        self.sign = float(sign)
        magnitude = max(float(initial_magnitude), 1e-8)
        inverse_softplus = math.log(math.expm1(magnitude))
        self.raw_alpha = nn.Parameter(torch.tensor([inverse_softplus], dtype=torch.float32))

    def gate(self) -> torch.Tensor:
        return self.sign * F.softplus(self.raw_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate() * self.core.attention_delta(x)


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


def module_gate_value(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "gate"):
        return module.gate().detach()
    return module.alpha.detach()


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


def train_hybrid_head(
    label: str,
    model,
    tokenizer,
    behavior_rows: list[dict],
    capability_rows: list[dict],
    device: torch.device,
    train_params: list[torch.nn.Parameter],
    head: nn.Module,
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
        head.train()
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(train_params, 1.0).item())
        opt.step()
        gate_value = float(module_gate_value(head).item())
        rec = {
            "step": step,
            "stage": label,
            "capability_loss": round(float(cap_loss.detach().item()), 5),
            **behavior_metrics,
            "weighted_behavior_loss": round(float((behavior_weight * behavior_loss).detach().item()), 5),
            "prompt_action": row["target"]["action"],
            "deployed_alpha_or_gate": round(gate_value, 7),
            "total_loss": round(float(total_loss.detach().item()), 5),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(
                f"    {label} step {step:02d}: cap={rec['capability_loss']:.3f} rew={rec['mean_reward']:.2f} gate={gate_value:.5f} grad={grad_norm:.3f}",
                flush=True,
            )
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--gate-steps", type=int, default=32)
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
    out_dir = ROOT / "experiments/exp138_subtractive_frozen_head_gate/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 138] Sign-constrained subtractive slider gate on {device}", flush=True)

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
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, [LAYER])
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp138_subtractive_frozen_head_gate",
        "controlled_variable": "sign of the deployed gate on identical frozen attention-feature weights",
        "scientific_question": "Can a sign-constrained subtractive gate remove interference from a learned residual feature better than the matched positive gate?",
        "identifiability_note": "q/k/v/o are frozen before sign-gate retraining; otherwise a negative scalar gate would be mathematically redundant because the feature extractor could invert its output weights.",
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": HEAD_DIM,
            "objective": "hybrid capability-primary",
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas/weight decay",
            "feature_steps": args.steps,
            "gate_steps": args.gate_steps,
            "capability_batch_size": args.batch_size,
            "behavior_group_size": args.group_size,
            "behavior_prompt_stream": "12 benign then 12 unsafe, fixed order",
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

    print("[Stage] train_shared_feature_head: unconstrained zero-init hybrid head", flush=True)
    torch.manual_seed(args.seed)
    feature_head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
    identity = zero_init_diff(feature_head, sft.config.hidden_size, device)
    if identity != 0.0:
        raise RuntimeError(f"zero-init identity failed: {identity}")
    handle = install_layer_module(sft, LAYER, feature_head)
    feature_trace = train_hybrid_head(
        "train_shared_feature_head",
        sft,
        tokenizer,
        behavior_rows,
        corpus["train"],
        device,
        list(feature_head.parameters()),
        feature_head,
        steps=args.steps,
        group_size=args.group_size,
        batch_size=args.batch_size,
        lr=args.lr,
        behavior_weight=args.behavior_weight,
        max_new_tokens=args.max_new_tokens,
    )
    feature_state = measure_state("trained_feature_additive", sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
    handle.remove()
    trained_alpha = float(feature_head.alpha.detach().item())
    initial_gate_magnitude = max(abs(trained_alpha), 0.001)
    feature_weights = {k: v.detach().cpu().clone() for k, v in feature_head.state_dict().items()}
    results["stages"]["trained_feature_additive"] = {
        **add_deltas(feature_state),
        "trained_alpha": round(trained_alpha, 7),
        "zero_init_identity_max_diff": identity,
        "trace": feature_trace,
    }
    del feature_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for arm in GATE_ARMS:
        sign = 1.0 if arm == "positive_gate_control" else -1.0
        print(f"[Arm] {arm}: frozen feature weights, sign-constrained gate only", flush=True)
        torch.manual_seed(args.seed + 200)
        signed = SignedFeatureHead(sft.config.hidden_size, HEAD_DIM, sign, initial_gate_magnitude).to(device)
        signed.core.load_state_dict(feature_weights)
        for p in signed.core.parameters():
            p.requires_grad_(False)
        handle = install_layer_module(sft, LAYER, signed)
        gate_trace = train_hybrid_head(
            arm,
            sft,
            tokenizer,
            behavior_rows,
            corpus["train"],
            device,
            [signed.raw_alpha],
            signed,
            steps=args.gate_steps,
            group_size=args.group_size,
            batch_size=args.batch_size,
            lr=args.lr,
            behavior_weight=args.behavior_weight,
            max_new_tokens=args.max_new_tokens,
        )
        arm_state = measure_state(arm, sft, tokenizer, eval_rows, corpus["validation"], device, args.bench_n, args.continuation_n, args.batch_size)
        handle.remove()
        results["stages"][arm] = {
            **add_deltas(arm_state),
            "parameter_count": head_parameter_count(signed),
            "trainable_parameter_count": 1,
            "initial_gate_magnitude": initial_gate_magnitude,
            "initial_gate": round(sign * initial_gate_magnitude, 7),
            "final_gate": round(float(signed.gate().detach().item()), 7),
            "sign_constraint": "positive softplus" if sign > 0 else "negative softplus",
            "trace": gate_trace,
        }
        del signed
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "subtractive_frozen_head_gate_summary.json"
    write_json(path, results)
    print(f"[Exp 138 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
