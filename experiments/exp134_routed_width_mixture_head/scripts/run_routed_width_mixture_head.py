#!/usr/bin/env python3
"""Experiment 134: Routed width-mixture slider head.

Iteration-5 creative blending test. Three causal slider heads of widths 16/32/64
operate in parallel at the same safe insertion point (layer 27). Their outputs are
mixed behind one exact-zero scalar gate. The controlled comparison is whether a
tiny input-dependent softmax router improves on an identical fixed uniform mixture,
with fresh width-16 and width-64 single-head references measured in the same harness.

The training objective is fixed to the validated fork-style behavioral grouped
REINFORCE loss. Width behavior is therefore the isolated architectural variable;
capability movement is measured as an endpoint through legacy and continuation-only
benchmarks rather than optimized here.
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
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_capability_objective_slider as capobj  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
WIDTHS = [16, 32, 64]
ARMS = [
    "width16_single_reference",
    "width64_single_reference",
    "uniform_width_mixture_control",
    "routed_width_mixture",
]


class WidthMixtureHead(nn.Module):
    """Width-heterogeneous parallel heads behind one exact-zero global gate."""

    def __init__(self, dim: int, widths: list[int], routed: bool):
        super().__init__()
        self.widths = list(widths)
        self.routed = bool(routed)
        self.heads = nn.ModuleList([PlainSliderHead(dim, width, include_alpha=False) for width in self.widths])
        self.router = nn.Linear(dim, len(self.widths), bias=False)
        nn.init.zeros_(self.router.weight)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.observe = False
        self._weight_total: torch.Tensor | None = None
        self._weight_count = 0
        self._entropy_total = 0.0

    def mixture_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.routed:
            return torch.softmax(self.router(x), dim=-1)
        shape = (*x.shape[:-1], len(self.widths))
        return torch.full(shape, 1.0 / len(self.widths), device=x.device, dtype=x.dtype)

    def reset_observation(self) -> None:
        self._weight_total = None
        self._weight_count = 0
        self._entropy_total = 0.0

    def observation(self) -> dict:
        if not self.routed or self._weight_total is None or self._weight_count == 0:
            return {}
        mean = self._weight_total / self._weight_count
        return {
            "widths": self.widths,
            "mean_route_weight_by_width": {str(w): round(float(mean[i]), 6) for i, w in enumerate(self.widths)},
            "mean_route_entropy": round(self._entropy_total / self._weight_count, 6),
            "observed_positions": self._weight_count,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deltas = torch.stack([head.attention_delta(x) for head in self.heads], dim=2)  # B,T,H,D
        weights = self.mixture_weights(x)
        if self.observe and self.routed:
            with torch.no_grad():
                flat = weights.detach().float().reshape(-1, len(self.widths)).cpu()
                if self._weight_total is None:
                    self._weight_total = torch.zeros(len(self.widths), dtype=torch.float32)
                self._weight_total += flat.sum(dim=0)
                self._weight_count += int(flat.shape[0])
                entropy = -(flat.clamp_min(1e-12) * flat.clamp_min(1e-12).log()).sum(dim=-1)
                self._entropy_total += float(entropy.sum().item())
        mixed = (weights.unsqueeze(-1) * deltas).sum(dim=2)
        return x + self.alpha * mixed


def install_layer_module(model: torch.nn.Module, layer: int, module: nn.Module):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        shifted = module(hidden)
        return (shifted,) + tuple(out[1:]) if isinstance(out, tuple) else shifted

    return model.model.layers[layer].register_forward_hook(hook)


def build_arm(name: str, dim: int, device: torch.device) -> nn.Module:
    if name == "width16_single_reference":
        return PlainSliderHead(dim, 16).to(device)
    if name == "width64_single_reference":
        return PlainSliderHead(dim, 64).to(device)
    if name == "uniform_width_mixture_control":
        return WidthMixtureHead(dim, WIDTHS, routed=False).to(device)
    if name == "routed_width_mixture":
        return WidthMixtureHead(dim, WIDTHS, routed=True).to(device)
    raise ValueError(f"unknown arm: {name}")


def train_behavior_module(
    model: torch.nn.Module,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    module: nn.Module,
    layer: int,
    steps: int,
    group_size: int,
    lr: float,
    max_new_tokens: int,
) -> tuple[object, list[dict]]:
    handle = install_layer_module(model, layer, module)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(module.parameters(), lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        row = rows[step % len(rows)]
        behavior_loss, metrics = capobj.behavior_grouped_loss(model, tokenizer, row, device, group_size, max_new_tokens)
        model.train()
        module.train()
        opt.zero_grad(set_to_none=True)
        behavior_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0).item())
        opt.step()
        rec = {
            "step": step,
            **metrics,
            "prompt_action": row["target"]["action"],
            "alpha": round(float(module.alpha.detach().item()), 6),
            "loss": round(float(behavior_loss.detach().item()), 5),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        if isinstance(module, WidthMixtureHead):
            rec["router_weight_l2"] = round(float(module.router.weight.detach().norm().item()), 8)
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(f"    step {step:02d}: rew={metrics['mean_reward']:.2f} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}", flush=True)
    return handle, log


def zero_init_diff(module: nn.Module, dim: int, device: torch.device) -> float:
    with torch.no_grad():
        probe = torch.randn(1, 5, dim, device=device)
        return float((module(probe) - probe).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--continuation-n", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp134_routed_width_mixture_head/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 134] Routed width-mixture slider head on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(train_rows, 12)

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, use_cache=False)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline legacy benchmark] {baseline_bench}", flush=True)
    baseline_continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
    print(f"  [baseline continuation benchmark] {baseline_continuation}", flush=True)
    probe_rows = behavior_rows[:4] + behavior_rows[-4:]
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, probe_rows, device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp134_routed_width_mixture_head",
        "controlled_variable": "width-mixture architecture only: width16/width64 references vs fixed-uniform width mixture vs input-routed width mixture",
        "scientific_question": "Can a tiny learned softmax router blend width-specific slider-head functions better than a fixed uniform mixture at one safe insertion point?",
        "fixed_conditions": {
            "layer": LAYER,
            "mixture_widths": WIDTHS,
            "objective": "behavior_only grouped REINFORCE",
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas/weight decay",
            "steps": args.steps,
            "behavior_group_size": args.group_size,
            "behavior_prompt_stream": "12 benign then 12 unsafe, fixed order",
            "zero_init": "global alpha initialized exactly zero; mixture router initialized exactly zero",
            "generation_cache_mode": "use_cache=False for inserted-head generation and behavioral evaluation",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_continuation_benchmark": baseline_continuation,
        "transformerlens_probe": tl_probe,
        "arms": {},
    }

    for arm in ARMS:
        print(f"[Run] {arm}: fixed behavior objective, architecture only", flush=True)
        torch.manual_seed(args.seed)
        module = build_arm(arm, sft.config.hidden_size, device)
        identity_diff = zero_init_diff(module, sft.config.hidden_size, device)
        if identity_diff != 0.0:
            raise RuntimeError(f"zero-init identity failed for {arm}: {identity_diff}")
        handle, trace = train_behavior_module(
            sft,
            tokenizer,
            behavior_rows,
            device,
            module,
            LAYER,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            max_new_tokens=args.max_new_tokens,
        )
        if isinstance(module, WidthMixtureHead):
            module.observe = True
            module.reset_observation()
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, use_cache=False)
        route_stats = module.observation() if isinstance(module, WidthMixtureHead) else {}
        if isinstance(module, WidthMixtureHead):
            module.observe = False
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        continuation = run_continuation_benchmark(sft, tokenizer, device, args.continuation_n)
        handle.remove()
        print(
            f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} alpha={module.alpha.item():.5f}",
            flush=True,
        )
        results["arms"][arm] = {
            "architecture": arm,
            "parameter_count": head_parameter_count(module),
            "zero_init_identity_max_diff": identity_diff,
            "final_alpha": round(float(module.alpha.detach().item()), 6),
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": capobj.metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": capobj.metric_delta(bench, baseline_bench),
            "continuation_benchmark": continuation,
            "continuation_delta_vs_baseline": capobj.continuation_delta(continuation, baseline_continuation),
            "behavior_eval_route_observation": route_stats,
            "trace": trace,
        }
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "routed_width_mixture_head_summary.json"
    write_json(path, results)
    print(f"[Exp 134 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
