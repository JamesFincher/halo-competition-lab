#!/usr/bin/env python3
"""Experiment 103: Failure-correlated neuron suppression dose-response.

TransformerLens identifies final-layer MLP neurons whose post-activation is
consistently stronger under canonical false refusal than correct benign compliance
on paired prompts. The fixed top-64 neuron set is then suppressed by scaling their
mlp.down_proj output columns. Only suppression strength varies.
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
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from tlens_utils import cache_text, load_hooked_hf_model, mlp_activation_hook_name  # noqa: E402
from methodical_slider import ordered_fork_rows, probe_layer_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402

LAYER = 29
TOP_NEURONS = 64
STRENGTHS = [0.25, 0.50, 1.00]
CANONICAL_REFUSAL = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def mean_mlp_post(tl, tokenizer, texts: list[str], layer: int) -> torch.Tensor:
    hook = mlp_activation_hook_name(layer)
    vals = []
    with torch.no_grad():
        for text in texts:
            _tokens, _logits, cache = cache_text(tl, text)
            act = cache[hook][0, -24:, :].detach().float().mean(dim=0).cpu()
            vals.append(act)
    return torch.stack(vals, dim=0)


def select_failure_neurons(sft, tokenizer, train_rows: list[dict], device: torch.device, n_pairs: int, top_n: int) -> dict:
    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_pairs]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][:n_pairs]
    pass_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in benign_rows]
    fail_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + CANONICAL_REFUSAL for r in benign_rows]
    unsafe_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in unsafe_rows]

    tl = load_hooked_hf_model(sft, tokenizer, device, torch.float32)
    h_pass = mean_mlp_post(tl, tokenizer, pass_texts, LAYER)
    h_fail = mean_mlp_post(tl, tokenizer, fail_texts, LAYER)
    h_unsafe = mean_mlp_post(tl, tokenizer, unsafe_texts, LAYER)
    del tl
    if device.type == "cuda":
        torch.cuda.empty_cache()

    paired_delta = h_fail - h_pass
    mean_delta = paired_delta.mean(dim=0)
    consistency = ((paired_delta * mean_delta.unsqueeze(0)) > 0).float().mean(dim=0)
    failure_score = mean_delta * consistency
    unsafe_association = (h_unsafe.mean(dim=0) - h_pass.mean(dim=0))
    positive = torch.nonzero(failure_score > 0).flatten()
    if positive.numel() == 0:
        raise RuntimeError("no positive failure-correlated MLP neurons found")
    k = min(top_n, int(positive.numel()))
    ranked = positive[torch.topk(failure_score[positive], k=k).indices]
    return {
        "neuron_indices": [int(x) for x in ranked.tolist()],
        "failure_scores": [round(float(x), 8) for x in failure_score[ranked].tolist()],
        "mean_failure_delta": [round(float(x), 8) for x in mean_delta[ranked].tolist()],
        "paired_sign_consistency": [round(float(x), 4) for x in consistency[ranked].tolist()],
        "unsafe_association": [round(float(x), 8) for x in unsafe_association[ranked].tolist()],
        "positive_failure_neuron_count": int(positive.numel()),
        "activation_hook": mlp_activation_hook_name(LAYER),
        "selection_rule": "top positive mean(paired false-refusal minus correct-compliance activation) weighted by paired sign consistency",
    }


def apply_neuron_suppression(model: torch.nn.Module, neuron_indices: list[int], strength: float) -> dict:
    with torch.no_grad():
        down = model.model.layers[LAYER].mlp.down_proj
        w = down.weight.detach().float().clone()
        idx = torch.tensor(neuron_indices, device=w.device, dtype=torch.long)
        before = w[:, idx].norm(dim=0).mean().item()
        w[:, idx] *= (1.0 - strength)
        after = w[:, idx].norm(dim=0).mean().item()
        down.weight.data.copy_(w.to(down.weight.dtype))
    return {
        "layer": LAYER,
        "neuron_count": len(neuron_indices),
        "edited_weight_count": int(model.config.hidden_size * len(neuron_indices)),
        "mean_selected_column_norm_before": round(before, 8),
        "mean_selected_column_norm_after": round(after, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=12)
    parser.add_argument("--bench-n", type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp103_failure_neuron_suppression_dose/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 103] Failure-neuron suppression dose-response on {device}", flush=True)

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
        baseline_probe = probe_layer_geometry(sft, tokenizer, probe_rows, device, LAYER)
    except Exception as exc:
        baseline_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [baseline TransformerLens probe failed; continuing] {baseline_probe['error']}", flush=True)

    selection = select_failure_neurons(sft, tokenizer, train_rows, device, args.n_pairs, TOP_NEURONS)
    print(f"  [selection] top={TOP_NEURONS} positive_candidates={selection['positive_failure_neuron_count']} first_scores={selection['failure_scores'][:5]}", flush=True)
    results = {
        "experiment": "exp103_failure_neuron_suppression_dose",
        "controlled_variable": "suppression strength only (0.25/0.50/1.00) for one fixed failure-correlated neuron set",
        "fixed_conditions": {
            "layer": LAYER,
            "top_neurons": TOP_NEURONS,
            "selection_pairs": args.n_pairs,
            "suppression_operator": "scale selected mlp.down_proj output columns by (1-strength)",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_transformerlens_probe": baseline_probe,
        "neuron_selection": selection,
        "arms": {},
    }
    for strength in STRENGTHS:
        arm = f"suppression_{str(strength).replace('.', 'p')}"
        print(f"[Run] {arm}: fixed top-{TOP_NEURONS} L29 failure-correlated neurons", flush=True)
        edited = copy.deepcopy(sft).to(device).eval()
        edit_info = apply_neuron_suppression(edited, selection["neuron_indices"], strength)
        behavior = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
        bench = run_benchmark(edited, tokenizer, device, args.bench_n)
        try:
            probe = probe_layer_geometry(edited, tokenizer, probe_rows, device, LAYER)
        except Exception as exc:
            probe = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{arm} TransformerLens probe failed; continuing] {probe['error']}", flush=True)
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench}", flush=True)
        results["arms"][arm] = {
            "strength": strength,
            "edit_info": edit_info,
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
    path = out_dir / "failure_neuron_suppression_dose_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 103 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
