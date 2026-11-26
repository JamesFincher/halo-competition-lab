#!/usr/bin/env python3
"""Experiment 67: Norm-calibrated single-coordinate decision steering.

Implements H16: find sparse MLP coordinates whose correct-vs-incorrect completion
activation differs reliably, then shift one coordinate by a norm-calibrated dose.
Decision-token mode hooks only cached single-token decoding steps; all-token mode
also shifts prompt/prefill positions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402

LAYERS = [24, 25, 26, 27, 28, 29]
HUBS = {(27, 110), (29, 1512), (29, 1022)}
WRONG_REFUSAL = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'
WRONG_CALL = '{"action": "call_tool", "tool": "send_email", "policy_code": "ALLOW", "reason": "Done."}'


def wrong_completion(row: dict) -> str:
    return WRONG_REFUSAL if row["target"]["action"] == "call_tool" else WRONG_CALL


def capture_down_inputs(model, tokenizer, rows: list[dict], device: torch.device, correct: bool) -> dict[int, list[torch.Tensor]]:
    out = {li: [] for li in LAYERS}
    model.eval()
    for row in rows:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        text = prompt + (row["assistant_text"] if correct else wrong_completion(row))
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320).to(device)
        plen = len(tokenizer(prompt, truncation=True, max_length=256)["input_ids"])
        caps: dict[int, torch.Tensor] = {}
        handles = []
        for li in LAYERS:
            def pre(_mod, inp, li=li):
                x = inp[0].detach()
                lo = min(plen, max(x.shape[1] - 1, 0))
                caps[li] = x[0, lo:, :].float().mean(dim=0).cpu()
            handles.append(model.model.layers[li].mlp.down_proj.register_forward_pre_hook(pre))
        with torch.no_grad():
            model(**enc)
        for h in handles:
            h.remove()
        for li in LAYERS:
            if li in caps:
                out[li].append(caps[li])
    return out


def capture_residual_norms(model, tokenizer, rows: list[dict], device: torch.device) -> dict[int, float]:
    vals = {li: [] for li in LAYERS}
    handles = []
    for li in LAYERS:
        def hk(_m, _i, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            vals[li].append(float(h[0, -1, :].detach().float().norm().cpu()))
        handles.append(model.model.layers[li].register_forward_hook(hk))
    try:
        with torch.no_grad():
            for row in rows:
                prompt = base_rl.build_prompt(tokenizer, row["messages"])
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                model(**enc)
    finally:
        for h in handles:
            h.remove()
    return {li: sorted(vals[li])[len(vals[li]) // 2] for li in LAYERS}


def rank_coordinates(correct: dict[int, list[torch.Tensor]], incorrect: dict[int, list[torch.Tensor]]) -> dict[int, dict]:
    ranking = {}
    for li in LAYERS:
        xc = torch.stack(correct[li])
        xi = torch.stack(incorrect[li])
        n = xc.shape[0]
        xa, xb = xc[: n // 2], xc[n // 2 :]
        ya, yb = xi[: n // 2], xi[n // 2 :]
        diff = xc.mean(0) - xi.mean(0)
        da = xa.mean(0) - ya.mean(0)
        db = xb.mean(0) - yb.mean(0)
        stability = torch.sign(da) * torch.sign(db)
        denom = xc.std(dim=0, unbiased=False) + xi.std(dim=0, unbiased=False) + 0.05 * diff.abs().mean()
        score = diff.abs() / denom * torch.relu(stability)
        for neuron in torch.topk(score, k=min(16, score.numel())).indices.tolist():
            if (li, int(neuron)) not in HUBS:
                ranking[li] = {
                    "neuron": int(neuron),
                    "direction_sign": int(torch.sign(diff[neuron]).item() or 1),
                    "score": round(float(score[neuron]), 5),
                    "mean_correct": round(float(xc[:, neuron].mean()), 5),
                    "mean_incorrect": round(float(xi[:, neuron].mean()), 5),
                    "stability": round(float(stability[neuron]), 4),
                }
                break
    return ranking


def steering_hook(model, layer: int, neuron: int, shift: float, mode: str):
    def pre(_mod, inp):
        x = inp[0]
        y = x.clone()
        if mode == "decision":
            if x.shape[1] == 1:
                y[..., neuron] += shift
            else:
                # Prefill has no generated completion positions yet; steering the final
                # prompt position is the only way to influence the first emitted decision token.
                y[:, -1, neuron] += shift
        else:
            y[..., neuron] += shift
        return (y,)

    return model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=8)
    parser.add_argument("--dose", type=float, default=0.10)
    parser.add_argument("--high-dose", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp67_single_coordinate_decision_steering/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 67] Single-coordinate decision steering on {device}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][: args.n_per_class]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][: args.n_per_class]
    rows = []
    for pair in zip(benign_rows, unsafe_rows):
        rows.extend(pair)
    model, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    baseline = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
    print(f"  [baseline] exact={baseline['exact_compliance_rate']:.1f}% unsafe={baseline['unsafe_refusal_rate']:.1f}% benign_fr={baseline['benign_false_refusal_rate']:.1f}%", flush=True)

    correct = capture_down_inputs(model, tokenizer, rows, device, True)
    incorrect = capture_down_inputs(model, tokenizer, rows, device, False)
    ranking = rank_coordinates(correct, incorrect)
    best_layer = max(ranking, key=lambda li: ranking[li]["score"])
    sel = ranking[best_layer]
    neuron = sel["neuron"]
    residual_norms = capture_residual_norms(model, tokenizer, rows, device)
    col_norm = float(model.model.layers[best_layer].mlp.down_proj.weight.detach().float()[:, neuron].norm().item())
    unit_shift = residual_norms[best_layer] / max(col_norm, 1e-6)
    aligned_unit = sel["direction_sign"] * unit_shift
    inter_dim = model.config.intermediate_size
    random_neuron = random.choice([j for j in range(inter_dim) if (best_layer, j) not in HUBS and j != neuron])

    arms = [
        ("aligned_decision_dose", neuron, aligned_unit * args.dose, "decision"),
        ("opposite_decision_dose", neuron, -aligned_unit * args.dose, "decision"),
        ("aligned_all_token_dose", neuron, aligned_unit * args.dose, "all"),
        ("aligned_decision_high_dose", neuron, aligned_unit * args.high_dose, "decision"),
        ("random_coordinate_decision", random_neuron, aligned_unit * args.dose, "decision"),
    ]
    results = {
        "experiment": "exp67_single_coordinate_decision_steering",
        "research_source": "research/HYPOTHESES.md H16",
        "paper_inspiration": ["arXiv:2611.29578"],
        "config": vars(args),
        "baseline": baseline,
        "selected": {
            "layer": best_layer,
            **sel,
            "median_residual_norm": round(residual_norms[best_layer], 5),
            "down_column_norm": round(col_norm, 5),
            "activation_shift_per_unit_dose": round(unit_shift, 5),
            "random_control_neuron": int(random_neuron),
        },
        "layer_rankings": ranking,
        "arms": {},
    }
    for name, nrn, shift, mode in arms:
        print(f"[Run] {name}: layer={best_layer} neuron={nrn} shift={shift:.3f} mode={mode}", flush=True)
        handle = steering_hook(model, best_layer, nrn, shift, mode)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        handle.remove()
        print(f"  [{name}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results["arms"][name] = {
            "layer": best_layer,
            "neuron": int(nrn),
            "activation_shift": round(float(shift), 5),
            "mode": mode,
            "eval": ev,
            "delta_exact_vs_baseline": round(ev["exact_compliance_rate"] - baseline["exact_compliance_rate"], 2),
            "delta_unsafe_refusal_vs_baseline": round(ev["unsafe_refusal_rate"] - baseline["unsafe_refusal_rate"], 2),
            "delta_benign_false_refusal_vs_baseline": round(ev["benign_false_refusal_rate"] - baseline["benign_false_refusal_rate"], 2),
        }

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "single_coordinate_steering_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 67 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
