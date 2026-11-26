#!/usr/bin/env python3
"""Experiment 85: Evolution Strategies in an Interpretable 32-D Surgical Subspace.

Upgrades SUBAGENT3 rank-5 beyond the earlier 8-D hyperparameter search: optimize
32 per-neuron edit-strength coefficients in the validated fp32 rank-1 residual
subspace. Uses train-dev fitness only, compares uniform doses and random search,
and evaluates untouched policy_eval as the final endpoint.

Research anchors: arXiv:1703.03864 and arXiv:2012.13255.
"""
from __future__ import annotations

import argparse, copy, json, random, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import build_validated_rank1_subspaces, top_projection_columns  # noqa: E402


def make_coordinate_map(selected):
    coords = []
    for li in sorted(selected):
        for neuron in selected[li][:16]:
            coords.append({"layer": li, "neuron": int(neuron)})
    if len(coords) != 32:
        raise ValueError(f"expected 32 coordinates, got {len(coords)}")
    return coords


def materialize_coefficients(model, base_weights, subspaces, coords, coeffs):
    with torch.no_grad():
        ws = {li: w0.clone().float() for li, w0 in base_weights.items()}
        for coef, coord in zip(coeffs, coords):
            li, neuron = coord["layer"], coord["neuron"]
            U = subspaces[li].to(device=ws[li].device, dtype=torch.float32)
            col = ws[li][:, neuron].unsqueeze(1)
            proj = torch.matmul(U.T, col)
            ws[li][:, neuron] = (col - float(coef) * torch.matmul(U, proj)).squeeze(1)
        for li, w in ws.items():
            down = model.model.layers[li].mlp.down_proj
            down.weight.data.copy_(w.to(dtype=down.weight.dtype))


def fitness(model, tokenizer, dev_rows, device):
    ev = base_rl.evaluate_heldout(model, tokenizer, dev_rows, device, max_new_tokens=64)
    score = ev["exact_compliance_rate"] + 0.25 * ev["unsafe_refusal_rate"] - 0.50 * ev["benign_false_refusal_rate"] + 5.0 * ev["mean_reward"]
    return float(score), ev


def centered_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = rank / max(len(values) - 1, 1) - 0.5
    return ranks


def run_es(model, tokenizer, dev_rows, device, base_weights, subspaces, coords, generations, population, sigma, lr, seed):
    rng = random.Random(seed)
    center = [0.9] * 32
    history = []
    best = None
    eval_count = 0
    for gen in range(generations):
        noise = [[rng.gauss(0.0, 1.0) for _ in range(32)] for _ in range(population - 1)]
        candidates = [center] + [[min(1.5, max(0.0, c + sigma * z)) for c, z in zip(center, eps)] for eps in noise]
        scored = []
        for cand in candidates:
            materialize_coefficients(model, base_weights, subspaces, coords, cand)
            fit, ev = fitness(model, tokenizer, dev_rows, device)
            eval_count += 1
            scored.append({"fit": fit, "coeffs": cand, "eval": ev})
        fits = [x["fit"] for x in scored]
        ranks = centered_ranks(fits[1:])
        grad = [sum(r * eps[j] for r, eps in zip(ranks, noise)) / max(len(noise), 1) / sigma for j in range(32)]
        center = [min(1.5, max(0.0, c + lr * g)) for c, g in zip(center, grad)]
        best_gen = max(scored, key=lambda x: x["fit"])
        if best is None or best_gen["fit"] > best["fit"]:
            best = best_gen
        history.append({"generation": gen, "best_fit": round(best_gen["fit"], 4), "center_mean": round(sum(center) / 32, 5), "center_l2": round(sum(c * c for c in center) ** 0.5, 5), "best_eval": best_gen["eval"]})
        print(f"  [ES gen{gen}] best_fit={best_gen['fit']:.2f} dev_exact={best_gen['eval']['exact_compliance_rate']:.1f}% center_mean={sum(center)/32:.3f}", flush=True)
    return best, history, center, eval_count


def random_search(model, tokenizer, dev_rows, device, base_weights, subspaces, coords, n_trials, seed):
    rng = random.Random(seed + 999)
    best = None; trials = []
    for i in range(n_trials):
        cand = [rng.uniform(0.3, 1.3) for _ in range(32)]
        materialize_coefficients(model, base_weights, subspaces, coords, cand)
        fit, ev = fitness(model, tokenizer, dev_rows, device)
        row = {"trial": i, "fit": fit, "coeffs": cand, "eval": ev}
        trials.append({"trial": i, "fit": round(fit, 4), "dev_exact": ev["exact_compliance_rate"]})
        if best is None or fit > best["fit"]: best = row
    print(f"  [random control] best_fit={best['fit']:.2f} dev_exact={best['eval']['exact_compliance_rate']:.1f}%", flush=True)
    return best, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--es-lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp85_evolution_strategies_32d_surgical_subspace/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 85] Evolution Strategies in 32-D Surgical Subspace on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = benign[20:28] + unsafe[20:28]
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    layers = list(vec_rl.EDIT_LAYERS)
    base_weights = {li: sft.model.layers[li].mlp.down_proj.weight.detach().clone() for li in layers}
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, layers, n_pairs=12)
    selected64 = top_projection_columns(sft, subspaces, layers, 64)
    coords = make_coordinate_map(selected64)
    print(f"  32 coordinates: {[(c['layer'], c['neuron']) for c in coords[:6]]} ...", flush=True)
    model = copy.deepcopy(sft)
    print("[Stage 1] Uniform dose-response references on train-dev...", flush=True)
    uniform = {}
    for dose in [0.0, 0.5, 0.9, 1.2]:
        coeffs = [dose] * 32
        materialize_coefficients(model, base_weights, subspaces, coords, coeffs)
        fit, ev = fitness(model, tokenizer, dev_rows, device)
        uniform[f"uniform_{dose}"] = {"coeffs": coeffs, "dev_fit": fit, "dev_eval": ev}
        print(f"  [uniform {dose}] fit={fit:.2f} dev_exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
    print("[Stage 2] OpenAI-ES optimization...", flush=True)
    es_best, es_history, es_center, es_evals = run_es(model, tokenizer, dev_rows, device, base_weights, subspaces, coords, args.generations, args.population, args.sigma, args.es_lr, args.seed)
    print("[Stage 3] Random-search control with matched trial count...", flush=True)
    rand_best, rand_trials = random_search(model, tokenizer, dev_rows, device, base_weights, subspaces, coords, es_evals, args.seed)
    print("[Stage 4] Untouched policy_eval endpoint evaluation...", flush=True)
    final_evals = {}
    for name, coeffs in [("uniform_0.9", [0.9] * 32), ("uniform_0.0", [0.0] * 32), ("es_best", es_best["coeffs"]), ("random_best", rand_best["coeffs"]), ("es_final_center", es_center)]:
        materialize_coefficients(model, base_weights, subspaces, coords, coeffs)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        final_evals[name] = {"eval": ev, "coefficients": [round(float(x), 6) for x in coeffs]}
        print(f"  [{name:16s}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% json={ev['json_valid_rate']:.1f}%", flush=True)
    results = {"experiment": "exp85_evolution_strategies_32d_surgical_subspace", "research_source": "research/SUBAGENT3_CREATIVE_HEURISTICS.md rank 5", "paper_inspiration": ["arXiv:1703.03864", "arXiv:2012.13255"], "config": vars(args), "device": str(device), "coordinate_map": coords, "top_singular_values": singulars, "dev_protocol": "8 benign + 8 unsafe train rows offset from direction-estimation prefix; policy_eval untouched until final", "uniform_dev_references": uniform, "es_history": es_history, "es_best_dev": {"fit": es_best["fit"], "eval": es_best["eval"], "coefficients": [round(float(x), 6) for x in es_best["coeffs"]]}, "random_trials": rand_trials, "random_best_dev": {"fit": rand_best["fit"], "eval": rand_best["eval"], "coefficients": [round(float(x), 6) for x in rand_best["coeffs"]]}, "final_policy_eval": final_evals, "wall_sec": round(time.time() - t0, 2)}
    path = out_dir / "es32_surgical_subspace_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 85 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
