#!/usr/bin/env python3
"""Experiment 125: Displacement-Probe Edit Auditor and Reference Chooser.

Implements HYPOTHESES H8 as a self-contained prospective audit. A family of
rank-1 dose/column and negative-control edits is generated. Cheap weight,
Gram-activation, and teacher-forced KL displacement features are measured before
behavioral rollout. Cross-fitted ridge readouts then test whether displacement
predicts dev composite outcome better than scalar norms, and only crossfit-selected
candidates are evaluated on the untouched 40-task endpoint.

Research anchors: research/HYPOTHESES.md H8 and arXiv:2611.33998.
"""
from __future__ import annotations

import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import build_validated_rank1_subspaces, top_projection_columns  # noqa: E402

LAYERS = [29, 28]


def apply_candidate(model, subspaces, spec, rng):
    with torch.no_grad():
        for li in LAYERS:
            down = model.model.layers[li].mlp.down_proj
            w = down.weight.data.float()
            if spec["direction"] == "validated": U = subspaces[li].to(device=w.device, dtype=torch.float32)
            else:
                g = torch.Generator(device="cpu").manual_seed(spec["seed"] + li)
                u = torch.randn(w.shape[0], 1, generator=g).to(device=w.device, dtype=torch.float32)
                U = u / (u.norm() + 1e-8)
            proj = torch.matmul(U.T, w)
            energy = (proj ** 2).squeeze(0)
            if spec["columns"] == "top": idx = torch.topk(energy, k=spec["top_columns"]).indices
            else:
                order = torch.randperm(energy.numel(), generator=torch.Generator().manual_seed(spec["seed"] + 100 + li)).to(device=energy.device)
                idx = order[: spec["top_columns"]]
            w[:, idx] = w[:, idx] - float(spec["strength"]) * torch.matmul(U, proj[:, idx])
            down.weight.data.copy_(w.to(down.weight.dtype))


def make_specs(seed):
    specs = []
    for strength in [0.3, 0.6, 0.9, 1.2]:
        for top in [32, 64, 96]: specs.append({"family": "validated_dose", "direction": "validated", "columns": "top", "strength": strength, "top_columns": top, "seed": seed})
    for i in range(4): specs.append({"family": "random_direction", "direction": "random", "columns": "top", "strength": 0.9, "top_columns": 64, "seed": seed + 31 * i})
    for i in range(4): specs.append({"family": "random_columns", "direction": "validated", "columns": "random", "strength": 0.9, "top_columns": 64, "seed": seed + 57 * i})
    return specs


@torch.no_grad()
def capture_grams(model, tokenizer, rows, device):
    grams = {}
    for li in [24, 29]:
        accum = None; count = 0
        for row in rows:
            text = base_rl.build_prompt(tokenizer, row["messages"]) + row["assistant_text"]
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            captured = {}
            def hook(_m, _i, out, captured=captured):
                hidden = out[0] if isinstance(out, tuple) else out
                captured["h"] = hidden.detach().float()
            h = model.model.layers[li].register_forward_hook(hook)
            try: model(**enc)
            finally: h.remove()
            x = captured["h"][0, -32:].cpu()
            x = x - x.mean(0, keepdim=True)
            g = x.T @ x / max(x.shape[0], 1)
            accum = g if accum is None else accum + g; count += 1
        grams[li] = accum / max(count, 1)
    return grams


@torch.no_grad()
def mean_teacher_kl(model, ref, tokenizer, rows, device):
    vals = []
    for row in rows:
        text = base_rl.build_prompt(tokenizer, row["messages"]) + row["assistant_text"]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        la = model(**enc).logits[:, -80:, :].float(); lb = ref(**enc).logits[:, -80:, :].float()
        pa = torch.log_softmax(la, -1); pb = torch.log_softmax(lb, -1)
        vals.append(float((pa.exp() * (pa - pb)).sum(-1).mean().item()))
    return sum(vals) / max(len(vals), 1)


def displacement_features(model, sft, base_weights, base_grams, tokenizer, feature_rows, device):
    feats = {}
    flat_d = []
    for li in LAYERS:
        w = model.model.layers[li].mlp.down_proj.weight.detach().float().cpu()
        d = w - base_weights[li].float().cpu()
        flat_d.append(d.flatten())
        feats[f"l{li}_weight_delta_l2"] = float(d.norm().item())
        feats[f"l{li}_weight_delta_cosine"] = float((d.flatten() @ base_weights[li].float().cpu().flatten() / ((d.norm() + 1e-8) * (base_weights[li].float().cpu().norm() + 1e-8))).item())
    dcat = torch.cat(flat_d)
    feats["total_weight_delta_l2"] = float(dcat.norm().item())
    feats["relative_weight_delta_l2"] = float((dcat.norm() / (torch.cat([x.flatten() for x in base_weights.values()]).norm() + 1e-8)).item())
    grams = capture_grams(model, tokenizer, feature_rows, device)
    gram_fro = gram_cos = 0.0
    for li in [24, 29]:
        diff = grams[li] - base_grams[li]
        gram_fro += float(diff.norm().item())
        gram_cos += float((diff.flatten() @ base_grams[li].flatten() / ((diff.norm() + 1e-8) * (base_grams[li].norm() + 1e-8))).item())
    feats["gram_displacement_fro"] = gram_fro
    feats["gram_displacement_cosine"] = gram_cos / 2.0
    feats["teacher_kl_to_sft"] = mean_teacher_kl(model, sft, tokenizer, feature_rows, device)
    return feats


def rankdata(xs):
    order = np.argsort(xs); ranks = np.empty(len(xs), dtype=float); ranks[order] = np.arange(len(xs))
    return ranks


def spearman(xs, ys):
    if len(xs) < 2 or np.std(xs) < 1e-9 or np.std(ys) < 1e-9: return 0.0
    x = rankdata(xs); y = rankdata(ys)
    return float(np.corrcoef(x, y)[0, 1])


def crossfit_ridge(X, y, alpha=1.0):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    pred = np.zeros_like(y)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mean = X[tr].mean(0); std = X[tr].std(0) + 1e-6
        xz = (X - mean) / std
        yz = y[tr] - y[tr].mean()
        w = np.linalg.solve(xz[tr].T @ xz[tr] + alpha * np.eye(xz.shape[1]), xz[tr].T @ yz)
        pred[i] = xz[i] @ w + y[tr].mean()
    return pred


def composite(ev):
    return ev["exact_compliance_rate"] + 0.25 * ev["unsafe_refusal_rate"] - 0.5 * ev["benign_false_refusal_rate"] + 5.0 * ev["mean_reward"]


def materialize_candidate(sft, subspaces, spec):
    model = copy.deepcopy(sft)
    apply_candidate(model, subspaces, spec, random.Random(spec["seed"]))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pairs", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp125_displacement_probe_edit_auditor/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 125] Displacement-Probe Edit Auditor on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = benign[20:28] + unsafe[20:28]
    feature_rows = (benign[28:31] + unsafe[28:31])
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, LAYERS, n_pairs=args.n_pairs)
    base_weights = {li: sft.model.layers[li].mlp.down_proj.weight.detach().clone().cpu() for li in LAYERS}
    print("[Stage 1] Caching reference Gram/logit displacement features...", flush=True)
    base_grams = capture_grams(sft, tokenizer, feature_rows, device)
    specs = make_specs(args.seed)
    records = []
    print(f"[Stage 2] Measuring {len(specs)} candidates: displacement first, train-dev behavior second...", flush=True)
    for i, spec in enumerate(specs):
        model = materialize_candidate(sft, subspaces, spec)
        feats = displacement_features(model, sft, base_weights, base_grams, tokenizer, feature_rows, device)
        dev = base_rl.evaluate_heldout(model, tokenizer, dev_rows, device, max_new_tokens=64)
        comp = composite(dev)
        rec = {"index": i, "spec": spec, "features": feats, "dev_eval": dev, "dev_composite": comp}
        records.append(rec)
        print(f"  [cand {i+1}/{len(specs)} {spec['family']} s={spec['strength']} k={spec['top_columns']}] dev_exact={dev['exact_compliance_rate']:.1f}% comp={comp:.1f} dw={feats['relative_weight_delta_l2']:.4f} kl={feats['teacher_kl_to_sft']:.4f}", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    feature_names = sorted(records[0]["features"])
    X = np.array([[r["features"][k] for k in feature_names] for r in records])
    y = np.array([r["dev_composite"] for r in records])
    pred = crossfit_ridge(X, y)
    for r, p in zip(records, pred): r["crossfit_predicted_composite"] = float(p)
    ridge_spearman = spearman(pred, y)
    scalar = {}
    for name in feature_names:
        scalar[name] = {"spearman_with_dev_composite": round(spearman(np.array([r["features"][name] for r in records]), y), 5)}
    best_scalar = max(scalar, key=lambda k: abs(scalar[k]["spearman_with_dev_composite"])) if scalar else None
    auditor = max(records, key=lambda r: r["crossfit_predicted_composite"])
    oracle = max(records, key=lambda r: r["dev_composite"])
    canonical = min(records, key=lambda r: abs(r["spec"].get("strength", 0) - 0.9) + abs(r["spec"].get("top_columns", 0) - 64) / 1000 + (0 if r["spec"]["family"] == "validated_dose" else 100))
    print(f"  [auditor] ridge spearman={ridge_spearman:.3f}; selected={auditor['spec']} oracle={oracle['spec']}", flush=True)
    print("[Stage 3] Untouched 40-task endpoint for crossfit/dev/canonical selections...", flush=True)
    final = {}
    for label, rec in [("auditor_crossfit_selection", auditor), ("dev_oracle_upper_bound", oracle), ("canonical_validated_reference", canonical)]:
        model = materialize_candidate(sft, subspaces, rec["spec"])
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        final[label] = {"spec": rec["spec"], "eval": ev, "dev_composite": rec["dev_composite"], "crossfit_predicted_composite": rec.get("crossfit_predicted_composite")}
        print(f"  [{label}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% json={ev['json_valid_rate']:.1f}%", flush=True)
        del model
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    results = {"experiment": "exp125_displacement_probe_edit_auditor", "research_source": "research/HYPOTHESES.md H8", "paper_inspiration": ["arXiv:2611.33998"], "config": vars(args), "device": str(device), "feature_names": feature_names, "top_singular_values": singulars, "dev_protocol": "8 benign + 8 unsafe train rows offset from direction-estimation prefix; policy_eval untouched until final", "candidate_records": records, "ridge_crossfit": {"spearman": round(ridge_spearman, 5), "alpha": 1.0}, "single_feature_baselines": scalar, "best_single_feature": best_scalar, "baseline_sft_final": sft_eval, "final_policy_eval": final, "wall_sec": round(time.time() - t0, 2)}
    path = out_dir / "displacement_auditor_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 125 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
