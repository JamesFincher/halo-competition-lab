#!/usr/bin/env python3
"""Experiment 128: Output-Trajectory Collateral Repair Adapter.

Implements the non-degenerate branch of HYPOTHESES H2 after exp122's rank audit.
Because same-state rank-1 down-projection discrepancies are structurally target-
direction-dominated, this experiment measures trajectory-level residual-stream
collateral on control text and fits a closed-form low-rank ridge adapter from
layer-29 inputs to the original-minus-edited output displacement. Full,
target-orthogonal, random-subspace, and rank controls are compared.

Research anchors: research/HYPOTHESES.md H2, exp122 preflight, and
arXiv:2611.33972-style closed-form low-label adaptation.
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit, build_validated_rank1_subspaces  # noqa: E402

ADAPTER_LAYER = 29


def hidden(out):
    return out[0] if isinstance(out, tuple) else out


def replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@torch.no_grad()
def capture_pair(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320).to(device)
    cap = {}
    def hook(_m, inp, out):
        cap["in"] = inp[0].detach().float()
        cap["out"] = hidden(out).detach().float()
    handle = model.model.layers[ADAPTER_LAYER].register_forward_hook(hook)
    try: model(**enc)
    finally: handle.remove()
    return cap["in"][0], cap["out"][0]


def control_texts(tokenizer, train_rows, smoke=False):
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"]
    n = 2 if smoke else 5
    texts = [
        "Question: What is the capital of France?\nAnswer:",
        "Question: Explain photosynthesis in one sentence.\nAnswer:",
        "Write a Python function that returns the square of a number.",
        "Question: Which planet is closest to the Sun?\nAnswer:",
        "Summarize why unit tests are useful.",
        "Question: What is 17 plus 25?\nAnswer:",
    ]
    for row in benign[:n] + unsafe[:n]: texts.append(base_rl.build_prompt(tokenizer, row["messages"]))
    return texts


@torch.no_grad()
def collect_trajectory_data(orig, edit, tokenizer, texts, device):
    xs = []; ys = []
    for text in texts:
        xe, ye = capture_pair(edit, tokenizer, text, device)
        _xo, yo = capture_pair(orig, tokenizer, text, device)
        k = min(48, xe.shape[0], ye.shape[0])
        xs.append(xe[-k:].cpu()); ys.append((yo[-k:] - ye[-k:]).cpu())
    return torch.cat(xs, 0), torch.cat(ys, 0)


def fit_repair(X, Y, rank, ridge=1e-2, target_direction=None, orthogonalize=False):
    X = X.float(); Y = Y.float()
    xm = X.mean(0, keepdim=True); ym = Y.mean(0, keepdim=True)
    xc = X - xm; yc = Y - ym
    gram = xc.T @ xc + ridge * torch.eye(xc.shape[1])
    sol = torch.linalg.solve(gram, xc.T @ yc)
    if orthogonalize and target_direction is not None:
        d = target_direction.float().cpu(); d = d / (d.norm() + 1e-8)
        sol = sol - d[:, None] * (d[None, :] @ sol)
    u, s, vh = torch.linalg.svd(sol, full_matrices=False)
    r = min(rank, u.shape[1])
    low = u[:, :r] @ torch.diag(s[:r]) @ vh[:r]
    intercept = ym - xm @ low
    return {"matrix": low, "intercept": intercept, "singular_values": s[: min(8, len(s))].tolist(), "rank": r}


def optimal_alpha(X, Y, repair):
    pred = X @ repair["matrix"] + repair["intercept"]
    denom = float(pred.pow(2).sum()) + 1e-12
    return float((pred * Y).sum() / denom)


def make_random_repair(template, seed):
    g = torch.Generator().manual_seed(seed)
    rnd = torch.randn(template["matrix"].shape, generator=g)
    rnd = rnd * (template["matrix"].norm() / (rnd.norm() + 1e-8))
    return {"matrix": rnd, "intercept": template["intercept"].clone(), "singular_values": template["singular_values"], "rank": template["rank"]}


def install_repair(model, repair, alpha=1.0):
    w = repair["matrix"].to(device=model.device if hasattr(model, "device") else next(model.parameters()).device, dtype=torch.float32)
    b = repair["intercept"].squeeze(0).to(device=w.device, dtype=torch.float32)
    def hook(_m, inp, out):
        h = hidden(out)
        x = inp[0].float()
        delta = alpha * (x @ w + b).to(dtype=h.dtype)
        return replace(out, h + delta)
    return model.model.layers[ADAPTER_LAYER].register_forward_hook(hook)


@torch.no_grad()
def trajectory_kl_and_restoration(model, orig, tokenizer, texts, device):
    kls = []; mses = []
    for text in texts[:8]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320).to(device)
        lo = orig(**enc).logits.float(); lm = model(**enc).logits.float()
        pa = torch.log_softmax(lm, -1); pb = torch.log_softmax(lo, -1)
        kls.append(float((pa.exp() * (pa - pb)).sum(-1).mean().item()))
        _xi, yo = capture_pair(orig, tokenizer, text, device)
        _xe, ym = capture_pair(model, tokenizer, text, device)
        diff = (ym - yo).float(); mses.append(float((diff.norm() / (yo.norm() + 1e-8)).item()))
    return {"mean_teacher_kl_to_original": round(sum(kls) / len(kls), 7), "mean_relative_layer29_residual_error": round(sum(mses) / len(mses), 7)}


def write_json(path, obj):
    def convert(x):
        if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [convert(v) for v in x]
        return x
    path.write_text(json.dumps(convert(obj), indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--benchmark-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); smoke = os.environ.get("HALO_SMOKE_TEST") == "1"
    if smoke: args.benchmark_n = 0; args.rank = min(args.rank, 4)
    out_dir = ROOT / "experiments/exp128_trajectory_collateral_repair_adapter/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 128] Output-Trajectory Collateral Repair on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90, n_pairs=args.n_pairs)
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, vec_rl.EDIT_LAYERS, n_pairs=args.n_pairs)
    target_direction = subspaces[29].squeeze(1).float().cpu()
    texts = control_texts(tokenizer, train_rows, smoke)
    print(f"[Stage 1] Fitting closed-form ridge repair on {len(texts)} control/policy-prompt texts...", flush=True)
    X, Y = collect_trajectory_data(sft, edited, tokenizer, texts, device)
    print(f"  trajectory design matrix: X={tuple(X.shape)} Y={tuple(Y.shape)} target_rms={float(Y.pow(2).mean().sqrt()):.5f}", flush=True)
    full = fit_repair(X, Y, args.rank, args.ridge, orthogonalize=False)
    orth = fit_repair(X, Y, args.rank, args.ridge, target_direction=target_direction, orthogonalize=True)
    rand = make_random_repair(full, args.seed)
    full_alpha = optimal_alpha(X, Y, full); orth_alpha = optimal_alpha(X, Y, orth); rand_alpha = optimal_alpha(X, Y, rand)
    arms = [("edit_only_no_repair", None, 1.0), ("full_trajectory_repair_r16", full, full_alpha), ("target_orthogonal_repair_r16", orth, orth_alpha), ("random_rank_matched_repair", rand, rand_alpha), ("half_dose_target_orthogonal", orth, 0.5 * orth_alpha)]
    if smoke: arms = arms[:3]
    results = {"experiment": "exp128_trajectory_collateral_repair_adapter", "research_source": "research/HYPOTHESES.md H2 output-trajectory branch after exp122 rank audit", "paper_inspiration": ["arXiv:2611.33972"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "top_singular_values": singulars, "control_protocol": "generic control prompts plus policy prompt-only controls; no policy_eval fitting", "fit": {"x_shape": list(X.shape), "y_shape": list(Y.shape), "target_rms": float(Y.pow(2).mean().sqrt()), "full_singular_values": full["singular_values"], "orthogonal_singular_values": orth["singular_values"]}, "arms": {}}
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    results["baseline_original_sft"] = {"eval": sft_eval, **trajectory_kl_and_restoration(sft, sft, tokenizer, texts, device)}
    print(f"  [original SFT] exact={sft_eval['exact_compliance_rate']:.1f}% unsafe={sft_eval['unsafe_refusal_rate']:.1f}% benign_fr={sft_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    for name, repair, alpha in arms:
        model = copy.deepcopy(edited).to(device)
        handle = install_repair(model, repair, alpha) if repair is not None else None
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        traj = trajectory_kl_and_restoration(model, sft, tokenizer, texts, device)
        bench = {}
        if args.benchmark_n:
            try:
                import real_benchmark
                bench = real_benchmark.run_benchmark(model, tokenizer, device, n=args.benchmark_n)
            except Exception as e: bench = {"error": str(e)[:200]}
        if handle is not None: handle.remove()
        results["arms"][name] = {"eval": ev, **traj, "benchmark": bench, "alpha": alpha, "repair_rank": repair["rank"] if repair else 0}
        print(f"  [{name}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% kl={traj['mean_teacher_kl_to_original']:.5f} restore_err={traj['mean_relative_layer29_residual_error']:.3f}", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "trajectory_collateral_repair_summary.json"; write_json(path, results)
    print(f"[Exp 128 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
