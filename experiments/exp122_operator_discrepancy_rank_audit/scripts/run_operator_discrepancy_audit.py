#!/usr/bin/env python3
"""Experiment 122: Operator-Discrepancy Rank Audit for H2 Repairability.

Pre-flight structural audit for HYPOTHESES H2. ReQOR-style collateral repair only
has a nontrivial orthogonal repair space when the same-state operator discrepancy
is multi-directional. This experiment measures the exact discrepancy spectra of
the Exp-1-style contrastive rank-1 edit and the validated residual rank-1 edit,
then evaluates both behavioral endpoints.

Research anchors: research/HYPOTHESES.md H2 and arXiv:2611.32755.
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
from serialization_utils import dumps_safe, write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402


@torch.no_grad()
def cache_down_inputs(model, tokenizer, rows, device, layers, max_rows):
    stored = {li: [] for li in layers}
    hooks = []
    def make_hook(li):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            stored[li].append(x.detach().float().reshape(-1, x.shape[-1]).cpu())
        return hook
    for li in layers: hooks.append(model.model.layers[li].mlp.down_proj.register_forward_hook(make_hook(li)))
    try:
        for row in rows[:max_rows]:
            text = base_rl.build_prompt(tokenizer, row["messages"]) + row["assistant_text"]
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320).to(device)
            model(**enc)
    finally:
        for h in hooks: h.remove()
    return {li: torch.cat(xs, dim=0) for li, xs in stored.items() if xs}


def audit_spectrum(delta_w, inputs, label, layer):
    delta = delta_w.float().cpu()
    ud, sd, _vhd = torch.linalg.svd(delta, full_matrices=False)
    weight_rank_energy = float((sd[0] ** 2 / (sd.pow(2).sum() + 1e-12)).item())
    target_dir = ud[:, 0]
    x = inputs[:4096].float()
    if x.shape[0] == 0: raise RuntimeError("no cached down_proj inputs")
    c = (delta @ x.T).float()
    u, s, _vh = torch.linalg.svd(c, full_matrices=False)
    energy = s.pow(2); total = energy.sum() + 1e-12
    frac = energy / total
    align = [float(torch.dot(u[:, i] / (u[:, i].norm() + 1e-8), target_dir / (target_dir.norm() + 1e-8)).abs().item()) for i in range(min(8, u.shape[1]))]
    target_aligned = torch.tensor(align) > 0.8
    collateral_energy = float((frac[: len(align)] * (~target_aligned).float()).sum().item())
    nonzero = int((frac > 1e-4).sum().item())
    collateral_rank = 0
    for i in range(min(len(align), len(frac))):
        if frac[i] > 1e-4 and align[i] < 0.8: collateral_rank += 1
    return {"operator": label, "layer": int(layer), "n_states": int(x.shape[0]), "weight_delta_rank1_energy": round(weight_rank_energy, 8), "discrepancy_top1_energy": round(float(frac[0].item()), 8), "discrepancy_top2_energy": round(float(frac[1].item()) if len(frac) > 1 else 0.0, 8), "discrepancy_top4_energy": [round(float(z), 8) for z in frac[:4]], "effective_rank": round(float((total ** 2 / (energy.pow(2).sum() + 1e-12)).item()), 5), "singular_direction_alignments_to_weight_delta": [round(z, 6) for z in align], "n_directions_energy_gt_1e-4": nonzero, "non_target_collateral_rank": collateral_rank, "non_target_collateral_energy_top8": round(collateral_energy, 8), "weight_delta_top_singular": round(float(sd[0].item()), 8), "weight_delta_second_over_first": round(float((sd[1] / (sd[0] + 1e-12)).item()) if len(sd) > 1 else 0.0, 8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp122_operator_discrepancy_rank_audit/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 122] Operator-Discrepancy Rank Audit on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% unsafe={sft_eval['unsafe_refusal_rate']:.1f}% benign_fr={sft_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp122_operator_discrepancy_rank_audit", "research_source": "research/HYPOTHESES.md H2 structural pre-flight", "paper_inspiration": ["arXiv:2611.32755"], "config": vars(args), "device": str(device), "baseline_sft": sft_eval, "audits": {}, "structural_conclusion": None}
    print("[Stage] Reconstructing Exp1-style contrastive edit...", flush=True)
    top_layers, _rand_layers, failure_dirs, ranked = base_rl.localize_and_compute_failure_dirs(sft, tokenizer, train_rows, device, n_pairs=args.n_pairs, top_k_layers=8, seed=args.seed)
    exp1_layers = [r["layer"] for r in sorted(ranked, key=lambda x: x["contrastive_l2"], reverse=True)[:2]]
    exp1_model = copy.deepcopy(sft)
    base_w1 = {li: exp1_model.model.layers[li].mlp.down_proj.weight.detach().clone() for li in exp1_layers}
    exp1_info = base_rl.apply_surgical_pre_edit(exp1_model, failure_dirs, exp1_layers, top_columns=32, strength=0.5)
    exp1_eval = base_rl.evaluate_heldout(exp1_model, tokenizer, eval_rows, device)
    print(f"  [exp1-style edit] layers={exp1_layers} exact={exp1_eval['exact_compliance_rate']:.1f}% unsafe={exp1_eval['unsafe_refusal_rate']:.1f}% benign_fr={exp1_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    inputs1 = cache_down_inputs(exp1_model, tokenizer, train_rows, device, exp1_layers, 16)
    audits1 = []
    for li in exp1_layers:
        edited_w = exp1_model.model.layers[li].mlp.down_proj.weight.detach().cpu()
        audits1.append(audit_spectrum(base_w1[li].cpu() - edited_w, inputs1[li], "exp1_contrastive_top32_strength0.5", li))
        print(f"  [audit exp1 L{li}] top1={audits1[-1]['discrepancy_top1_energy']:.6f} effective_rank={audits1[-1]['effective_rank']:.2f} collateral_rank={audits1[-1]['non_target_collateral_rank']}", flush=True)
    print("[Stage] Reconstructing validated residual rank-1 edit...", flush=True)
    val_model = copy.deepcopy(sft)
    base_wv = {li: val_model.model.layers[li].mlp.down_proj.weight.detach().clone() for li in vec_rl.EDIT_LAYERS}
    val_info = apply_validated_rank1_edit(val_model, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    val_eval = base_rl.evaluate_heldout(val_model, tokenizer, eval_rows, device)
    print(f"  [validated edit] exact={val_eval['exact_compliance_rate']:.1f}% unsafe={val_eval['unsafe_refusal_rate']:.1f}% benign_fr={val_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    inputsv = cache_down_inputs(val_model, tokenizer, train_rows, device, vec_rl.EDIT_LAYERS, 16)
    auditv = []
    for li in vec_rl.EDIT_LAYERS:
        edited_w = val_model.model.layers[li].mlp.down_proj.weight.detach().cpu()
        auditv.append(audit_spectrum(base_wv[li].cpu() - edited_w, inputsv[li], "validated_rank1_top64_strength0.9", li))
        print(f"  [audit validated L{li}] top1={auditv[-1]['discrepancy_top1_energy']:.6f} effective_rank={auditv[-1]['effective_rank']:.2f} collateral_rank={auditv[-1]['non_target_collateral_rank']}", flush=True)
    all_audits = audits1 + auditv
    max_collateral_rank = max(x["non_target_collateral_rank"] for x in all_audits)
    max_collateral_energy = max(x["non_target_collateral_energy_top8"] for x in all_audits)
    conclusion = "STRUCTURALLY_DEGENERATE_FOR_RANK1_EDITS: same-state discrepancy spectra are target-direction-dominated; H2 needs a multi-direction operator or output-trajectory collateral measurement before adaptive repair can be meaningful." if max_collateral_rank == 0 or max_collateral_energy < 0.01 else "MULTI_DIRECTIONAL_COLLATERAL_PRESENT: H2 repair audit is meaningful."
    print(f"  [conclusion] {conclusion}", flush=True)
    results.update({"exp1_operator": {"layers": exp1_layers, "edit_info": exp1_info, "eval": exp1_eval, "audit": audits1}, "validated_operator": {"edit_info": val_info, "eval": val_eval, "audit": auditv}, "max_non_target_collateral_rank": max_collateral_rank, "max_non_target_collateral_energy_top8": max_collateral_energy, "structural_conclusion": conclusion, "wall_sec": round(time.time() - t0, 2)})
    path = out_dir / "operator_discrepancy_rank_audit.json"; write_json(path, results)
    print(f"[Exp 122 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
