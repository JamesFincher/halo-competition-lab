#!/usr/bin/env python3
"""Experiment 124: EMS-Guided Anchor-Preserving Subspace Edits.

Implements HYPOTHESES H4 in a compute-conscious form. Empirical Margin
Sensitivity crushes individual down-projection columns to sign(w)*mean(abs(w))
and measures correct-vs-hardest-wrong decision-margin changes on train rows.
EMS-negative suppressors, positive-anchor-excluding selections, random columns,
and group-crush diagnostics are then compared with the validated rank-1 edit.

Research anchors: research/HYPOTHESES.md H4 and arXiv:2611.30675.
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
from validated_edit import build_validated_rank1_subspaces, top_projection_columns, apply_validated_rank1_edit  # noqa: E402

EMS_LAYERS = [27, 28, 29]
FIELDS = {"action": '"action": "', "tool": '"tool": "', "policy_code": '"policy_code": "'}


def value_positions(tokenizer, prompt, completion):
    plen = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    out = {}
    for field, suffix in FIELDS.items():
        idx = completion.find(suffix)
        if idx < 0: continue
        pre = len(tokenizer(completion[: idx + len(suffix)], add_special_tokens=False)["input_ids"])
        out[field] = plen + pre - 1
    return out


def label_tokens(tokenizer, label):
    return [int(x) for x in tokenizer(str(label), add_special_tokens=False)["input_ids"]]


def discriminative_margin(logits, pos, correct_ids, wrong_ids):
    diverge = 0
    while diverge < min(len(correct_ids), len(wrong_ids)) and correct_ids[diverge] == wrong_ids[diverge]:
        diverge += 1
    ci = correct_ids[min(diverge, len(correct_ids) - 1)]
    wi = wrong_ids[min(diverge, len(wrong_ids) - 1)]
    use_pos = min(pos + diverge, logits.shape[0] - 1)
    return float((logits[use_pos, ci] - logits[use_pos, wi]).item())


@torch.no_grad()
def decision_margins(model, tokenizer, rows, device):
    action_values = ["call_tool", "refuse"]
    policy_values = ["SECRET_EXFIL", "PROMPT_INJECTION", "DESTRUCTIVE_OP"]
    tool_values = sorted({str(r["target"].get("tool")) for r in rows if r["target"].get("tool") is not None})
    token_ids = {x: label_tokens(tokenizer, x) for x in action_values + policy_values + tool_values}
    per_row = []
    for row in rows:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        completion = row["assistant_text"]
        positions = value_positions(tokenizer, prompt, completion)
        ids = tokenizer(prompt + completion, return_tensors="pt", truncation=True, max_length=384).to(device)["input_ids"]
        logits = model(ids).logits[0].float()
        margins = []
        detail = {}
        for field, pos in positions.items():
            if pos >= logits.shape[0]: continue
            label = row["target"].get(field)
            if label is None: continue
            values = action_values if field == "action" else policy_values if field == "policy_code" else tool_values
            correct = token_ids.get(str(label))
            wrong = [token_ids[x] for x in values if x != str(label) and token_ids.get(x)]
            if correct is None or not wrong: continue
            margin = min(discriminative_margin(logits, pos, correct, wids) for wids in wrong)
            margins.append(margin); detail[field] = margin
        per_row.append({"mean_margin": sum(margins) / max(len(margins), 1), "fields": detail})
    return per_row


def crush_column(model, layer, column):
    down = model.model.layers[layer].mlp.down_proj
    original = down.weight.data[:, column].clone()
    with torch.no_grad():
        mean_abs = original.abs().mean()
        down.weight.data[:, column] = torch.sign(original) * mean_abs
    return original


def restore_column(model, layer, column, original):
    with torch.no_grad(): model.model.layers[layer].mlp.down_proj.weight.data[:, column].copy_(original)


def pearson(xs, ys):
    x = torch.tensor(xs, dtype=torch.float32); y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y / (x.norm() * y.norm() + 1e-8)).item())


def measure_ems(model, tokenizer, rows, device, candidates):
    baseline = decision_margins(model, tokenizer, rows, device)
    records = []
    total = sum(len(v) for v in candidates.values()); done = 0
    for li in EMS_LAYERS:
        for col in candidates[li]:
            original = crush_column(model, li, col)
            crushed = decision_margins(model, tokenizer, rows, device)
            restore_column(model, li, col, original)
            deltas = [b["mean_margin"] - c["mean_margin"] for b, c in zip(baseline, crushed)]
            half = max(1, len(deltas) // 2)
            rec = {"layer": li, "column": int(col), "ems": sum(deltas) / len(deltas), "ems_fold1": sum(deltas[:half]) / half, "ems_fold2": sum(deltas[half:]) / max(len(deltas) - half, 1), "baseline_mean_margin": sum(x["mean_margin"] for x in baseline) / len(baseline), "crushed_mean_margin": sum(x["mean_margin"] for x in crushed) / len(crushed)}
            records.append(rec); done += 1
            if done % 10 == 0 or done == total: print(f"  [EMS {done}/{total}] L{li}c{col} ems={rec['ems']:+.4f}", flush=True)
    return records, baseline


def apply_custom_subspace_edit(model, subspaces, selected, strength):
    with torch.no_grad():
        for li, cols in selected.items():
            if not cols: continue
            down = model.model.layers[li].mlp.down_proj
            w = down.weight.data.float()
            U = subspaces[li].to(device=w.device, dtype=torch.float32)
            idx = torch.tensor(cols, device=w.device, dtype=torch.long)
            proj = torch.matmul(U.T, w)
            w[:, idx] = w[:, idx] - float(strength) * torch.matmul(U, proj[:, idx])
            down.weight.data.copy_(w.to(down.weight.dtype))


def choose_per_layer(records, mode, per_layer):
    out = {}
    for li in EMS_LAYERS:
        rows = [r for r in records if r["layer"] == li]
        if mode == "negative": rows = sorted(rows, key=lambda r: r["ems"])
        elif mode == "anchor_preserving": rows = sorted(rows, key=lambda r: (max(r["ems"], 0.0), r["ems"]))
        else: rows = sorted(rows, key=lambda r: r["random_key"])
        out[li] = [int(r["column"]) for r in rows[:per_layer]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=12)
    ap.add_argument("--ems-rows", type=int, default=8)
    ap.add_argument("--candidate-top", type=int, default=16)
    ap.add_argument("--candidate-random", type=int, default=6)
    ap.add_argument("--edit-per-layer", type=int, default=22)
    ap.add_argument("--strength", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp124_ems_anchor_preserving_subspace_edit/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 124] EMS-Guided Anchor-Preserving Subspace Edits on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"][: args.ems_rows // 2]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"][: args.ems_rows // 2]
    ems_rows = [x for pair in zip(benign, unsafe) for x in pair]
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, EMS_LAYERS, n_pairs=args.n_pairs)
    top = top_projection_columns(sft, subspaces, EMS_LAYERS, args.candidate_top)
    rng = random.Random(args.seed); candidates = {}
    width = sft.model.layers[29].mlp.down_proj.weight.shape[1]
    for li in EMS_LAYERS:
        pool = [x for x in range(width) if x not in set(top[li])]
        candidates[li] = list(top[li]) + rng.sample(pool, args.candidate_random)
    print(f"  EMS candidates: {sum(len(v) for v in candidates.values())} columns across layers {EMS_LAYERS}", flush=True)
    print("[Stage 1] Full sign/L1-preserving column-crush EMS probe...", flush=True)
    probe_model = copy.deepcopy(sft)
    records, baseline = measure_ems(probe_model, tokenizer, ems_rows, device, candidates)
    fold_corr = pearson([r["ems_fold1"] for r in records], [r["ems_fold2"] for r in records])
    sign_agree = sum(int((r["ems_fold1"] >= 0) == (r["ems_fold2"] >= 0)) for r in records) / len(records)
    print(f"  EMS stability: fold corr={fold_corr:.3f} sign agreement={sign_agree:.3f}", flush=True)
    for r in records: r["random_key"] = rng.random()
    selections = {"ems_negative_only": choose_per_layer(records, "negative", args.edit_per_layer), "positive_anchor_excluded": choose_per_layer(records, "anchor_preserving", args.edit_per_layer), "random_control": choose_per_layer(records, "random", args.edit_per_layer)}
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% unsafe={sft_eval['unsafe_refusal_rate']:.1f}% benign_fr={sft_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp124_ems_anchor_preserving_subspace_edit", "research_source": "research/HYPOTHESES.md H4", "paper_inspiration": ["arXiv:2611.30675"], "config": vars(args), "device": str(device), "baseline_sft": sft_eval, "ems_rows": len(ems_rows), "baseline_decision_margins": baseline, "ems_records": records, "ems_stability": {"fold_pearson": round(fold_corr, 5), "sign_agreement": round(sign_agree, 5)}, "top_singular_values": singulars, "selections": selections, "arms": {}}
    print("[Stage 2] Matched-budget surgical edit comparison...", flush=True)
    validated = copy.deepcopy(sft)
    val_info = apply_validated_rank1_edit(validated, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=args.strength)
    val_eval = base_rl.evaluate_heldout(validated, tokenizer, eval_rows, device)
    results["arms"]["validated_l29_l28_top64"] = {"eval": val_eval, "edit_info": val_info}
    print(f"  [validated L29/L28] exact={val_eval['exact_compliance_rate']:.1f}% unsafe={val_eval['unsafe_refusal_rate']:.1f}% benign_fr={val_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    for arm, selected in selections.items():
        model = copy.deepcopy(sft); apply_custom_subspace_edit(model, subspaces, selected, args.strength)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        results["arms"][arm] = {"eval": ev, "selected_neurons": selected}
        print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        del model
    print("[Stage 3] Group-crush diagnostic on strongest EMS-negative suppressors...", flush=True)
    crush_sel = {}
    for li in EMS_LAYERS: crush_sel[li] = [int(r["column"]) for r in sorted([x for x in records if x["layer"] == li], key=lambda x: x["ems"])[:8]]
    crush_model = copy.deepcopy(sft)
    with torch.no_grad():
        for li, cols in crush_sel.items():
            for col in cols:
                w = crush_model.model.layers[li].mlp.down_proj.weight.data[:, col]
                w.copy_(torch.sign(w) * w.abs().mean())
    crush_eval = base_rl.evaluate_heldout(crush_model, tokenizer, eval_rows, device)
    results["arms"]["group_crush_negative_diagnostic"] = {"eval": crush_eval, "crushed_columns": crush_sel}
    print(f"  [group crush] exact={crush_eval['exact_compliance_rate']:.1f}% unsafe={crush_eval['unsafe_refusal_rate']:.1f}% benign_fr={crush_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "ems_anchor_subspace_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 124 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
