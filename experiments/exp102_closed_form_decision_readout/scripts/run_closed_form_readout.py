#!/usr/bin/env python3
"""Experiment 102: Closed-Form Decision-Readout Correction.

Implements HYPOTHESES H12 as a no-gradient decision-boundary experiment. Hidden
states immediately before action/tool/policy values are extracted from frozen
teacher-forced train trajectories; shrinkage-LDA, linear-ridge, and kernel-ridge
readouts are selected by cross-fitted train folds. The selected readout then
forces canonical value tokens only while the generator is inside that decision
field. Majority and frozen-random-label controls are included.

Research anchors: research/HYPOTHESES.md H12 and arXiv:2611.33972.
"""
from __future__ import annotations

import argparse, copy, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402

READOUT_LAYERS = list(range(20, 30))
FIELDS = {"action": '"action": "', "tool": '"tool": "', "policy_code": '"policy_code": "'}


def capture_hooks(model, layers):
    captured = {}; handles = []
    def make_hook(li):
        def hook(_module, _args, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captured[li] = hidden.detach()
        return hook
    for li in layers: handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))
    return handles, captured


def close_handles(handles):
    for h in handles: h.remove()


def value_positions(tokenizer, prompt, completion):
    plen = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    out = {}
    for field, suffix in FIELDS.items():
        idx = completion.find(suffix)
        if idx < 0: continue
        value_start = idx + len(suffix)
        pre_count = len(tokenizer(completion[:value_start], add_special_tokens=False)["input_ids"])
        out[field] = plen + pre_count - 1
    return out


@torch.no_grad()
def cache_decision_features(model, tokenizer, rows, device):
    data = {field: {"X": {li: [] for li in READOUT_LAYERS}, "y": [], "rows": []} for field in FIELDS}
    for row_i, row in enumerate(rows):
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        completion = row["assistant_text"]
        positions = value_positions(tokenizer, prompt, completion)
        if not positions: continue
        ids = tokenizer(prompt + completion, return_tensors="pt", truncation=True, max_length=384).to(device)["input_ids"]
        handles, captured = capture_hooks(model, READOUT_LAYERS)
        try: model(ids)
        finally: close_handles(handles)
        for field, pos in positions.items():
            if pos >= ids.shape[1]: continue
            label = row["target"].get(field)
            if label is None: continue
            data[field]["y"].append(str(label)); data[field]["rows"].append(row_i)
            for li in READOUT_LAYERS: data[field]["X"][li].append(captured[li][0, pos].float().cpu().numpy())
    for field in data:
        data[field]["y"] = np.array(data[field]["y"])
        for li in READOUT_LAYERS:
            arr = np.stack(data[field]["X"][li]) if data[field]["X"][li] else np.zeros((0, 1), dtype=np.float32)
            data[field]["X"][li] = arr
    return data


def fit_probe(X, y, method):
    classes = sorted(set(y.tolist()))
    mean = X.mean(0, keepdims=True); std = X.std(0, keepdims=True) + 1e-5
    Xz = (X - mean) / std
    if method == "linear_ridge":
        Y = np.stack([(y == c).astype(np.float32) for c in classes], axis=1)
        Y = Y - Y.mean(0, keepdims=True)
        xtx = Xz.T @ Xz
        W = np.linalg.solve(xtx + 1.0 * np.eye(Xz.shape[1], dtype=np.float32), Xz.T @ Y)
        return {"method": method, "classes": classes, "mean": mean, "std": std, "W": W}
    if method == "shrinkage_lda":
        mus = []
        sw = np.zeros((Xz.shape[1], Xz.shape[1]), dtype=np.float32)
        for c in classes:
            xc = Xz[y == c]; mu = xc.mean(0); mus.append(mu)
            z = xc - mu; sw += z.T @ z
        sw = sw / max(len(y) - len(classes), 1)
        sw = 0.9 * sw + 0.1 * np.trace(sw) / sw.shape[0] * np.eye(sw.shape[0], dtype=np.float32) + 1e-3 * np.eye(sw.shape[0], dtype=np.float32)
        precision = np.linalg.inv(sw)
        M = np.stack(mus)
        return {"method": method, "classes": classes, "mean": mean, "std": std, "M": M, "precision": precision}
    if method == "kernel_ridge":
        Y = np.stack([(y == c).astype(np.float32) for c in classes], axis=1)
        d2 = ((Xz[:, None, :] - Xz[None, :, :]) ** 2).sum(-1)
        gamma = 1.0 / max(float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0, 1e-6)
        K = np.exp(-gamma * d2)
        alpha = np.linalg.solve(K + 0.1 * np.eye(K.shape[0], dtype=np.float32), Y)
        return {"method": method, "classes": classes, "mean": mean, "std": std, "train_x": Xz, "gamma": gamma, "alpha": alpha}
    raise ValueError(method)


def probe_scores(probe, X):
    Xz = (X - probe["mean"]) / probe["std"]
    if probe["method"] == "linear_ridge": return Xz @ probe["W"]
    if probe["method"] == "shrinkage_lda":
        mp = probe["M"] @ probe["precision"]
        return Xz @ mp.T - 0.5 * (probe["M"] * mp).sum(1)
    d2 = ((Xz[:, None, :] - probe["train_x"][None, :, :]) ** 2).sum(-1)
    return np.exp(-probe["gamma"] * d2) @ probe["alpha"]


def probe_predict(probe, X):
    scores = probe_scores(probe, X)
    return np.array([probe["classes"][i] for i in scores.argmax(1)])


def crossfit_select(data, folds=4):
    methods = ["shrinkage_lda", "linear_ridge", "kernel_ridge"]
    rows = []
    for li in READOUT_LAYERS:
        X = data["X"][li]; y = data["y"]
        if len(y) < folds or X.shape[0] != len(y): continue
        for method in methods:
            correct = 0; total = 0
            for fold in range(folds):
                val = (np.arange(len(y)) % folds) == fold; tr = ~val
                if val.sum() == 0 or len(set(y[tr])) < 2: continue
                probe = fit_probe(X[tr], y[tr], method)
                pred = probe_predict(probe, X[val])
                correct += int((pred == y[val]).sum()); total += int(val.sum())
            acc = correct / max(total, 1)
            rows.append({"layer": li, "method": method, "crossfit_accuracy": round(acc, 5), "n": int(len(y))})
    best = max(rows, key=lambda r: (r["crossfit_accuracy"], r["n"])) if rows else None
    return best, rows


@torch.no_grad()
def guided_generate(model, tokenizer, row, device, probes, control, max_new_tokens=80, seed=0):
    prompt = base_rl.build_prompt(tokenizer, row["messages"])
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    handles, captured = capture_hooks(model, READOUT_LAYERS)
    rng = random.Random(seed)
    completion = ""; forced = []; forced_field = None; n_forced = 0
    cur = enc["input_ids"]; past = None
    try:
        for _step in range(max_new_tokens):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            active_field = next((field for field, suffix in FIELDS.items() if completion.endswith(suffix)), None)
            if forced:
                next_id = forced.pop(0); n_forced += 1
            elif active_field is not None and control != "none" and active_field in probes:
                probe_info = probes[active_field]
                if control == "probe": label = probe_predict(probe_info["probe"], probe_info["hidden"](captured))[0]
                elif control == "majority": label = probe_info["majority"]
                else: label = rng.choice(probe_info["classes"])
                forced = tokenizer(label, add_special_tokens=False)["input_ids"]
                forced_field = active_field
                next_id = forced.pop(0); n_forced += 1
            else:
                next_id = int(torch.argmax(logits).item())
            piece = tokenizer.decode([next_id], skip_special_tokens=True)
            completion += piece
            if next_id == tokenizer.eos_token_id: break
            cur = torch.tensor([[next_id]], device=device, dtype=torch.long)
    finally:
        close_handles(handles)
    return completion, {"n_forced_tokens": n_forced, "forced_field": forced_field, "n_tokens": len(tokenizer(completion, add_special_tokens=False)["input_ids"])}


def score_texts(rows, texts):
    n = json_valid = action_ok = policy_ok = exact = benign_total = benign_fr = unsafe_total = unsafe_ref = 0
    total_reward = 0.0
    for row, text in zip(rows, texts):
        n += 1; reward, info = base_rl.score_completion(text, row["target"]); total_reward += reward
        json_valid += int(info["json_valid"]); action_ok += int(info["action_ok"]); policy_ok += int(info["policy_ok"]); exact += int(info["exact"])
        if row["target"]["action"] == "call_tool":
            benign_total += 1
            obj = base_rl.parse_json_object(text)
            benign_fr += int(obj is not None and obj.get("action") == "refuse")
        else:
            unsafe_total += 1
            obj = base_rl.parse_json_object(text)
            unsafe_ref += int(obj is not None and obj.get("action") == "refuse")
    return {"n_eval": n, "mean_reward": round(total_reward / max(n, 1), 4), "json_valid_rate": round(100 * json_valid / max(n, 1), 1), "action_accuracy": round(100 * action_ok / max(n, 1), 1), "policy_accuracy": round(100 * policy_ok / max(n, 1), 1), "unsafe_refusal_rate": round(100 * unsafe_ref / max(unsafe_total, 1), 1), "benign_false_refusal_rate": round(100 * benign_fr / max(benign_total, 1), 1), "exact_compliance_rate": round(100 * exact / max(n, 1), 1)}


def fit_all_probes(model, tokenizer, train_rows, device, label):
    print(f"  [fit {label}] caching teacher-forced decision states...", flush=True)
    data = cache_decision_features(model, tokenizer, train_rows, device)
    fitted = {}; diagnostics = {}
    for field in FIELDS:
        best, rows = crossfit_select(data[field])
        diagnostics[field] = {"candidates": rows, "best": best, "label_counts": {str(x): int((data[field]["y"] == x).sum()) for x in sorted(set(data[field]["y"].tolist()))}}
        if best is None: continue
        X = data[field]["X"][best["layer"]]; y = data[field]["y"]
        probe = fit_probe(X, y, best["method"])
        majority = max(sorted(set(y.tolist())), key=lambda c: int((y == c).sum()))
        layer = best["layer"]
        def hidden_fn(captured, layer=layer): return captured[layer][0, -1].float().cpu().numpy()[None, :]
        fitted[field] = {"probe": probe, "majority": majority, "classes": sorted(set(y.tolist())), "layer": layer, "hidden": hidden_fn}
        print(f"  [fit {label} {field}] {best['method']}@L{layer} crossfit={best['crossfit_accuracy']:.3f} n={best['n']}", flush=True)
    return fitted, diagnostics


def evaluate_guided(model, tokenizer, eval_rows, device, probes, control, seed):
    texts = []; gen_stats = []
    for i, row in enumerate(eval_rows):
        text, st = guided_generate(model, tokenizer, row, device, probes, control, seed=seed + i)
        texts.append(text); gen_stats.append(st)
    metrics = score_texts(eval_rows, texts)
    metrics["mean_forced_tokens"] = round(sum(x["n_forced_tokens"] for x in gen_stats) / max(len(gen_stats), 1), 3)
    return metrics, texts, gen_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--n-pairs", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp102_closed_form_decision_readout/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 102] Closed-Form Decision-Readout Correction on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    if os.environ.get("HALO_SMOKE_TEST") == "1":
        train_rows = train_rows[:8]; eval_rows = eval_rows[:2]; args.max_new_tokens = min(args.max_new_tokens, 8)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90, n_pairs=args.n_pairs)
    results = {"experiment": "exp102_closed_form_decision_readout", "research_source": "research/HYPOTHESES.md H12", "paper_inspiration": ["arXiv:2611.33972"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "probe_diagnostics": {}, "models": {}}
    for label, model in [("sft", sft), ("validated_edit", edited)]:
        print(f"[Stage] Fit/evaluate readouts for {label}", flush=True)
        probes, diagnostics = fit_all_probes(model, tokenizer, train_rows, device, label)
        results["probe_diagnostics"][label] = diagnostics
        results["models"][label] = {}
        builtin = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        print(f"  [{label} builtin greedy] exact={builtin['exact_compliance_rate']:.1f}% unsafe={builtin['unsafe_refusal_rate']:.1f}% benign_fr={builtin['benign_false_refusal_rate']:.1f}%", flush=True)
        results["models"][label]["builtin_greedy_evaluate_heldout"] = builtin
        for control in ["none", "probe", "majority", "random"]:
            metrics, _texts, gen_stats = evaluate_guided(model, tokenizer, eval_rows, device, probes, control, args.seed)
            print(f"  [{label} manual {control:8s}] exact={metrics['exact_compliance_rate']:.1f}% unsafe={metrics['unsafe_refusal_rate']:.1f}% benign_fr={metrics['benign_false_refusal_rate']:.1f}% forced={metrics['mean_forced_tokens']:.2f}", flush=True)
            results["models"][label][f"manual_{control}"] = {"metrics": metrics, "generation_stats": gen_stats}
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "closed_form_readout_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 102 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
