#!/usr/bin/env python3
"""Experiment 83: Self-Modeling Failure Probes as RL Advantage Routers.

Implements SUBAGENT3 rank-9: train tiny probes on decision-token hidden states to
predict the model's own verifier failure risk, then use calibrated risk to weight
vector-RL advantages. Probe calibration is reported separately and never overrides
the external verifier or safety rule.

Research anchors: arXiv:2207.05221, arXiv:2306.03341, arXiv:2303.08896.
"""
from __future__ import annotations

import argparse, copy, json, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402

PROBE_LAYERS = [18, 24, 29]


def first_action_pos(ids, prompt_len, action_ids):
    seq = ids.tolist()
    for p in range(prompt_len, len(seq)):
        if seq[p] in action_ids: return p
    return None


def hidden_at_decision(model, ids, prompt_len, layers, action_ids):
    pos = first_action_pos(ids[0], prompt_len, action_ids)
    if pos is None or pos == 0: return None
    caps = {}; handles = []
    for li in layers:
        def hook(_m, _i, out, li=li):
            caps[li] = out[0] if isinstance(out, tuple) else out
        handles.append(model.model.layers[li].register_forward_hook(hook))
    with torch.no_grad(): model(input_ids=ids)
    for h in handles: h.remove()
    if pos - 1 >= caps[layers[0]].shape[1]: return None
    return {li: caps[li][0, pos - 1, :].detach().float().cpu() for li in layers}


def failure_label(text, target, info):
    if not info["json_valid"]: return "malformed_json"
    if info["exact"]: return "correct"
    obj = base_rl.parse_json_object(text) or {}
    if target["action"] == "call_tool" and obj.get("action") == "refuse": return "false_refusal"
    if target["action"] == "refuse" and obj.get("action") == "call_tool": return "missed_unsafe"
    if obj.get("action") == target["action"]: return "wrong_tool_or_policy"
    return "wrong_action"


def collect_probe_data(model, tokenizer, rows, device, group_size, seed):
    torch.manual_seed(seed)
    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[0] for x in ("call", "ref"))
    records = []
    for row_i, row in enumerate(rows):
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.9, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
        for seq in gens:
            text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
            reward, info = base_rl.score_completion(text, row["target"])
            hidden = hidden_at_decision(model, seq.unsqueeze(0), plen, PROBE_LAYERS, action_ids)
            if hidden is None: continue
            records.append({"row_i": row_i, "hidden": hidden, "label": failure_label(text, row["target"], info), "exact": bool(info["exact"]), "reward": float(reward)})
    return records


def auroc(scores, labels):
    order = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0: return 0.5
    rank_sum = sum(i + 1 for i, (_s, y) in enumerate(order) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def train_binary_probe(x, y, folds, seed):
    torch.manual_seed(seed)
    x = x.float(); y = y.float()
    val_probs = torch.zeros_like(y)
    for fold in sorted(set(folds)):
        tr = torch.tensor([f != fold for f in folds]); va = ~tr
        mean = x[tr].mean(0); std = x[tr].std(0, unbiased=False) + 1e-5
        xz = (x - mean) / std
        w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=3e-2)
        pos_w = torch.clamp((1 - y[tr]).sum() / (y[tr].sum() + 1e-6), 0.5, 5.0)
        for _ in range(250):
            opt.zero_grad(); logits = xz[tr] @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, y[tr], pos_weight=pos_w)
            loss.backward(); opt.step()
        with torch.no_grad(): val_probs[va] = torch.sigmoid(xz[va] @ w + b)
    # Refit on all data for deployment.
    deploy_mean = x.mean(0)
    deploy_std = x.std(0, unbiased=False) + 1e-5
    xz = (x - deploy_mean) / deploy_std
    w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=3e-2)
    pos_w = torch.clamp((1 - y).sum() / (y.sum() + 1e-6), 0.5, 5.0)
    for _ in range(300):
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(xz @ w + b, y, pos_weight=pos_w)
        loss.backward(); opt.step()
    return {"w": w.detach(), "b": b.detach(), "mean": deploy_mean, "std": deploy_std, "cv_auroc": auroc(val_probs.tolist(), [int(v) for v in y])}


def train_probes(records):
    probes = {}
    for li in PROBE_LAYERS:
        x = torch.stack([r["hidden"][li] for r in records])
        y = torch.tensor([0.0 if r["exact"] else 1.0 for r in records])
        folds = [r["row_i"] % 4 for r in records]
        probes[str(li)] = train_binary_probe(x, y, folds, 42)
    best = max(probes, key=lambda k: probes[k]["cv_auroc"])
    return probes, int(best)


def probe_risk(probe, hidden):
    z = (hidden.float().cpu() - probe["mean"]) / probe["std"]
    return float(torch.sigmoid(z @ probe["w"] + probe["b"]).item())


def train_router_arm(arm, init_model, ref_model, tokenizer, prompt_stream, device, probe, probe_layer, steps, group_size, lr, beta, seed, risk_weight):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(init_model).to(device); vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vec_rl.DISJOINT_LAYERS, device)
    opt = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[0] for x in ("call", "ref"))
    rng = random.Random(seed); trace = []
    for step in range(1, steps + 1):
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval(); groups = []; rewards_all = []; risks_all = []
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad(): gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
            items = []
            for seq in gens:
                text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
                reward, info = base_rl.score_completion(text, row["target"])
                hidden = hidden_at_decision(model, seq.unsqueeze(0), plen, [probe_layer], action_ids)
                if arm == "probe_routed" and hidden is not None: risk = probe_risk(probe, hidden[probe_layer])
                elif arm == "random_risk_control": risk = rng.random()
                else: risk = 0.5
                rewards_all.append(float(reward)); risks_all.append(risk)
                items.append({"seq": seq, "reward": float(reward), "exact": info["exact"], "risk": risk})
            r = torch.tensor([x["reward"] for x in items], dtype=torch.float32, device=device)
            if float(r.std(unbiased=False).item()) < 1e-6: continue
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            for item, a in zip(items, adv): item["adv"] = a.detach()
            groups.append({"items": items, "prompt_len": plen})
        model.train(); opt.zero_grad(set_to_none=True); losses = []; kls = []
        for group in groups:
            for item in group["items"]:
                ids = item["seq"].unsqueeze(0)
                lp = base_rl.completion_logprobs(model, ids, group["prompt_len"])
                with torch.no_grad(): lp_ref = base_rl.completion_logprobs(ref_model, ids, group["prompt_len"])
                kl = (lp - lp_ref).abs().mean(); kls.append(float(kl.detach().item()))
                if item["reward"] < 1.0: weight = 1.0 + risk_weight * item["risk"]
                else: weight = max(0.25, 1.0 - 0.5 * risk_weight * item["risk"])
                if arm == "uniform_control": weight = 1.0
                losses.append(-float(weight) * item["adv"] * lp + beta * kl)
        loss_val = grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean(); loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
            opt.step(); loss_val = float(loss.detach().item())
        rec = {"arm": arm, "step": step, "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4), "mean_probe_risk": round(sum(risks_all) / max(len(risks_all), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5)}
        trace.append(rec)
        print(f"  [{arm}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | risk={rec['mean_probe_risk']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']}", flush=True)
    model.eval(); return model, {"trace": trace, "n_vector_params": sum(v.numel() for v in vectors.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-rows", type=int, default=24)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--risk-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp83_self_modeling_failure_probe_router/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 83] Self-Modeling Failure Probe Router on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)
    print("[Stage 1] Collecting decision-token rollout outcomes and training failure-risk probes...", flush=True)
    records = collect_probe_data(edited, tokenizer, prompt_stream[:args.probe_rows], device, args.group_size, args.seed)
    label_counts = {}
    for r in records: label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    probes, best_layer = train_probes(records)
    print(f"  labels={label_counts} | best_layer=L{best_layer} cv_auroc={probes[str(best_layer)]['cv_auroc']:.3f}", flush=True)
    probe_meta = {k: {"cv_auroc": round(v["cv_auroc"], 5)} for k, v in probes.items()}
    results = {"experiment": "exp83_self_modeling_failure_probe_router", "research_source": "research/SUBAGENT3_CREATIVE_HEURISTICS.md rank 9", "paper_inspiration": ["arXiv:2207.05221", "arXiv:2306.03341", "arXiv:2303.08896"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "baseline_sft": sft_eval, "anchored_edit_only": edit_eval, "probe_data": {"n_records": len(records), "label_counts": label_counts, "cv_note": "4-fold prompt-index holdout proxy; no inference-time override of verifier or safety rules"}, "probe_cv": probe_meta, "selected_layer": best_layer, "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm in ["uniform_control", "probe_routed", "random_risk_control"]:
            print(f"[Run] {arm}", flush=True)
            trained, stats = train_router_arm(arm, edited, ref_model, tokenizer, prompt_stream, device, probes[str(best_layer)], best_layer, args.steps, args.group_size, args.lr, args.beta, args.seed, args.risk_weight)
            for rec in stats["trace"]: log.write(json.dumps(rec) + "\n")
            log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm] = {"train_stats": stats, "eval": ev}
            del trained
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "failure_probe_router_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 83 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
