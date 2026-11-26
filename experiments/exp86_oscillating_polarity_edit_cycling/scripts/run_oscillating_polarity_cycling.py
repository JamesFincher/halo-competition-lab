#!/usr/bin/env python3
"""Experiment 86: Oscillating-Polarity Adversarial Edit Cycling.

Implements SUBAGENT3 rank-10 in a <=10-step adaptation. A small validated surgical
perturbation flips sign during vector-RL cycles with decaying amplitude. Last
positive/negative vector endpoints are then souped and evaluated with the manual
scaffold removed, testing whether behavior depends on a fragile single polarity.

Research anchors: arXiv:1506.01186 and arXiv:2203.05482.
"""
from __future__ import annotations

import argparse, copy, json, random, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
from serialization_utils import dumps_safe, write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import build_validated_rank1_subspaces, top_projection_columns  # noqa: E402


def materialize_edit(model, base_weights, subspaces, selected, strength: float):
    with torch.no_grad():
        for li, w0 in base_weights.items():
            down = model.model.layers[li].mlp.down_proj
            w = w0.clone().float()
            U = subspaces[li].to(device=w.device, dtype=torch.float32)
            idx = torch.tensor(selected[li], device=w.device, dtype=torch.long)
            proj = torch.matmul(U.T, w)
            w[:, idx] = w[:, idx] - float(strength) * torch.matmul(U, proj[:, idx])
            down.weight.data.copy_(w.to(dtype=down.weight.dtype))


def polarity_strength(arm: str, step: int) -> float:
    if arm == "constant_positive_control": return 0.5
    if arm == "constant_negative_control": return -0.5
    table = {1: 0.5, 2: 0.5, 3: -0.5, 4: -0.5, 5: 0.2, 6: 0.2, 7: -0.2, 8: -0.2, 9: 0.1, 10: -0.1}
    return table.get(step, 0.0)


def vector_snapshot(vectors):
    return {li: v.detach().clone().cpu() for li, v in vectors.items()}


def set_vectors(vectors, values):
    with torch.no_grad():
        for li, v in values.items():
            vectors[li].copy_(v.to(device=vectors[li].device, dtype=vectors[li].dtype))


def train_polarity_arm(arm, sft, ref_model, tokenizer, prompt_stream, device, base_weights, subspaces, selected, eval_rows, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(sft).to(device); vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vec_rl.DISJOINT_LAYERS, device)
    opt = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    trace = []; last_pos = last_neg = None
    for step in range(1, steps + 1):
        strength = polarity_strength(arm, step)
        materialize_edit(model, base_weights, subspaces, selected, strength)
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval(); groups = []; rewards_all = []
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad(): gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
            items = []
            for seq in gens:
                text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
                reward, info = base_rl.score_completion(text, row["target"])
                rewards_all.append(float(reward)); items.append({"seq": seq, "reward": float(reward), "exact": info["exact"]})
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
                losses.append(-item["adv"] * lp + beta * kl)
        loss_val = grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean(); loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
            opt.step(); loss_val = float(loss.detach().item())
        snap = vector_snapshot(vectors)
        if strength > 0: last_pos = snap
        elif strength < 0: last_neg = snap
        rec = {"arm": arm, "step": step, "signed_strength": strength, "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5)}
        trace.append(rec)
        print(f"  [{arm}] step {step:02d}/{steps} | s={strength:+.1f} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']}", flush=True)
    final = vector_snapshot(vectors)
    if last_pos is None: last_pos = final
    if last_neg is None: last_neg = final
    soup = {li: (last_pos[li] + last_neg[li]) * 0.5 for li in last_pos}
    endpoints = {}
    for name, vals in [("final_vectors", final), ("last_positive_endpoint", last_pos), ("last_negative_endpoint", last_neg), ("positive_negative_vector_soup", soup)]:
        set_vectors(vectors, vals)
        materialize_edit(model, base_weights, subspaces, selected, 0.0)
        endpoints[name] = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        print(f"  [{arm} {name} lambda0] exact={endpoints[name]['exact_compliance_rate']:.1f}% unsafe={endpoints[name]['unsafe_refusal_rate']:.1f}% benign_fr={endpoints[name]['benign_false_refusal_rate']:.1f}%", flush=True)
    set_vectors(vectors, final)
    stats = {"trace": trace, "n_vector_params": sum(v.numel() for v in vectors.values()), "endpoint_evals_lambda0": endpoints}
    model.eval(); return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp86_oscillating_polarity_edit_cycling/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 86] Oscillating-Polarity Adversarial Edit Cycling on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    layers = list(vec_rl.EDIT_LAYERS)
    base_weights = {li: sft.model.layers[li].mlp.down_proj.weight.detach().clone() for li in layers}
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, layers, n_pairs=12)
    selected = top_projection_columns(sft, subspaces, layers, 64)
    accepted = copy.deepcopy(sft); materialize_edit(accepted, base_weights, subspaces, selected, 0.9)
    ref_model = copy.deepcopy(accepted).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); accepted_eval = base_rl.evaluate_heldout(accepted, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [accepted edit] exact={accepted_eval['exact_compliance_rate']:.1f}% unsafe={accepted_eval['unsafe_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp86_oscillating_polarity_edit_cycling", "research_source": "research/SUBAGENT3_CREATIVE_HEURISTICS.md rank 10", "paper_inspiration": ["arXiv:1506.01186", "arXiv:2203.05482"], "config": vars(args), "device": str(device), "selected_neurons": selected, "top_singular_values": singulars, "baseline_sft": sft_eval, "accepted_edit_only": accepted_eval, "schedule_note": "compressed <=10-step adaptation: amplitudes .5/.2/.1, sign flips every 2 steps until final single-step .1 pair", "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm in ["oscillating_decay", "constant_positive_control", "constant_negative_control"]:
            print(f"[Run] {arm}", flush=True)
            trained, stats = train_polarity_arm(arm, sft, ref_model, tokenizer, prompt_stream, device, base_weights, subspaces, selected, eval_rows, args.steps, args.group_size, args.lr, args.beta, args.seed)
            for rec in stats["trace"]: log.write(dumps_safe(rec) + "\n")
            log.flush()
            results["arms"][arm] = {"train_stats": stats}
            del trained
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "oscillating_polarity_cycling_summary.json"; write_json(path, results)
    print(f"[Exp 86 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
