#!/usr/bin/env python3
"""Experiment 80: Edit-Strength Scaffold Annealing.

Implements SUBAGENT3 rank-8: use a strong surgical edit as temporary training
scaffolding, fade it while disjoint steering-vector RL learns to own the behavior,
and evaluate probe copies with the scaffold removed. Compares fixed-strength,
RL-only, immediate-removal, and annealed schedules at identical gradient-step count.

Research anchors: arXiv:2203.05482 and arXiv:1506.01186.
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


def snapshot_down_weights(model, layers):
    return {li: model.model.layers[li].mlp.down_proj.weight.detach().clone() for li in layers}


def materialize_edit(model, base_weights, subspaces, selected, strength: float):
    with torch.no_grad():
        for li, w0 in base_weights.items():
            down = model.model.layers[li].mlp.down_proj
            w = w0.detach().clone().float()
            U = subspaces[li].to(device=w.device, dtype=torch.float32)
            idx = torch.tensor(selected[li], device=w.device, dtype=torch.long)
            proj = torch.matmul(U.T, w)
            w[:, idx] = w[:, idx] - float(strength) * torch.matmul(U, proj[:, idx])
            down.weight.data.copy_(w.to(dtype=down.weight.dtype))


def schedule_for(arm: str, step: int) -> float:
    stage = min((step - 1) // 2, 3)
    if arm == "annealed_scaffold":
        return [1.5, 1.0, 0.5, 0.0][stage]
    if arm == "fixed_strength":
        return 1.0
    if arm == "immediate_removal":
        return 1.5 if step <= 2 else 0.0
    return 0.0


def train_arm(arm_name, sft, ref_model, tokenizer, prompt_stream, device, base_weights, subspaces, selected, steps, group_size, lr, beta, seed, eval_rows):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(sft).to(device)
    vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vec_rl.DISJOINT_LAYERS, device)
    opt = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    trace = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        strength = schedule_for(arm_name, step)
        materialize_edit(model, base_weights, subspaces, selected, strength)
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval(); groups = []; rewards_all = []
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
            items = []
            for seq in gens:
                text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
                reward, info = base_rl.score_completion(text, row["target"])
                rewards_all.append(float(reward)); items.append({"seq": seq, "reward": float(reward), "exact": info["exact"]})
            r = torch.tensor([x["reward"] for x in items], dtype=torch.float32, device=device)
            if float(r.std(unbiased=False).item()) < 1e-6:
                continue
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            for item, a in zip(items, adv): item["adv"] = a.detach()
            groups.append({"items": items, "prompt_len": plen})
        model.train(); opt.zero_grad(set_to_none=True)
        losses = []; kls = []
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
        rec = {"arm": arm_name, "step": step, "scaffold_strength": strength, "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5)}
        if step % 2 == 0 or step == steps:
            materialize_edit(model, base_weights, subspaces, selected, 0.0)
            probe = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
            materialize_edit(model, base_weights, subspaces, selected, strength)
            rec["probe_lambda0_exact"] = probe["exact_compliance_rate"]
            rec["probe_lambda0_unsafe_refusal"] = probe["unsafe_refusal_rate"]
            rec["probe_lambda0_benign_false_refusal"] = probe["benign_false_refusal_rate"]
        else:
            rec["probe_lambda0_exact"] = None
        trace.append(rec)
        print(f"  [{arm_name}] step {step:02d}/{steps} | lambda={strength:.1f} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | probe0={rec.get('probe_lambda0_exact')}", flush=True)
    materialize_edit(model, base_weights, subspaces, selected, 0.0)
    eval_zero = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
    final_strength = schedule_for(arm_name, steps)
    materialize_edit(model, base_weights, subspaces, selected, final_strength)
    eval_current = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
    stats = {"trace": trace, "train_wall_sec": round(time.perf_counter() - t0, 2), "n_vector_params": sum(v.numel() for v in vectors.values()), "vector_norms": {str(k): round(float(v.detach().norm().item()), 5) for k, v in vectors.items()}, "final_scaffold_strength": final_strength}
    return model, stats, eval_zero, eval_current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp80_edit_strength_scaffold_annealing/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 80] Edit-Strength Scaffold Annealing on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    layers = list(vec_rl.EDIT_LAYERS)
    base_weights = snapshot_down_weights(sft, layers)
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, layers, n_pairs=12)
    selected = top_projection_columns(sft, subspaces, layers, 64)
    scaffolded = copy.deepcopy(sft)
    materialize_edit(scaffolded, base_weights, subspaces, selected, 1.0)
    ref_model = copy.deepcopy(scaffolded).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    fixed_eval = base_rl.evaluate_heldout(scaffolded, tokenizer, eval_rows, device)
    print(f"  [SFT lambda0] exact={sft_eval['exact_compliance_rate']:.1f}% | [edit lambda1] exact={fixed_eval['exact_compliance_rate']:.1f}% unsafe={fixed_eval['unsafe_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp80_edit_strength_scaffold_annealing", "research_source": "research/SUBAGENT3_CREATIVE_HEURISTICS.md rank 8", "paper_inspiration": ["arXiv:2203.05482", "arXiv:1506.01186"], "config": vars(args), "device": str(device), "edit_layers": layers, "selected_neurons": selected, "top_singular_values": singulars, "baseline_sft_lambda0": sft_eval, "fixed_edit_only_lambda1": fixed_eval, "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm in ["annealed_scaffold", "fixed_strength", "immediate_removal", "rl_only"]:
            print(f"[Run] {arm}", flush=True)
            trained, stats, eval_zero, eval_current = train_arm(arm, sft, ref_model, tokenizer, prompt_stream, device, base_weights, subspaces, selected, args.steps, args.group_size, args.lr, args.beta, args.seed, eval_rows)
            for rec in stats["trace"]: log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  [{arm} final lambda0] exact={eval_zero['exact_compliance_rate']:.1f}% unsafe={eval_zero['unsafe_refusal_rate']:.1f}% benign_fr={eval_zero['benign_false_refusal_rate']:.1f}%", flush=True)
            print(f"  [{arm} final current ] exact={eval_current['exact_compliance_rate']:.1f}% unsafe={eval_current['unsafe_refusal_rate']:.1f}% benign_fr={eval_current['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm] = {"train_stats": stats, "eval_lambda0_primary": eval_zero, "eval_current_scaffold": eval_current}
            del trained
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "edit_strength_scaffold_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 80 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
