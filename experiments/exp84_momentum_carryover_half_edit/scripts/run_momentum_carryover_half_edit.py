#!/usr/bin/env python3
"""Experiment 84: Momentum-Carryover Half-Edits.

Implements SUBAGENT3 rank-6. Part of the validated surgical edit is applied in
weight space and the remainder is encoded as AdamW first momentum in directly
trainable sparse down_proj columns. Tests whether RL rollback is caused by
momentum incompatibility rather than the edit direction itself.

Research anchors: arXiv:1412.6980 and arXiv:2203.05482.
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

COMPLEMENT_LAYERS = [24, 25, 26, 27]


def compute_full_delta(base_weights, subspaces, selected):
    deltas = {}
    for li, w0 in base_weights.items():
        U = subspaces[li].to(device=w0.device, dtype=torch.float32)
        idx = torch.tensor(selected[li], device=w0.device, dtype=torch.long)
        proj = torch.matmul(U.T, w0)
        col_delta = -torch.matmul(U, proj[:, idx])
        full = torch.zeros_like(w0)
        full[:, idx] = col_delta
        deltas[li] = full
    return deltas


def initialize_weights(model, base_weights, deltas, direct_scale):
    with torch.no_grad():
        for li, w0 in base_weights.items():
            model.model.layers[li].mlp.down_proj.weight.copy_((w0 + direct_scale * deltas[li]).to(model.model.layers[li].mlp.down_proj.weight.dtype))


def prepare_sparse_trainables(model, selected):
    vec_rl.freeze_weights(model)
    params = []; edited_params = {}
    for li, idxs in selected.items():
        p = model.model.layers[li].mlp.down_proj.weight
        mask = torch.zeros_like(p)
        idx = torch.tensor(idxs, device=p.device, dtype=torch.long)
        mask[:, idx] = 1.0
        p.requires_grad_(True); p.register_hook(lambda g, mask=mask: g * mask)
        params.append(p); edited_params[li] = p
    for li in COMPLEMENT_LAYERS:
        p = model.model.layers[li].mlp.down_proj.weight
        p.requires_grad_(True); params.append(p)
    return params, edited_params


def seed_adam_momentum(opt, edited_params, deltas, mode, seed, momentum_scale):
    rng = torch.Generator().manual_seed(seed)
    for li, p in edited_params.items():
        state = opt.state[p]
        state["step"] = torch.tensor(10.0, device=p.device)
        direction = -momentum_scale * deltas[li].to(device=p.device, dtype=p.dtype)
        if mode == "random_sign":
            signs_cpu = torch.where(torch.rand(direction.shape, generator=rng) < 0.5, -1.0, 1.0)
            direction = direction * signs_cpu.to(device=direction.device, dtype=direction.dtype)
        state["exp_avg"] = direction.clone()
        state["exp_avg_sq"] = torch.full_like(direction, (0.1 * momentum_scale) ** 2)


def rollback_cosine(model, initial_weights, deltas, selected):
    num = den_d = den_w = 0.0
    with torch.no_grad():
        for li, idxs in selected.items():
            idx = torch.tensor(idxs, device=model.model.layers[li].mlp.down_proj.weight.device)
            cur = model.model.layers[li].mlp.down_proj.weight[:, idx].float()
            init = initial_weights[li][:, idx].float().to(cur.device)
            delta = deltas[li][:, idx].float().to(cur.device)
            diff = cur - init
            num += float((diff * delta).sum().item()); den_w += float((diff * diff).sum().item()); den_d += float((delta * delta).sum().item())
    return num / ((den_w ** 0.5) * (den_d ** 0.5) + 1e-12)


def train_arm(arm, init_model, ref_model, tokenizer, prompt_stream, device, base_weights, deltas, selected, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    direct_scale = {"hard_edit_no_momentum": 1.0, "half_edit_momentum": 0.5, "momentum_only_ghost": 0.0, "random_sign_momentum": 0.5}[arm]
    initialize_weights(model, base_weights, deltas, direct_scale)
    params, edited_params = prepare_sparse_trainables(model, selected)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if arm == "half_edit_momentum": seed_adam_momentum(opt, edited_params, deltas, "surgical", seed, momentum_scale=0.5)
    elif arm == "momentum_only_ghost": seed_adam_momentum(opt, edited_params, deltas, "surgical", seed, momentum_scale=1.0)
    elif arm == "random_sign_momentum": seed_adam_momentum(opt, edited_params, deltas, "random_sign", seed, momentum_scale=0.5)
    initial_weights = {li: p.detach().clone().cpu() for li, p in edited_params.items()}
    trace = []
    for step in range(1, steps + 1):
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
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
            opt.step(); loss_val = float(loss.detach().item())
        rollback = rollback_cosine(model, initial_weights, deltas, selected)
        rec = {"arm": arm, "step": step, "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "rollback_cosine": round(rollback, 5), "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5)}
        trace.append(rec)
        print(f"  [{arm}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | rollback_cos={rollback:+.3f} | active={len(groups)}/2 dead={rec['dead_groups']}", flush=True)
    stats = {"trace": trace, "optimizer_tensor_params": sum(p.numel() for p in params), "effective_sparse_note": "selected edited columns plus full down_proj in complementary layers"}
    model.eval(); return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp84_momentum_carryover_half_edit/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 84] Momentum-Carryover Half-Edits on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    layers = list(vec_rl.EDIT_LAYERS)
    base_weights = {li: sft.model.layers[li].mlp.down_proj.weight.detach().clone() for li in layers}
    subspaces, singulars = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, layers, n_pairs=12)
    selected = top_projection_columns(sft, subspaces, layers, 64)
    deltas = compute_full_delta(base_weights, subspaces, selected)
    hard = copy.deepcopy(sft); initialize_weights(hard, base_weights, deltas, 1.0)
    ref_model = copy.deepcopy(hard).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); hard_eval = base_rl.evaluate_heldout(hard, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [hard edit] exact={hard_eval['exact_compliance_rate']:.1f}% unsafe={hard_eval['unsafe_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp84_momentum_carryover_half_edit", "research_source": "research/SUBAGENT3_CREATIVE_HEURISTICS.md rank 6", "paper_inspiration": ["arXiv:1412.6980", "arXiv:2203.05482"], "config": vars(args), "device": str(device), "selected_neurons": selected, "top_singular_values": singulars, "baseline_sft": sft_eval, "hard_edit_only": hard_eval, "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm in ["hard_edit_no_momentum", "half_edit_momentum", "momentum_only_ghost", "random_sign_momentum"]:
            print(f"[Run] {arm}", flush=True)
            trained, stats = train_arm(arm, sft, ref_model, tokenizer, prompt_stream, device, base_weights, deltas, selected, args.steps, args.group_size, args.lr, args.beta, args.seed)
            for rec in stats["trace"]: log.write(json.dumps(rec) + "\n")
            log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm] = {"train_stats": stats, "eval": ev}
            del trained
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "momentum_carryover_half_edit_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 84 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
