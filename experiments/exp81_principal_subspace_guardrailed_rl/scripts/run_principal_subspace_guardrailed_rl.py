#!/usr/bin/env python3
"""Experiment 81: Principal-Subspace Guardrails for Edit-Safe RL.

Implements SUBAGENT1 rank-9. Full matrices in complementary layers are trainable,
but gradients in the constrained arm are projected into the bilateral orthogonal
complement of dominant pretrained singular subspaces. Measures forbidden-subspace
overlap even when unconstrained, linking geometric intrusion to exact-compliance
and safety outcomes.

Research anchors: arXiv:2608.11674, arXiv:2609.34344, arXiv:2606.31813.
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
from validated_edit import apply_validated_rank1_edit  # noqa: E402

TRAIN_LAYERS = [24, 25, 26, 27]
MODULE_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def matrix_named_modules(model):
    for li in TRAIN_LAYERS:
        layer = model.model.layers[li]
        mods = {"q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj, "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj, "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj, "down_proj": layer.mlp.down_proj}
        for name, mod in mods.items():
            yield li, name, mod


def spectral_factors(model, k: int):
    factors = {}
    with torch.no_grad():
        for li, name, mod in matrix_named_modules(model):
            w = mod.weight.detach().float()
            u, _s, vh = torch.linalg.svd(w, full_matrices=False)
            kk = min(k, u.shape[1])
            factors[(li, name)] = {"u": u[:, :kk].clone(), "v": vh.T[:, :kk].clone(), "top_singular": float(_s[0].item()), "k": kk}
    return factors


def overlap_and_project(grad, u, v, mode):
    left = u @ (u.T @ grad)
    right = (grad @ v) @ v.T
    both = (u @ (u.T @ grad @ v)) @ v.T
    forbidden = left + right - both
    frac = float((forbidden.norm() / (grad.norm() + 1e-12)).detach().cpu())
    if mode == "bilateral":
        return grad - forbidden, frac
    if mode == "left_only":
        return grad - left, frac
    return grad, frac


def prepare_model(init_model, factors, mode, device):
    model = copy.deepcopy(init_model).to(device)
    vec_rl.freeze_weights(model)
    params = []
    stats = {"overlap_sum": 0.0, "overlap_count": 0}
    for li, name, mod in matrix_named_modules(model):
        p = mod.weight
        p.requires_grad_(True)
        params.append(p)
        f = factors[(li, name)]
        u = f["u"].to(device=p.device, dtype=p.dtype)
        v = f["v"].to(device=p.device, dtype=p.dtype)
        def hook(grad, u=u, v=v, mode=mode, stats=stats):
            projected, frac = overlap_and_project(grad, u, v, mode)
            stats["overlap_sum"] += frac; stats["overlap_count"] += 1
            return projected
        p.register_hook(hook)
    return model, params, stats


def train_matrix_arm(arm_name, mode, init_model, ref_model, tokenizer, prompt_stream, device, factors, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed); random.seed(seed)
    model, params, overlap_stats = prepare_model(init_model, factors, mode, device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    trace = []; t0 = time.perf_counter()
    for step in range(1, steps + 1):
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
            if float(r.std(unbiased=False).item()) < 1e-6: continue
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            for item, a in zip(items, adv): item["adv"] = a.detach()
            groups.append({"items": items, "prompt_len": plen})
        model.train(); opt.zero_grad(set_to_none=True)
        before_count = overlap_stats["overlap_count"]
        before_sum = overlap_stats["overlap_sum"]
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
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
            opt.step(); loss_val = float(loss.detach().item())
        n_new = overlap_stats["overlap_count"] - before_count
        mean_overlap = (overlap_stats["overlap_sum"] - before_sum) / max(n_new, 1)
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "forbidden_grad_overlap": round(mean_overlap, 5), "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5)}
        trace.append(rec)
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | forbidden={mean_overlap:.3f}", flush=True)
    stats = {"trace": trace, "train_wall_sec": round(time.perf_counter() - t0, 2), "optimizer_params": sum(p.numel() for p in params), "mean_forbidden_overlap": round(overlap_stats["overlap_sum"] / max(overlap_stats["overlap_count"], 1), 5)}
    model.eval(); return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp81_principal_subspace_guardrailed_rl/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 81] Principal-Subspace Guardrailed RL on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [validated edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)
    print(f"[Stage 1] Computing top-{args.rank} singular subspaces for L24-27 attention+MLP matrices...", flush=True)
    factors = spectral_factors(edited, args.rank)
    results = {"experiment": "exp81_principal_subspace_guardrailed_rl", "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 9", "paper_inspiration": ["arXiv:2608.11674", "arXiv:2609.34344", "arXiv:2606.31813"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "train_layers": TRAIN_LAYERS, "baseline_sft": sft_eval, "anchored_edit_only": edit_eval, "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm, mode in [("kl_layer_disjoint_unconstrained", "none"), ("bilateral_complement_constrained", "bilateral"), ("left_subspace_only_control", "left_only")]:
            print(f"[Run] {arm}", flush=True)
            trained, stats = train_matrix_arm(arm, mode, edited, ref_model, tokenizer, prompt_stream, device, factors, args.steps, args.group_size, args.lr, args.beta, args.seed)
            for rec in stats["trace"]: log.write(dumps_safe(rec) + "\n")
            log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm] = {"train_stats": stats, "eval": ev}
            del trained
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "principal_subspace_guardrailed_summary.json"; write_json(path, results)
    print(f"[Exp 81 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
