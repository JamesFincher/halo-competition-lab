#!/usr/bin/env python3
"""Experiment 121: Curvature-Constrained Acquisition Delta RL.

Implements a diagonal-factor approximation of HYPOTHESES H9. Preservation-loss
Fisher curvature is estimated on correct benign/unsafe trajectories. Direct
full-matrix acquisition RL is then restricted to low-curvature coordinates by
gradient masks and compared against random, high-curvature, and unconstrained
controls in both unedited-SFT and validated-post-edit start states.

Research anchors: research/HYPOTHESES.md H9 and arXiv:2611.24654.
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

LAYERS = [24, 25, 26, 27]
MODULE_PARTS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]


def candidate_params(model):
    out = []
    for name, p in model.named_parameters():
        if not name.endswith("weight"): continue
        if any(f".layers.{li}." in name for li in LAYERS) and any(part in name for part in MODULE_PARTS): out.append((name, p))
    if not out: raise RuntimeError("no curvature candidate weights found")
    return out


def estimate_fisher(model, tokenizer, rows, device, candidates, n_examples):
    for p in model.parameters(): p.requires_grad_(False)
    for _n, p in candidates: p.requires_grad_(True)
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in candidates}
    fa = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in candidates}
    fb = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in candidates}
    counts = [0, 0]
    model.eval()
    for i, row in enumerate(rows[:n_examples]):
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        comp_ids = tokenizer(row["assistant_text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        ids = torch.cat([prompt_ids, comp_ids], dim=1)
        loss = -base_rl.completion_logprobs(model, ids, prompt_ids.shape[1])
        model.zero_grad(set_to_none=True); loss.backward()
        bucket = fa if i % 2 == 0 else fb; counts[i % 2] += 1
        for n, p in candidates:
            if p.grad is None: continue
            g2 = p.grad.detach().float().pow(2)
            fisher[n] += g2; bucket[n] += g2
    for n in fisher:
        fisher[n] /= max(sum(counts), 1); fa[n] /= max(counts[0], 1); fb[n] /= max(counts[1], 1)
    return fisher, fa, fb


def corr_flat(a, b):
    x = torch.log(a.flatten().float() + 1e-12); y = torch.log(b.flatten().float() + 1e-12)
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y / (x.norm() * y.norm() + 1e-8)).item())


def make_masks(fisher, fraction, mode, seed):
    masks = {}; counts = {}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for n, f in fisher.items():
        flat = f.flatten(); k = max(1, int(round(flat.numel() * fraction)))
        if mode == "random":
            order = torch.randperm(flat.numel(), generator=rng)[:k].to(f.device)
            mask = torch.zeros_like(flat, dtype=torch.bool); mask[order] = True
        else:
            largest = mode == "high"
            idx = torch.topk(flat, k=k, largest=largest).indices
            mask = torch.zeros_like(flat, dtype=torch.bool); mask[idx] = True
        masks[n] = mask.view_as(f)
        counts[n] = {"kept": int(mask.sum().item()), "total": int(mask.numel()), "fraction": round(float(mask.float().mean().item()), 5)}
    return masks, counts


def stability_metrics(fa, fb, fraction):
    rows = []
    for n in fa:
        c = corr_flat(fa[n], fb[n])
        ma, _ = make_masks({n: fa[n]}, fraction, "low", 1); mb, _ = make_masks({n: fb[n]}, fraction, "low", 2)
        overlap = float((ma[n] & mb[n]).float().sum().item()) / max(float(ma[n].sum().item()), 1.0)
        rows.append({"parameter": n, "log_fisher_split_corr": round(c, 5), "bottom_fraction_mask_overlap": round(overlap, 5)})
    return rows


def train_masked_arm(arm_name, init_model, ref_model, tokenizer, prompt_stream, device, masks, steps, group_size, lr, beta, seed, log):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(init_model).to(device); vec_rl.freeze_weights(model)
    selected = []
    for n, p in model.named_parameters():
        if n in masks:
            p.requires_grad_(True)
            mask = masks[n].to(device=p.device)
            p.register_hook(lambda g, mask=mask: g * mask.to(dtype=g.dtype))
            selected.append((n, p))
    params = [p for _n, p in selected]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    trace = []; t0 = time.perf_counter(); initial = {n: p.detach().clone() for n, p in selected}
    for step in range(1, steps + 1):
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval(); groups = []; all_rewards = []
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device); plen = enc["input_ids"].shape[1]
            with torch.no_grad(): gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
            seqs = []; rewards = []
            for seq in gens:
                text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
                reward, _info = base_rl.score_completion(text, row["target"])
                seqs.append(seq); rewards.append(float(reward)); all_rewards.append(float(reward))
            r = torch.tensor(rewards, dtype=torch.float32, device=device)
            if float(r.std(unbiased=False).item()) < 1e-6: continue
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            groups.append((seqs, plen, adv))
        model.train(); opt.zero_grad(set_to_none=True); losses = []; kls = []
        for seqs, plen, advs in groups:
            for seq, adv in zip(seqs, advs):
                ids = seq.unsqueeze(0)
                lp = base_rl.completion_logprobs(model, ids, plen)
                with torch.no_grad(): lp_ref = base_rl.completion_logprobs(ref_model, ids, plen)
                kl = (lp - lp_ref).abs().mean(); kls.append(float(kl.detach().item()))
                losses.append((-adv.detach() * lp + beta * kl) / max(len(groups), 1) / group_size)
        grad_norm = loss_val = 0.0
        if losses:
            loss = torch.stack(losses).sum(); loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item()); opt.step(); loss_val = float(loss.detach().item())
        drift = float(sum(((p.detach() - initial[n]).float().norm() ** 2).item() for n, p in selected) ** 0.5)
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5), "weight_drift_l2": round(drift, 5)}
        trace.append(rec); log.write(dumps_safe(rec) + "\n"); log.flush()
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | kl={rec['mean_abs_kl']:.4f} | drift={drift:.3f}", flush=True)
    stats = {"trainable_params": sum(p.numel() for p in params), "masked_trainable_coordinates": int(sum(masks[n].sum().item() for n, _p in selected)), "final_weight_drift_l2": trace[-1]["weight_drift_l2"] if trace else 0.0, "train_wall_sec": round(time.perf_counter() - t0, 2), "trace": trace}
    model.eval(); return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp121_curvature_constrained_delta_rl/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 121] Curvature-Constrained Acquisition Delta RL on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"][: args.n_pairs // 2]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"][: args.n_pairs // 2]
    preservation = [x for pair in zip(benign, unsafe) for x in pair]
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [post-edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}% benign_fr={edit_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp121_curvature_constrained_delta_rl", "research_source": "research/HYPOTHESES.md H9", "paper_inspiration": ["arXiv:2611.24654"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "curvature_note": "diagonal Fisher/empirical curvature proxy on correct policy trajectories; real-benchmark preservation text is unavailable in this repo snapshot", "layers": LAYERS, "module_parts": MODULE_PARTS, "baseline_sft": sft_eval, "post_edit_anchor": edit_eval, "starts": {}}
    start_specs = [("unedited_sft", sft), ("validated_post_edit", edited)]
    curvature_cache = {}
    for label, model in start_specs:
        print(f"[Curvature] estimating diagonal preservation Fisher for {label}...", flush=True)
        candidates = candidate_params(model)
        fisher, fa, fb = estimate_fisher(model, tokenizer, preservation, device, candidates, args.n_pairs)
        stab = stability_metrics(fa, fb, args.fraction)
        curvature_cache[label] = {"fisher": fisher, "stability": stab}
        print(f"  mean split corr={sum(x['log_fisher_split_corr'] for x in stab)/len(stab):.3f} mean mask overlap={sum(x['bottom_fraction_mask_overlap'] for x in stab)/len(stab):.3f}", flush=True)
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for label, start in start_specs:
            ref_model = copy.deepcopy(start).to(device).eval(); vec_rl.freeze_weights(ref_model)
            fisher = curvature_cache[label]["fisher"]
            all_masks = {n: torch.ones_like(f, dtype=torch.bool) for n, f in fisher.items()}
            arm_specs = []
            if label == "validated_post_edit":
                low, low_counts = make_masks(fisher, args.fraction, "low", args.seed)
                high, high_counts = make_masks(fisher, args.fraction, "high", args.seed)
                rnd, rnd_counts = make_masks(fisher, args.fraction, "random", args.seed)
                arm_specs += [("low_curvature_bottom20", low, low_counts), ("random_mask20_control", rnd, rnd_counts), ("high_curvature_top20_negative", high, high_counts), ("unconstrained_dense_control", all_masks, {})]
            else:
                low, low_counts = make_masks(fisher, args.fraction, "low", args.seed)
                arm_specs += [("low_curvature_bottom20", low, low_counts), ("unconstrained_dense_control", all_masks, {})]
            results["starts"][label] = {"curvature_stability": curvature_cache[label]["stability"], "arms": {}}
            for arm, masks, counts in arm_specs:
                full_arm = f"{label}_{arm}"
                print(f"[Run] {full_arm}", flush=True)
                model, stats = train_masked_arm(full_arm, start, ref_model, tokenizer, prompt_stream, device, masks, args.steps, args.group_size, args.lr, args.beta, args.seed, log)
                ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
                print(f"  [{full_arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
                results["starts"][label]["arms"][arm] = {"train_stats": stats, "eval": ev, "mask_counts": counts}
                del model
            del ref_model
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "curvature_constrained_delta_summary.json"; write_json(path, results)
    print(f"[Exp 121 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
