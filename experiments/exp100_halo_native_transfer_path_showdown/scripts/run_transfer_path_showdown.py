#!/usr/bin/env python3
"""Experiment 100: Halo-Native Post-Edit Transfer-Path Showdown.

Directly compares the two SUBAGENT4 transfer paths under identical post-edit
initialization, prompts, seeds, exact-verifier rewards, explicit KL anchoring,
and weight_decay=0: targeted LoRA versus partial full fine-tuning, with a
vector-only lower-bound reference. Complement layers exclude the surgical layers.

Research anchors: research/SUBAGENT4_HALO_NATIVE_RL.md, arXiv:2403.10704,
arXiv:2609.18587.
"""
from __future__ import annotations

import argparse, copy, json, random, re, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp57_halo_native_targeted_lora/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
from serialization_utils import dumps_safe, write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_halo_native_targeted_lora as halo_lora  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402


def unfreeze_complement(model, layers, mode):
    for p in model.parameters(): p.requires_grad_(False)
    layer_alt = "|".join(str(x) for x in layers)
    if mode == "mlp": pattern = re.compile(rf"\.layers\.({layer_alt})\.mlp\.(gate|up|down)_proj\.")
    elif mode == "attention": pattern = re.compile(rf"\.layers\.({layer_alt})\.self_attn\.(q|k|v|o)_proj\.")
    else: pattern = re.compile(rf"\.layers\.({layer_alt})\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.")
    selected = []
    for name, p in model.named_parameters():
        if pattern.search(name):
            p.requires_grad_(True); selected.append((name, p))
    if not selected: raise RuntimeError(f"no trainable complement parameters matched mode={mode}")
    return selected


def train_fullft_arm(arm_name, init_model, ref_model, tokenizer, prompt_stream, device, layers, mode, steps, group_size, lr, beta, seed, log):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    selected = unfreeze_complement(model, layers, mode)
    params = [p for _n, p in selected]
    initial = {n: p.detach().clone() for n, p in selected}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    trace = []; t0 = time.perf_counter()
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
        drift = math_sqrt(sum(float(((p.detach() - initial[n]).float().norm() ** 2).item()) for n, p in selected))
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5), "complement_weight_drift_l2": round(drift, 5)}
        trace.append(rec); log.write(dumps_safe(rec) + "\n"); log.flush()
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | kl={rec['mean_abs_kl']:.4f} | drift={drift:.3f}", flush=True)
    stats = {"mode": mode, "layers": layers, "trainable_params": sum(p.numel() for p in params), "trainable_tensor_names": [n for n, _p in selected], "final_weight_drift_l2": trace[-1]["complement_weight_drift_l2"] if trace else 0.0, "train_wall_sec": round(time.perf_counter() - t0, 2), "trace": trace}
    model.eval(); return model, stats


def math_sqrt(x): return float(x) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--fullft-lr", type=float, default=3e-5)
    ap.add_argument("--vector-lr", type=float, default=2e-2)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp100_halo_native_transfer_path_showdown/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 100] Halo-Native Transfer-Path Showdown on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [post-edit anchor] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}% benign_fr={edit_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp100_halo_native_transfer_path_showdown", "research_source": "research/SUBAGENT4_HALO_NATIVE_RL.md transfer-path comparison", "paper_inspiration": ["arXiv:2403.10704", "arXiv:2609.18587"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "complement_layers": halo_lora.COMPLEMENT_LAYERS, "baseline_sft": sft_eval, "post_edit_anchor": edit_eval, "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        print("[Run] vector_only_disjoint_lower_bound", flush=True)
        model, stats = vec_rl.train_vector_arm("vector_only_disjoint_lower_bound", edited, ref_model, tokenizer, prompt_stream, device, vec_rl.DISJOINT_LAYERS, args.steps, args.group_size, args.vector_lr, args.beta, args.seed)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        results["arms"]["vector_only_disjoint_lower_bound"] = {"train_stats": stats, "eval": ev}
        print(f"  [vector_only] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        del model
        for arm, targets in [("targeted_lora_complement_allproj", halo_lora.layer_regex(halo_lora.COMPLEMENT_LAYERS, mlp_only=False)), ("targeted_lora_complement_mlp_only", halo_lora.layer_regex(halo_lora.COMPLEMENT_LAYERS, mlp_only=True))]:
            print(f"[Run] {arm}", flush=True)
            model, stats = base_rl.run_grpo_arm(arm, edited, ref_model, tokenizer, prompt_stream, device, targets, 16, args.steps, args.group_size, args.lora_lr, args.beta, log, args.seed)
            ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
            results["arms"][arm] = {"train_stats": stats, "eval": ev, "target_regex": targets}
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            del model
        for arm, mode in [("partial_fullft_complement_allproj", "all"), ("partial_fullft_complement_mlp_only", "mlp")]:
            print(f"[Run] {arm}", flush=True)
            model, stats = train_fullft_arm(arm, edited, ref_model, tokenizer, prompt_stream, device, halo_lora.COMPLEMENT_LAYERS, mode, args.steps, args.group_size, args.fullft_lr, args.beta, args.seed, log)
            ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
            results["arms"][arm] = {"train_stats": stats, "eval": ev}
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            del model
        if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "transfer_path_showdown_summary.json"; write_json(path, results)
    print(f"[Exp 100 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
