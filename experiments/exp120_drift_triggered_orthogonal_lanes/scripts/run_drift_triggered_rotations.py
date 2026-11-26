#!/usr/bin/env python3
"""Experiment 120: Drift-Triggered Orthogonal Rotation Lanes.

Implements HYPOTHESES H7. Output-side block-skew lanes rotate complementary
projection matrices as W_eff = R0(I+S)W. S is skew-symmetric and trainable; when
its spectral drift crosses a threshold, the accumulated rotation is polar-restored
into R0 and S resets exactly to zero. The design preserves exact identity at
initialization while measuring restoration timing and singular-value collateral.

Research anchors: research/HYPOTHESES.md H7 and arXiv:2611.33589.
"""
from __future__ import annotations

import argparse, copy, json, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

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

COMPLEMENT_LAYERS = [24, 25, 26, 27]


class BlockSkewRotationLinear(nn.Module):
    def __init__(self, base: nn.Linear, block_size: int = 64, init_std: float = 0.0):
        super().__init__()
        if base.out_features % block_size != 0: raise ValueError(f"out_features {base.out_features} not divisible by {block_size}")
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.block_size = block_size; self.nblocks = base.out_features // block_size
        raw = torch.zeros(self.nblocks, block_size, block_size, device=base.weight.device, dtype=torch.float32)
        if init_std > 0: raw += torch.randn_like(raw) * init_std
        self.raw_skew_blocks = nn.Parameter(raw)
        self.register_buffer("restored_rotation", torch.eye(base.out_features, device=base.weight.device, dtype=torch.float32))

    def small_rotation(self):
        tri = torch.tril(self.raw_skew_blocks, diagonal=-1)
        skew = tri - tri.transpose(-1, -2)
        eye = torch.eye(self.block_size, device=skew.device, dtype=skew.dtype).expand(self.nblocks, -1, -1)
        return torch.block_diag(*torch.unbind(eye + skew))

    def rotation(self):
        return self.restored_rotation @ self.small_rotation()

    def effective_weight(self):
        return self.rotation() @ self.base.weight.float()

    def forward(self, x):
        return F.linear(x, self.effective_weight().to(dtype=x.dtype), self.base.bias)

    def skew_spectral_norm(self):
        return float(torch.linalg.matrix_norm(self.small_rotation() - torch.eye(self.base.out_features, device=self.restored_rotation.device, dtype=torch.float32), ord=2).item())

    @torch.no_grad()
    def restore_if_needed(self, tau: float):
        norm = self.skew_spectral_norm()
        if norm < tau: return False, norm
        accumulated = self.restored_rotation @ self.small_rotation()
        u, _s, vh = torch.linalg.svd(accumulated, full_matrices=False)
        self.restored_rotation.copy_(u @ vh)
        self.raw_skew_blocks.zero_()
        return True, norm


def install_rotation_lanes(model, layers, scope, block_size, init_std):
    registry = []
    for li in layers:
        layer = model.model.layers[li]
        pairs = []
        if scope in {"attention", "all"}: pairs += [(layer.self_attn, x) for x in ["q_proj", "k_proj", "v_proj", "o_proj"]]
        if scope in {"mlp", "all"}: pairs += [(layer.mlp, x) for x in ["gate_proj", "up_proj", "down_proj"]]
        for parent, attr in pairs:
            lane = BlockSkewRotationLinear(getattr(parent, attr), block_size=block_size, init_std=init_std)
            setattr(parent, attr, lane)
            registry.append({"layer": li, "attr": attr, "lane": lane})
    return registry


def independent_skew_params(registry):
    total = 0
    for rec in registry:
        b = rec["lane"].block_size; n = rec["lane"].nblocks
        total += n * b * (b - 1) // 2
    return total


def train_rotation_arm(arm_name, edited, ref_model, tokenizer, prompt_stream, device, scope, tau, init_std, steps, group_size, lr, beta, seed, log):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(edited).to(device); vec_rl.freeze_weights(model)
    registry = install_rotation_lanes(model, COMPLEMENT_LAYERS, scope, block_size=64, init_std=init_std)
    params = [rec["lane"].raw_skew_blocks for rec in registry]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    trace = []; t0 = time.perf_counter(); total_restorations = 0
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
        grad_norm = loss_val = 0.0; restored_now = 0; max_skew_norm = 0.0
        if losses:
            loss = torch.stack(losses).sum(); loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item()); opt.step(); loss_val = float(loss.detach().item())
            for rec in registry:
                did, norm = rec["lane"].restore_if_needed(tau)
                if did and rec["lane"].raw_skew_blocks in opt.state: del opt.state[rec["lane"].raw_skew_blocks]
                restored_now += int(did); max_skew_norm = max(max_skew_norm, norm)
        total_restorations += restored_now
        rec_out = {"arm": arm_name, "step": step, "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5), "restorations": restored_now, "max_postupdate_skew_spectral_norm": round(max_skew_norm, 5)}
        trace.append(rec_out); log.write(dumps_safe(rec_out) + "\n"); log.flush()
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec_out['mean_reward']:.3f} | active={len(groups)}/2 dead={rec_out['dead_groups']} | restore={restored_now} | kl={rec_out['mean_abs_kl']:.4f}", flush=True)
    with torch.no_grad():
        sv_drifts = []
        for rec in registry:
            base_sv = torch.linalg.svdvals(rec["lane"].base.weight.float())
            new_sv = torch.linalg.svdvals(rec["lane"].effective_weight())
            sv_drifts.append(float((new_sv - base_sv).abs().mean().item()))
    stats = {"scope": scope, "tau": tau, "init_std": init_std, "n_lanes": len(registry), "raw_lane_params": sum(p.numel() for p in params), "independent_skew_params": independent_skew_params(registry), "total_restorations": total_restorations, "mean_singular_value_absolute_drift": round(sum(sv_drifts) / max(len(sv_drifts), 1), 6), "train_wall_sec": round(time.perf_counter() - t0, 2), "trace": trace}
    model.eval(); return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp120_drift_triggered_orthogonal_lanes/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 120] Drift-Triggered Orthogonal Rotation Lanes on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [post-edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}% benign_fr={edit_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp120_drift_triggered_orthogonal_lanes", "research_source": "research/HYPOTHESES.md H7", "paper_inspiration": ["arXiv:2611.33589"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "complement_layers": COMPLEMENT_LAYERS, "baseline_sft": sft_eval, "post_edit_anchor": edit_eval, "arms": {}}
    arms = [
        ("rotation_all_trigger_tau0.10", "all", 0.10, 0.0),
        ("rotation_all_no_restore", "all", 999.0, 0.0),
        ("rotation_all_restore_every_step", "all", 0.0, 0.0),
        ("rotation_mlp_trigger_tau0.10", "mlp", 0.10, 0.0),
        ("rotation_random_init_no_restore", "all", 999.0, 0.01),
    ]
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm, scope, tau, init_std in arms:
            print(f"[Run] {arm}", flush=True)
            model, stats = train_rotation_arm(arm, edited, ref_model, tokenizer, prompt_stream, device, scope, tau, init_std, args.steps, args.group_size, args.lr, args.beta, args.seed, log)
            ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% restorations={stats['total_restorations']}", flush=True)
            results["arms"][arm] = {"train_stats": stats, "eval": ev}
            del model
        print("[Run] matched_targeted_lora_control", flush=True)
        targets = halo_lora.layer_regex(COMPLEMENT_LAYERS, mlp_only=False)
        model, stats = base_rl.run_grpo_arm("matched_targeted_lora_control", edited, ref_model, tokenizer, prompt_stream, device, targets, 16, args.steps, args.group_size, args.lora_lr, args.beta, log, args.seed)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        print(f"  [matched_targeted_lora_control] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results["arms"]["matched_targeted_lora_control"] = {"train_stats": stats, "eval": ev, "target_regex": targets}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "drift_triggered_rotations_summary.json"; write_json(path, results)
    print(f"[Exp 103 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
