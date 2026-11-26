#!/usr/bin/env python3
"""Experiment 123: Input-Gated Kronecker-Factored Adaptation Lanes.

Implements HYPOTHESES H6 with exact Kronecker geometry: for a weight whose input
and output dimensions are divisible by p=3, ΔW=A⊗(B1 diag(g(x)) B2), with frozen
random A and trainable B1/B2. A shared input gate quantizes gate values on a
four-step lattice, so effective rank changes in discrete multiples of p. Constant
and frozen-random gate controls isolate input adaptivity.

Research anchors: research/HYPOTHESES.md H6 and arXiv:2611.33236.
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


class SharedGateNetwork(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, rank))
    def forward(self, x): return self.net(x)


class KiRALinear(nn.Module):
    def __init__(self, base, p, rank, mode, levels, shared_gate):
        super().__init__()
        if base.out_features % p or base.in_features % p: raise ValueError(f"bad KiRA dimensions {base.weight.shape} for p={p}")
        self.base = base
        for q in self.base.parameters(): q.requires_grad_(False)
        self.p = p; self.rank = rank; self.mode = mode; self.levels = levels
        self.m = base.out_features // p; self.n = base.in_features // p
        dev = base.weight.device
        a = torch.randn(p, p, device=dev, dtype=torch.float32) / max(p, 1) ** 0.5
        probe = torch.randn(64, base.in_features, device=dev, dtype=torch.float32) / max(base.in_features, 1) ** 0.25
        self.register_buffer("frozen_A", a)
        self.register_buffer("input_probe", probe)
        self.B1 = nn.Parameter(torch.randn(self.m, rank, device=dev, dtype=torch.float32) * 0.02)
        self.B2 = nn.Parameter(torch.zeros(rank, self.n, device=dev, dtype=torch.float32))
        self.constant_gate = nn.Parameter(torch.zeros(rank, device=dev, dtype=torch.float32))
        self.register_buffer("random_gate", torch.rand(rank, device=dev, dtype=torch.float32))
        self.shared_gate = shared_gate
        self.last_gate = torch.zeros(rank, device=dev)

    def quantize(self, soft):
        hard = torch.round(soft * self.levels) / self.levels
        return soft + (hard - soft).detach()

    def gate(self, x):
        if self.mode == "random":
            g = torch.round(self.random_gate * self.levels) / self.levels
        elif self.mode == "constant":
            g = self.quantize(torch.sigmoid(self.constant_gate))
        else:
            flat = x.detach().reshape(-1, x.shape[-1]).mean(0).float()
            z = flat @ self.input_probe.T
            g = self.quantize(torch.sigmoid(self.shared_gate(z)))
        self.last_gate = g.detach().float().cpu()
        return g

    def effective_weight(self, x):
        g = self.gate(x).to(device=self.B1.device, dtype=torch.float32)
        M = self.B1 @ torch.diag(g) @ self.B2
        delta = torch.kron(self.frozen_A, M)
        return self.base.weight.float() + delta

    def forward(self, x):
        return F.linear(x, self.effective_weight(x).to(dtype=x.dtype), self.base.bias)


def install_kira(model, layers, scope, mode, rank, levels):
    shared = SharedGateNetwork(rank).to(device=next(model.parameters()).device)
    registry = []
    for li in layers:
        layer = model.model.layers[li]
        pairs = []
        if scope in {"attention", "all"}: pairs += [(layer.self_attn, x) for x in ["q_proj", "o_proj"]]
        if scope in {"mlp", "all"}: pairs += [(layer.mlp, x) for x in ["up_proj", "down_proj"]]
        for parent, attr in pairs:
            lane = KiRALinear(getattr(parent, attr), p=3, rank=rank, mode=mode, levels=levels, shared_gate=shared)
            setattr(parent, attr, lane)
            registry.append({"layer": li, "attr": attr, "lane": lane})
    return registry, shared


def gate_snapshot(registry):
    if not registry: return {"mean_gate": 0.0, "mean_active_lattice_steps": 0.0, "mean_off_fraction": 0.0}
    g = torch.stack([rec["lane"].last_gate.float().cpu() for rec in registry])
    return {"mean_gate": round(float(g.mean().item()), 5), "mean_active_lattice_steps": round(float((g * 4).mean().item()), 4), "mean_off_fraction": round(float((g < 0.125).float().mean().item()), 5)}


def train_kira_arm(arm_name, edited, ref_model, tokenizer, prompt_stream, device, scope, mode, steps, group_size, lr, beta, seed, rank, levels, log):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(edited).to(device); vec_rl.freeze_weights(model)
    registry, shared = install_kira(model, COMPLEMENT_LAYERS, scope, mode, rank, levels)
    adapter_params = [x for rec in registry for x in (rec["lane"].B1, rec["lane"].B2)]
    if mode == "constant": adapter_params.extend(rec["lane"].constant_gate for rec in registry)
    gate_params = list(shared.parameters()) if mode == "input" else []
    params = adapter_params + gate_params
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
        gs = gate_snapshot(registry)
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5), **gs}
        trace.append(rec); log.write(dumps_safe(rec) + "\n"); log.flush()
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | gate={rec['mean_gate']:.2f} | kl={rec['mean_abs_kl']:.4f}", flush=True)
    stats = {"scope": scope, "mode": mode, "kira_rank": rank, "lattice_levels": levels, "n_lanes": len(registry), "trainable_params": sum(p.numel() for p in params), "adapter_params": sum(p.numel() for p in adapter_params), "shared_gate_params": sum(p.numel() for p in gate_params), "final_gate_stats": gate_snapshot(registry), "train_wall_sec": round(time.perf_counter() - t0, 2), "trace": trace}
    model.eval(); return model, stats


@torch.no_grad()
def activation_adaptivity_diagnostic(model, tokenizer, rows, device):
    benign = [r for r in rows if r["target"]["action"] == "call_tool"][:12]; unsafe = [r for r in rows if r["target"]["action"] == "refuse"][:12]
    feats = []
    labels = []
    for label, group in [("benign_allow", benign), ("true_refusal", unsafe)]:
        for row in group:
            enc = tokenizer(base_rl.build_prompt(tokenizer, row["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
            captured = {}
            h = model.model.layers[24].register_forward_hook(lambda _m, _i, out: captured.setdefault("x", (out[0] if isinstance(out, tuple) else out).detach()))
            try: model(**enc)
            finally: h.remove()
            feats.append(captured["x"][0, -1].float().cpu()); labels.append(label)
    x = torch.stack(feats); x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = x @ x.T
    same = []; cross = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (same if labels[i] == labels[j] else cross).append(float(sim[i, j].item()))
    ms = sum(same) / len(same); mc = sum(cross) / len(cross)
    return {"diagnostic_note": "crude layer-24 final-prompt activation separability proxy; wrong-tool/malformed classes are absent from schema labels", "mean_within_class_cosine": round(ms, 5), "mean_cross_class_cosine": round(mc, 5), "within_minus_cross": round(ms - mc, 5), "predicts_input_adaptivity_useful": bool(ms - mc > 0.02)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp123_input_gated_kronecker_lanes/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 123] Input-Gated Kronecker-Factored Lanes on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    diagnostic = activation_adaptivity_diagnostic(edited, tokenizer, train_rows, device)
    print(f"  [adaptivity diagnostic] within-cross={diagnostic['within_minus_cross']:.3f} useful={diagnostic['predicts_input_adaptivity_useful']} | [post-edit] exact={edit_eval['exact_compliance_rate']:.1f}%", flush=True)
    results = {"experiment": "exp123_input_gated_kronecker_lanes", "research_source": "research/HYPOTHESES.md H6", "paper_inspiration": ["arXiv:2611.33236"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "adaptivity_diagnostic": diagnostic, "baseline_sft": sft_eval, "post_edit_anchor": edit_eval, "arms": {}}
    arms = [("kira_input_hard_all", "all", "input"), ("kira_constant_hard_all", "all", "constant"), ("kira_random_hard_all", "all", "random"), ("kira_input_hard_attention", "attention", "input"), ("kira_input_hard_mlp", "mlp", "input")]
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm, scope, mode in arms:
            print(f"[Run] {arm}", flush=True)
            model, stats = train_kira_arm(arm, edited, ref_model, tokenizer, prompt_stream, device, scope, mode, args.steps, args.group_size, args.lr, args.beta, args.seed, args.rank, args.levels, log)
            ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% params={stats['trainable_params']}", flush=True)
            results["arms"][arm] = {"train_stats": stats, "eval": ev}
            del model
        print("[Run] matched_lora_r16_control", flush=True)
        targets = halo_lora.layer_regex(COMPLEMENT_LAYERS, mlp_only=False)
        model, stats = base_rl.run_grpo_arm("matched_lora_r16_control", edited, ref_model, tokenizer, prompt_stream, device, targets, 16, args.steps, args.group_size, args.lora_lr, args.beta, log, args.seed)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        print(f"  [matched_lora_r16_control] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results["arms"]["matched_lora_r16_control"] = {"train_stats": stats, "eval": ev, "target_regex": targets}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "input_gated_kronecker_summary.json"; write_json(path, results)
    print(f"[Exp 123 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
