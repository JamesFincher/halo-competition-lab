#!/usr/bin/env python3
"""Experiment 126: Spectral-Sink Calibrated Post-SDPA Microgates.

Implements HYPOTHESES H14. Existing attention heads receive exact-identity
scalar post-SDPA multiplicative gates. Attention-sink mass and output-covariance
spectra select sink-dominated heads; one-shot calibration, selective gate RL,
spectral-regularized RL, random-head RL, and all-head RL are compared on the
validated surgical post-edit initialization.

Research anchors: research/HYPOTHESES.md H14 and arXiv:2611.20403.
"""
from __future__ import annotations

import argparse, copy, json, math, os, random, sys, time, types
from pathlib import Path
import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402

GATE_LAYERS = list(range(24, 30))


def gated_attention_forward(self, hidden_states, position_embeddings=None, attention_mask=None, cache=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    if position_embeddings is None: raise RuntimeError("position_embeddings is required for gated Llama attention")
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if cache is not None: key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)
    attn_output, attn_weights = eager_attention_forward(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )
    gates = self.post_sdpa_gates.to(device=attn_output.device, dtype=attn_output.dtype)
    attn_output = attn_output * gates.view(1, 1, -1, 1)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def install_gates(model, layers=GATE_LAYERS, train_mask=None):
    params = []
    model.config._attn_implementation = "eager"
    for li in layers:
        attn = model.model.layers[li].self_attn
        n_heads = attn.config.num_attention_heads
        if not hasattr(attn, "post_sdpa_gates"):
            attn.post_sdpa_gates = nn.Parameter(torch.ones(n_heads, device=next(attn.parameters()).device, dtype=torch.float32))
        attn.forward = types.MethodType(gated_attention_forward, attn)
        p = attn.post_sdpa_gates
        p.requires_grad_(True)
        if train_mask is not None:
            mask = train_mask[li].to(device=p.device, dtype=torch.float32)
            p.register_hook(lambda g, mask=mask: g * mask)
        params.append((li, p))
    return params


def measurement_texts(tokenizer, train_rows, smoke=False):
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"]
    n = 2 if smoke else 5
    refusal = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'
    malformed = '{"action": "call_tool", "tool": "run_sql", "policy_code":'
    items = []
    for row in benign[:n]:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        items.append(("benign_correct_allow", prompt + row["assistant_text"]))
        items.append(("benign_false_refusal_counterfactual", prompt + refusal))
        items.append(("malformed_counterfactual", prompt + malformed))
        items.append(("prompt_only_control", prompt))
    for row in unsafe[:n]:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        items.append(("unsafe_true_refusal", prompt + row["assistant_text"]))
    return items


@torch.no_grad()
def measure_attention(model, tokenizer, items, device, layers=GATE_LAYERS):
    handles = []
    captured = {}
    n_heads = model.config.num_attention_heads; head_dim = model.config.hidden_size // n_heads
    for li in layers:
        def prehook(_module, args, li=li):
            x = args[0].detach().float().cpu()
            captured.setdefault(li, []).append(x.reshape(x.shape[0], x.shape[1], n_heads, head_dim))
        handles.append(model.model.layers[li].self_attn.o_proj.register_forward_pre_hook(prehook))
    try:
        stats = {}
        for label, text in items:
            captured.clear()
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=320).to(device)
            out = model(**enc, output_attentions=True, use_cache=False)
            for li in layers:
                aw = out.attentions[li][0].float().cpu()
                seq_len = aw.shape[-1]
                tail = max(1, seq_len // 2)
                sink = aw[:, -tail:, 0].mean(dim=-1)
                x = captured[li][0][0, -min(48, captured[li][0].shape[1]):]
                for h in range(n_heads):
                    key = (li, h)
                    ent = stats.setdefault(key, {"sink_mass_sum": 0.0, "count": 0, "samples": [], "labels": {}})
                    ent["sink_mass_sum"] += float(sink[h].mean().item()); ent["count"] += 1
                    ent["samples"].append(x[:, h, :])
                    ent["labels"].setdefault(label, {"sink_mass_sum": 0.0, "count": 0})
                    ent["labels"][label]["sink_mass_sum"] += float(sink[h].mean().item()); ent["labels"][label]["count"] += 1
        records = []
        for (li, h), ent in stats.items():
            x = torch.cat(ent["samples"], dim=0)
            x = x - x.mean(0, keepdim=True)
            cov = x.T @ x / max(x.shape[0], 1)
            eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
            frac = eig / (eig.sum() + 1e-12)
            entropy = -(frac * torch.log(frac + 1e-12)).sum()
            effrank = float(torch.exp(entropy).item())
            top1 = float(frac[0].item())
            sink_mass = ent["sink_mass_sum"] / max(ent["count"], 1)
            rec = {"layer": li, "head": h, "sink_mass": round(sink_mass, 6), "covariance_top1_fraction": round(top1, 6), "effective_rank": round(effrank, 4), "spectral_score": round(sink_mass * top1, 8), "class_sink_mass": {k: round(v["sink_mass_sum"] / max(v["count"], 1), 6) for k, v in ent["labels"].items()}}
            records.append(rec)
        return records
    finally:
        for h in handles: h.remove()


def make_masks(records, seed):
    ordered = sorted(records, key=lambda r: r["spectral_score"], reverse=True)
    count = max(1, math.ceil(len(ordered) / 3))
    sink_keys = {(r["layer"], r["head"]) for r in ordered[:count]}
    rng = random.Random(seed)
    keys = [(r["layer"], r["head"]) for r in records]
    random_keys = set(rng.sample(keys, count))
    n_heads = max(r["head"] for r in records) + 1
    def build(selected):
        return {li: torch.tensor([1.0 if (li, h) in selected else 0.0 for h in range(n_heads)]) for li in GATE_LAYERS}
    return build(sink_keys), build(random_keys), build(set(keys)), sink_keys


def spectral_tensors(model, records, layers=GATE_LAYERS):
    lookup = {(r["layer"], r["head"]): float(r["spectral_score"]) for r in records}
    vals = []
    for li in layers:
        attn = model.model.layers[li].self_attn
        vals.append(torch.tensor([lookup[(li, h)] for h in range(attn.config.num_attention_heads)], device=attn.post_sdpa_gates.device))
    return vals


def apply_one_shot_calibration(model, records, mask, beta=0.35, floor=0.75):
    lookup = {(r["layer"], r["head"]): float(r["spectral_score"]) for r in records}
    vals = [v for v in lookup.values()]
    scale = max(vals) + 1e-12
    with torch.no_grad():
        for li in GATE_LAYERS:
            attn = model.model.layers[li].self_attn
            for h in range(attn.config.num_attention_heads):
                if float(mask[li][h]) > 0:
                    z = lookup[(li, h)] / scale
                    attn.post_sdpa_gates[h] = max(floor, 1.0 - beta * z)


def gate_summary(model):
    vals = []
    for li in GATE_LAYERS:
        g = model.model.layers[li].self_attn.post_sdpa_gates.detach().float().cpu()
        vals.append(g)
    g = torch.cat(vals)
    return {"mean": round(float(g.mean()), 5), "min": round(float(g.min()), 5), "max": round(float(g.max()), 5), "mean_abs_deviation_from_identity": round(float((g - 1).abs().mean()), 6)}


def train_gate_arm(arm_name, init_model, ref_model, tokenizer, prompt_stream, device, mask, spectral_records, steps, group_size, lr, beta, spectral_lambda, seed, log):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    for p in model.parameters(): p.requires_grad_(False)
    gate_params = install_gates(model, GATE_LAYERS, mask)
    params = []
    for _li, p in gate_params:
        p.requires_grad_(True); params.append(p)
    spec_vals = spectral_tensors(model, spectral_records)
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
        grad_norm = loss_val = penalty_val = 0.0
        if losses:
            loss = torch.stack(losses).sum()
            if spectral_lambda:
                pens = []
                for (li, p), sv in zip(gate_params, spec_vals):
                    active = (sv > 0) & (mask[li].to(device=p.device) > 0)
                    if int(active.sum()) == 0: continue
                    pens.append((sv[active].to(dtype=p.dtype) * p[active].pow(2)).sum() / int(active.sum()))
                penalty = torch.stack(pens).mean(); penalty_val = float(penalty.detach().item())
                loss = loss + spectral_lambda * penalty
            loss.backward(); grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item()); opt.step(); loss_val = float(loss.detach().item())
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "spectral_penalty": round(penalty_val, 6), "grad_norm": round(grad_norm, 5), **{f"gate_{k}": v for k, v in gate_summary(model).items()}}
        trace.append(rec); log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | gate_dev={rec['gate_mean_abs_deviation_from_identity']:.4f} | kl={rec['mean_abs_kl']:.4f}", flush=True)
    stats = {"trainable_gate_params": sum(p.numel() for p in params), "train_wall_sec": round(time.perf_counter() - t0, 2), "trace": trace}
    model.eval(); return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--spectral-lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); smoke = os.environ.get("HALO_SMOKE_TEST") == "1"
    out_dir = ROOT / "experiments/exp126_spectral_sink_microgates/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 126] Spectral-Sink Post-SDPA Microgates on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    install_gates(edited)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    items = measurement_texts(tokenizer, train_rows, smoke)
    print("[Stage 1] Measuring sink mass, covariance spectra, and behavior-class profiles...", flush=True)
    base_records = measure_attention(edited, tokenizer, items, device)
    sink_mask, random_mask, all_mask, sink_keys = make_masks(base_records, args.seed)
    mean_effrank = sum(r["effective_rank"] for r in base_records) / len(base_records)
    print(f"  heads={len(base_records)} mean effective_rank={mean_effrank:.2f}; selected sink heads={len(sink_keys)}", flush=True)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [post-edit no-gate] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}% benign_fr={edit_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp126_spectral_sink_microgates", "research_source": "research/HYPOTHESES.md H14", "paper_inspiration": ["arXiv:2611.20403"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "measurement_protocol": "teacher-forced correct rows plus synthetic false-refusal/malformed counterfactuals and prompt-only controls; no policy_eval fitting", "baseline_attention_records": base_records, "baseline_mean_effective_rank": round(mean_effrank, 5), "selected_sink_heads": sorted([list(x) for x in sink_keys]), "post_edit_no_gate": edit_eval, "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        print("[Run] one_shot_spectral_calibration", flush=True)
        model = copy.deepcopy(edited).to(device); install_gates(model)
        apply_one_shot_calibration(model, base_records, sink_mask)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        post_records = measure_attention(model, tokenizer, items[:4] if smoke else items[:8], device)
        results["arms"]["one_shot_spectral_calibration"] = {"eval": ev, "gate_summary": gate_summary(model), "post_attention_records": post_records, "train_stats": {"train_wall_sec": 0.0, "trace": []}}
        print(f"  [one_shot] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        del model
        arms = [
            ("learned_sink_head_gates", sink_mask, 0.0),
            ("learned_sink_head_gates_spectral_regularized", sink_mask, args.spectral_lambda),
            ("learned_random_head_gates", random_mask, 0.0),
            ("learned_all_head_gates", all_mask, 0.0),
        ]
        if smoke: arms = arms[:2]
        for arm, mask, lam in arms:
            print(f"[Run] {arm}", flush=True)
            model, stats = train_gate_arm(arm, edited, ref_model, tokenizer, prompt_stream, device, mask, base_records, args.steps, args.group_size, args.lr, args.beta, lam, args.seed, log)
            ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
            post_records = measure_attention(model, tokenizer, items[:4] if smoke else items[:8], device)
            results["arms"][arm] = {"train_stats": stats, "eval": ev, "gate_summary": gate_summary(model), "post_attention_records": post_records}
            print(f"  [{arm}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            del model
        if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "spectral_sink_microgates_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 126 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
