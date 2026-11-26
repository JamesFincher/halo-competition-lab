#!/usr/bin/env python3
"""Experiment 53: Dual-Sided Internal Advantage Sculpting.

Implements SUBAGENT1 rank-4: shape RL advantages with two internal directions,
a positive correct-routing/compliance direction and a negative false-over-refusal
direction orthogonalized against true safety refusal. The external verifier
remains dominant; internal scores only rescue promising wrong rollouts and
discount internally over-defensive behavior before group normalization.

Research anchors: arXiv:2610.40998, arXiv:2604.01476, arXiv:2611.30627.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402

PROBE_LAYER = 29


def action_pos(ids: torch.Tensor, prompt_len: int, action_ids: set[int]) -> int | None:
    seq = ids.tolist()
    for pos in range(prompt_len, len(seq)):
        if seq[pos] in action_ids:
            return pos
    return None


def capture_hidden(model: torch.nn.Module, ids: torch.Tensor, layer: int) -> torch.Tensor:
    cap = {}

    def hook(_module, _inp, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
    return cap["h"][0]


def hidden_at_anchor(model, tokenizer, prompt: str, completion: str, device, action_ids: set[int]) -> torch.Tensor | None:
    full = prompt + completion
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=320).to(device)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=256)["input_ids"])
    pos = action_pos(enc["input_ids"][0], prompt_len, action_ids)
    if pos is None or pos == 0:
        return None
    h = capture_hidden(model, enc["input_ids"], PROBE_LAYER)
    return h[pos - 1].detach().float()


def hidden_for_rollout(model, seq: torch.Tensor, prompt_len: int, action_ids: set[int]) -> torch.Tensor | None:
    pos = action_pos(seq, prompt_len, action_ids)
    if pos is None or pos == 0:
        return None
    h = capture_hidden(model, seq.unsqueeze(0), PROBE_LAYER)
    return h[pos - 1].detach().float()


def mean_stack(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs).mean(dim=0)


def build_directions(model, tokenizer, train_rows, device, action_ids) -> dict:
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"][:8]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"][:8]
    false_refusal_text = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'
    unsafe_pass_text = '{"action": "call_tool", "tool": "read_file", "policy_code": "ALLOW", "reason": "Called read_file."}'

    groups = {"correct": [], "incorrect": [], "false_refusal": [], "benign_allow": [], "true_refusal": [], "unsafe_pass": []}
    for row in benign + unsafe:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        h = hidden_at_anchor(model, tokenizer, prompt, row["assistant_text"], device, action_ids)
        if h is not None:
            groups["correct"].append(h)
            if row["target"]["action"] == "call_tool":
                groups["benign_allow"].append(h)
            else:
                groups["true_refusal"].append(h)
    for row in benign:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        h = hidden_at_anchor(model, tokenizer, prompt, false_refusal_text, device, action_ids)
        if h is not None:
            groups["false_refusal"].append(h)
            groups["incorrect"].append(h)
    for row in unsafe:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        h = hidden_at_anchor(model, tokenizer, prompt, unsafe_pass_text, device, action_ids)
        if h is not None:
            groups["unsafe_pass"].append(h)
            groups["incorrect"].append(h)

    v_comp = mean_stack(groups["correct"]) - mean_stack(groups["incorrect"])
    v_comp = v_comp / (v_comp.norm() + 1e-8)
    v_over = mean_stack(groups["false_refusal"]) - mean_stack(groups["benign_allow"])
    v_true = mean_stack(groups["true_refusal"]) - mean_stack(groups["unsafe_pass"])
    v_true = v_true / (v_true.norm() + 1e-8)
    v_over = v_over - torch.dot(v_over, v_true) * v_true
    v_over = v_over / (v_over.norm() + 1e-8)

    comp_pool = torch.stack([x @ v_comp for xs in (groups["correct"], groups["incorrect"]) for x in xs])
    over_pool = torch.stack([x @ v_over for xs in (groups["false_refusal"], groups["benign_allow"]) for x in xs])
    return {
        "v_compliance": v_comp,
        "v_overrefusal": v_over,
        "v_true_unsafe": v_true,
        "comp_mean": float(comp_pool.mean().item()),
        "comp_std": float(comp_pool.std(unbiased=False).item()) + 1e-6,
        "over_mean": float(over_pool.mean().item()),
        "over_std": float(over_pool.std(unbiased=False).item()) + 1e-6,
        "group_counts": {k: len(v) for k, v in groups.items()},
        "cos_over_true": float(torch.dot(v_over, v_true).item()),
    }


def internal_scores(hidden: torch.Tensor | None, dirs: dict) -> tuple[float, float]:
    if hidden is None:
        return 0.0, 0.0
    zc = float(((hidden @ dirs["v_compliance"]) - dirs["comp_mean"]) / dirs["comp_std"])
    zo = float(((hidden @ dirs["v_overrefusal"]) - dirs["over_mean"]) / dirs["over_std"])
    return zc, zo


def train_sculpted_arm(arm_name, init_model, ref_model, tokenizer, prompt_stream, device, dirs, lam_plus, lam_minus, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vec_rl.DISJOINT_LAYERS, device)
    opt = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[0] for x in ("call", "ref"))
    trace = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval()
        groups = []
        raw_rewards = []
        shaped_rewards = []
        dead_groups = 0
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
            items = []
            for seq in gens:
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                reward, info = base_rl.score_completion(text, row["target"])
                hidden = hidden_for_rollout(model, seq, prompt_len, action_ids)
                zc, zo = internal_scores(hidden, dirs)
                shaped = float(reward)
                if reward < 1.0:
                    shaped += lam_plus * zc
                shaped -= lam_minus * max(0.0, zo)
                items.append({"seq": seq, "reward": float(reward), "shaped": shaped, "zc": zc, "zo": zo, "exact": info["exact"]})
                raw_rewards.append(float(reward))
                shaped_rewards.append(shaped)
            r_t = torch.tensor([x["shaped"] for x in items], dtype=torch.float32, device=device)
            if float(r_t.std(unbiased=False).item()) < 1e-6:
                dead_groups += 1
                continue
            adv = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)
            for item, a in zip(items, adv):
                item["adv"] = a.detach()
            groups.append({"items": items, "prompt_len": prompt_len})

        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        kls = []
        for group in groups:
            for item in group["items"]:
                ids = item["seq"].unsqueeze(0)
                lp = base_rl.completion_logprobs(model, ids, group["prompt_len"])
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref_model, ids, group["prompt_len"])
                kl = (lp - lp_ref).abs().mean()
                kls.append(float(kl.detach().item()))
                losses.append(-item["adv"] * lp + beta * kl)
        loss_val = grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
            opt.step()
            loss_val = float(loss.detach().item())

        mean_zc = sum(x["zc"] for g in groups for x in g["items"]) / max(sum(len(g["items"]) for g in groups), 1)
        mean_zo = sum(x["zo"] for g in groups for x in g["items"]) / max(sum(len(g["items"]) for g in groups), 1)
        rec = {
            "arm": arm_name,
            "step": step,
            "mean_verifier_reward": round(sum(raw_rewards) / max(len(raw_rewards), 1), 4),
            "mean_shaped_reward": round(sum(shaped_rewards) / max(len(shaped_rewards), 1), 4),
            "active_groups": len(groups),
            "dead_groups": dead_groups,
            "mean_z_compliance": round(mean_zc, 4),
            "mean_z_overrefusal": round(mean_zo, 4),
            "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0,
            "loss": round(loss_val, 5),
            "grad_norm": round(grad_norm, 5),
        }
        trace.append(rec)
        print(f"  [{arm_name}] step {step:02d}/{steps} | raw={rec['mean_verifier_reward']:.3f} shaped={rec['mean_shaped_reward']:.3f} | active={len(groups)}/2 dead={dead_groups} | zc={mean_zc:+.2f} zo={mean_zo:+.2f}", flush=True)

    stats = {"lambda_plus": lam_plus, "lambda_minus": lam_minus, "trace": trace, "train_wall_sec": round(time.perf_counter() - t0, 2), "n_vector_params": sum(v.numel() for v in vectors.values())}
    model.eval()
    return model, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp53_dual_sided_advantage_sculpting/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 53] Dual-Sided Internal Advantage Sculpting on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, dtype)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    selected = edit_info["selected_neurons"]
    ref_model = copy.deepcopy(edited).to(device).eval()
    vec_rl.freeze_weights(ref_model)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [validated edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)

    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[0] for x in ("call", "ref"))
    print("[Stage 1] Building compliance and orthogonalized over-refusal directions...", flush=True)
    dirs = build_directions(edited, tokenizer, train_rows, device, action_ids)
    print(f"  direction groups={dirs['group_counts']} cos(over,true)={dirs['cos_over_true']:.4f}", flush=True)

    arms = [
        ("verifier_only_control", 0.0, 0.0),
        ("balanced_0.1_0.1", 0.1, 0.1),
        ("safetyheavy_0.1_0.3", 0.1, 0.3),
        ("complianceheavy_0.3_0.1", 0.3, 0.1),
    ]
    results = {
        "experiment": "exp53_dual_sided_advantage_sculpting",
        "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 4",
        "paper_inspiration": ["arXiv:2610.40998", "arXiv:2604.01476", "arXiv:2611.30627"],
        "config": vars(args),
        "device": str(device),
        "selected_neurons": selected,
        "validated_edit_operator": edit_info,
        "baseline_sft": sft_eval,
        "anchored_edit_only": edit_eval,
        "directions": {
            "comp_mean": dirs["comp_mean"],
            "comp_std": dirs["comp_std"],
            "over_mean": dirs["over_mean"],
            "over_std": dirs["over_std"],
            "cos_over_true": dirs["cos_over_true"],
            "v_compliance_first8": [round(x, 6) for x in dirs["v_compliance"][:8].detach().cpu().tolist()],
            "v_overrefusal_first8": [round(x, 6) for x in dirs["v_overrefusal"][:8].detach().cpu().tolist()],
        },
        "direction_group_counts": dirs["group_counts"],
        "arms": {},
    }

    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm_name, lp, lm in arms:
            print(f"[Run] {arm_name}: lambda+= {lp}, lambda-= {lm}", flush=True)
            trained, stats = train_sculpted_arm(arm_name, edited, ref_model, tokenizer, prompt_stream, device, dirs, lp, lm, args.steps, args.group_size, args.lr, args.beta, args.seed)
            for rec in stats["trace"]:
                log.write(json.dumps(rec) + "\n")
            log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm_name] = {"train_stats": stats, "eval": ev}
            del trained
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "dual_sided_advantage_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 53 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
