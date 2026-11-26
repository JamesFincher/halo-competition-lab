#!/usr/bin/env python3
"""Experiment 52: PGS-Credit Direction Discovery for Surgical Edits.

Implements SUBAGENT1 rank-3: estimate return-weighted policy-gradient steering
directions v_l = E[A_t grad_hl log pi(a_t|o_t)] at the action-decision token,
then causally compare those directions with DLA/unembed, contrastive, and random
directions in the existing rank-1 surgical edit harness.

Research anchors: arXiv:2607.27574 (Policy Gradient Steering) and
arXiv:2509.06608 (Small Vectors, Big Effects).
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
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402

PROBE_LAYERS = [20, 28, 29]


def first_action_pos(ids: torch.Tensor, prompt_len: int, action_ids: set[int]) -> int | None:
    seq = ids.tolist()
    for pos in range(prompt_len, len(seq)):
        if seq[pos] in action_ids:
            return pos
    return None


@torch.no_grad()
def sample_rollouts(model, tokenizer, row, device, group_size: int, seed: int):
    torch.manual_seed(seed)
    prompt = base_rl.build_prompt(tokenizer, row["messages"])
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    prompt_len = enc["input_ids"].shape[1]
    gens = model.generate(
        **enc,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.eos_token_id,
    )
    items = []
    for seq in gens:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        reward, info = base_rl.score_completion(text, row["target"])
        items.append({"seq": seq, "prompt_len": prompt_len, "reward": float(reward), "exact": bool(info["exact"]), "text": text})
    return items


def choose_boundary_prompts(model, tokenizer, train_rows, device, seed: int, per_class: int) -> list[dict]:
    chosen = []
    for action, count in (("call_tool", per_class), ("refuse", per_class)):
        rows = [r for r in train_rows if r["target"]["action"] == action][:10]
        scored = []
        for i, row in enumerate(rows):
            items = sample_rollouts(model, tokenizer, row, device, 2, seed + i + (100 if action == "refuse" else 0))
            rewards = [x["reward"] for x in items]
            mean = sum(rewards) / len(rewards)
            var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
            hard_score = var + 0.05 * (1.0 - abs(mean - 0.6))
            scored.append((hard_score, row, round(mean, 3), round(var, 4)))
        scored.sort(key=lambda x: -x[0])
        chosen.extend([x[1] for x in scored[:count]])
        print(f"  [boundary select:{action}] " + ", ".join(f"m={x[2]:.2f}/v={x[3]:.2f}" for x in scored[:count]), flush=True)
    return chosen


def capture_hidden_gradients(model, ids: torch.Tensor, layers: list[int]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    captured = {}
    hooks = []

    def make_hook(layer_id):
        def hook(_module, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden.retain_grad()
            captured[layer_id] = hidden
        return hook

    for layer_id in layers:
        hooks.append(model.model.layers[layer_id].register_forward_hook(make_hook(layer_id)))
    try:
        out = model(input_ids=ids)
    finally:
        for hook in hooks:
            hook.remove()
    return out.logits, captured


def estimate_pgs_directions(model, tokenizer, rows, device, group_size: int, seed: int) -> tuple[dict[int, torch.Tensor], dict, torch.Tensor]:
    # One downstream matrix must require grad so autograd retains activation gradients.
    for param in model.parameters():
        param.requires_grad_(False)
    probe_param = model.model.layers[-1].mlp.down_proj.weight
    probe_param.requires_grad_(True)

    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[:1] for x in ("call", "ref"))
    accum = {li: torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32) for li in PROBE_LAYERS}
    correct_hidden_sum = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32)
    incorrect_hidden_sum = torch.zeros_like(correct_hidden_sum)
    correct_count = incorrect_count = 0
    weight_sum = 0.0
    used = skipped = 0
    prompt_records = []

    for row_i, row in enumerate(rows):
        items = sample_rollouts(model, tokenizer, row, device, group_size, seed + 17 * row_i)
        rewards = torch.tensor([x["reward"] for x in items], dtype=torch.float32, device=device)
        if float(rewards.std(unbiased=False).item()) < 1e-6:
            skipped += len(items)
            prompt_records.append({"row": row_i, "target_action": row["target"]["action"], "mean_reward": round(float(rewards.mean().item()), 3), "used_rollouts": 0})
            continue
        advs = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        n_used = 0
        for item, adv in zip(items, advs):
            pos = first_action_pos(item["seq"], item["prompt_len"], action_ids)
            if pos is None or pos == 0:
                skipped += 1
                continue
            ids = item["seq"].unsqueeze(0)
            logits, captured = capture_hidden_gradients(model, ids, PROBE_LAYERS)
            token_logp = F.log_softmax(logits[0, pos - 1, :].float(), dim=-1)[ids[0, pos]]
            model.zero_grad(set_to_none=True)
            token_logp.backward()
            for li in PROBE_LAYERS:
                grad = captured[li].grad[0, pos - 1, :].detach().float()
                accum[li] += float(adv.item()) * grad
            anchor_hidden_l29 = captured[29][0, pos - 1, :].detach().float()
            if item["exact"]:
                correct_hidden_sum += anchor_hidden_l29
                correct_count += 1
            else:
                incorrect_hidden_sum += anchor_hidden_l29
                incorrect_count += 1
            weight_sum += abs(float(adv.item()))
            used += 1
            n_used += 1
        prompt_records.append({"row": row_i, "target_action": row["target"]["action"], "mean_reward": round(float(rewards.mean().item()), 3), "used_rollouts": n_used})

    denom = max(weight_sum, 1e-6)
    directions = {}
    for li, vec in accum.items():
        v = vec / denom
        directions[li] = (v / (v.norm() + 1e-8)).detach().cpu()
    probe_param.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if correct_count and incorrect_count:
        contrast = correct_hidden_sum / correct_count - incorrect_hidden_sum / incorrect_count
        contrast = (contrast / (contrast.norm() + 1e-8)).detach().cpu()
    else:
        contrast = torch.zeros_like(directions[29])
    meta = {
        "used_rollouts": used,
        "skipped_rollouts": skipped,
        "adv_abs_weight_sum": round(weight_sum, 4),
        "correct_anchor_count": correct_count,
        "incorrect_anchor_count": incorrect_count,
        "prompts": prompt_records,
    }
    return directions, meta, contrast


def apply_direction_edit(model, selected, direction, strength: float):
    rnd_mod.apply_neuron_subset_edit(model, selected, direction, strength=strength)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--strength", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp52_pgs_credit_direction/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 52] PGS-Credit Direction Discovery on {device} ({dtype})", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    sft, tokenizer = vec_rl.load_sft(device, dtype)
    scores, u_dla = rnd_mod.compute_dla_scores(sft, tokenizer, train_rows, device)
    selected = {li: torch.topk(scores[li], k=64).indices.tolist() for li in vec_rl.EDIT_LAYERS}

    edited = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(edited, selected, u_dla, strength=args.strength)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [DLA reference edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}% benign_fr={edit_eval['benign_false_refusal_rate']:.1f}%", flush=True)

    print("[Stage 1] Selecting high-variance boundary prompts from training data...", flush=True)
    boundary_rows = choose_boundary_prompts(edited, tokenizer, train_rows, device, args.seed, args.per_class)
    print("[Stage 2] Estimating return-weighted policy-gradient directions at layers 20/28/29...", flush=True)
    pgs_dirs, pgs_meta, rollout_contrast_dir = estimate_pgs_directions(edited, tokenizer, boundary_rows, device, args.group_size, args.seed)

    rng = torch.Generator().manual_seed(args.seed)
    random_dir = torch.randn(u_dla.shape, generator=rng, dtype=u_dla.dtype, device=u_dla.device)
    random_dir = random_dir / (random_dir.norm() + 1e-8)
    contrast_dir = rollout_contrast_dir

    directions = {
        "dla_unembed_reference": u_dla.cpu(),
        "pgs_l20": pgs_dirs[20],
        "pgs_l28": pgs_dirs[28],
        "pgs_l29": pgs_dirs[29],
        "pgs_minus_dla_residual": contrast_dir,
        "random_control": random_dir.cpu(),
    }
    cosines = {}
    for name, vec in directions.items():
        cosines[name] = {other: round(float(F.cosine_similarity(vec.float(), other_vec.float(), dim=0).item()), 5) for other, other_vec in directions.items() if other != name}

    print("[Stage 3] Causal edit bake-off on identical DLA top-64 neurons...", flush=True)
    arms = {}
    for name, direction in directions.items():
        model = copy.deepcopy(sft)
        apply_direction_edit(model, selected, direction, args.strength)
        ev = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
        arms[name] = {"eval": ev, "direction_l2": round(float(direction.norm().item()), 6), "cosine_to_dla": round(float(F.cosine_similarity(direction.float().cpu(), u_dla.float().cpu(), dim=0).item()), 5)}
        print(f"  [{name:24s}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "experiment": "exp52_pgs_credit_direction",
        "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 3",
        "paper_inspiration": ["arXiv:2607.27574", "arXiv:2509.06608"],
        "config": vars(args),
        "device": str(device),
        "wall_sec": round(time.time() - t0, 2),
        "selected_neurons": selected,
        "dla_reference_eval": edit_eval,
        "pgs_meta": pgs_meta,
        "direction_cosines": cosines,
        "causal_edit_arms": arms,
    }
    path = out_dir / "pgs_direction_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 52 DONE] wrote {path} in {summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
