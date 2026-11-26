#!/usr/bin/env python3
"""Experiment 51: Decision-Anchor Hidden-Align for Correct Tool Routings.

Converts SUBAGENT1 rank-2 research idea into a runnable micro-benchmark. During
surgically anchored vector RL, exact-compliant rollouts receive an auxiliary
hidden-state alignment objective at the token immediately before the action
value (`call` or `ref`). Only correct rollouts are aligned; false refusals and
unsafe passes are never positive examples.

Research anchors: arXiv:2606.03234 (Right Makes Might / Hidden-Align) and
arXiv:2611.30627 (Attention Process Reward Model).
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402

ALIGN_LAYERS = {"mid_l20": 20, "late_l29": 29}


def policy_forward_with_anchor(
    model: torch.nn.Module,
    ids: torch.Tensor,
    prompt_len: int,
    action_token_ids: set[int],
    align_layer: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int | None]:
    """Return completion mean logprob, differentiable anchor hidden state, and anchor position."""
    captured = {}

    def capture_hook(_module, _inp, out):
        hidden_out = out[0] if isinstance(out, tuple) else out
        captured["hidden"] = hidden_out

    handle = model.model.layers[align_layer].register_forward_hook(capture_hook)
    try:
        out = model(input_ids=ids)
    finally:
        handle.remove()
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp_lp = log_probs[:, max(prompt_len - 1, 0) :].mean()

    seq = ids[0].tolist()
    rel_action = None
    for pos in range(prompt_len, len(seq)):
        if seq[pos] in action_token_ids:
            rel_action = pos
            break
    if rel_action is None or rel_action == 0 or "hidden" not in captured:
        return comp_lp, None, None
    anchor_pos = rel_action - 1
    hidden = captured["hidden"][0, anchor_pos, :]
    return comp_lp, hidden, anchor_pos


def pairwise_alignment_loss(anchors: list[torch.Tensor]) -> tuple[torch.Tensor | None, float]:
    if len(anchors) < 2:
        return None, 0.0
    normed = [F.normalize(a.float(), dim=-1) for a in anchors]
    sims = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            sims.append(torch.dot(normed[i], normed[j]))
    mean_sim = torch.stack(sims).mean()
    return 1.0 - mean_sim, float(mean_sim.detach().item())


def train_hidden_align_arm(
    arm_name: str,
    init_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_stream: list[dict],
    device: torch.device,
    vector_layers: list[int],
    align_layer: int | None,
    align_lambda: float,
    steps: int,
    group_size: int,
    lr: float,
    beta: float,
    seed: int,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vector_layers, device)
    optimizer = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    action_token_ids = set(tokenizer.encode(x, add_special_tokens=False)[:1] for x in ("call", "ref"))

    trace = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        step_t0 = time.perf_counter()
        batch_rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval()
        groups = []
        all_rewards = []
        exact_count = rollout_count = 0
        for row in batch_rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.95,
                    num_return_sequences=group_size,
                    pad_token_id=tokenizer.eos_token_id,
                )
            items = []
            rewards = []
            for seq in generated:
                completion = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                reward, info = base_rl.score_completion(completion, row["target"])
                rewards.append(float(reward))
                exact_count += int(info["exact"])
                rollout_count += 1
                items.append({"seq": seq, "reward": float(reward), "exact": bool(info["exact"])})
            all_rewards.extend(rewards)
            r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            if float(r_t.std(unbiased=False).item()) < 1e-6:
                continue
            adv = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)
            for item, advantage in zip(items, adv):
                item["adv"] = advantage.detach()
            groups.append({"items": items, "prompt_len": prompt_len})

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        kls = []
        align_vals = []
        aligned_pairs = 0
        missing_anchor = 0
        for group in groups:
            correct_anchors = []
            for item in group["items"]:
                ids = item["seq"].unsqueeze(0)
                lp_cur, anchor_hidden, anchor_pos = policy_forward_with_anchor(
                    model, ids, group["prompt_len"], action_token_ids, align_layer or 29
                )
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref_model, ids, group["prompt_len"])
                kl = (lp_cur - lp_ref).abs().mean()
                kls.append(float(kl.detach().item()))
                losses.append(-item["adv"] * lp_cur + beta * kl)
                if align_layer is not None and item["exact"]:
                    if anchor_hidden is None:
                        missing_anchor += 1
                    else:
                        correct_anchors.append(anchor_hidden)
            if align_layer is not None:
                a_loss, mean_cos = pairwise_alignment_loss(correct_anchors)
                if a_loss is not None:
                    aligned_pairs += len(correct_anchors) * (len(correct_anchors) - 1) // 2
                    align_vals.append(mean_cos)
                    losses.append(align_lambda * a_loss)

        loss_val = grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
            optimizer.step()
            loss_val = float(loss.detach().item())

        rec = {
            "arm": arm_name,
            "step": step,
            "mean_verifier_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4),
            "exact_rollout_rate": round(100.0 * exact_count / max(rollout_count, 1), 1),
            "active_groups": len(groups),
            "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0,
            "align_mean_cosine": round(sum(align_vals) / max(len(align_vals), 1), 5) if align_vals else 0.0,
            "aligned_pairs": aligned_pairs,
            "missing_anchor_count": missing_anchor,
            "loss": round(loss_val, 5),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 2),
        }
        trace.append(rec)
        print(
            f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_verifier_reward']:.3f} | "
            f"exact_roll={rec['exact_rollout_rate']:.1f}% | align_cos={rec['align_mean_cosine']:.3f} | "
            f"pairs={aligned_pairs} | kl={rec['mean_abs_kl']:.4f}",
            flush=True,
        )

    stats = {
        "vector_layers": vector_layers,
        "align_layer": align_layer,
        "align_lambda": align_lambda,
        "n_vector_params": sum(v.numel() for v in vectors.values()),
        "final_vector_l2_norms": {str(k): round(float(v.detach().norm().item()), 5) for k, v in vectors.items()},
        "train_wall_sec": round(time.perf_counter() - t0, 2),
        "trace": trace,
    }
    model.eval()
    return model, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--align-lambda", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp51_decision_anchor_hidden_align/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 51] Decision-Anchor Hidden-Align RL on {device} ({dtype})", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, dtype)

    scores, u_refusal = rnd_mod.compute_dla_scores(sft, tokenizer, train_rows, device)
    selected = {li: torch.topk(scores[li], k=64).indices.tolist() for li in vec_rl.EDIT_LAYERS}
    edited = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(edited, selected, u_refusal, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval()
    vec_rl.freeze_weights(ref_model)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [DLA] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)

    arms = [
        ("vector_only_control", None, 0.0),
        ("hidden_align_late_l29_lambda1em3", ALIGN_LAYERS["late_l29"], args.align_lambda),
        ("hidden_align_late_l29_lambda1em2", ALIGN_LAYERS["late_l29"], args.align_lambda * 10.0),
        ("hidden_align_mid_l20_lambda1em3", ALIGN_LAYERS["mid_l20"], args.align_lambda),
    ]
    results = {
        "experiment": "exp51_decision_anchor_hidden_align",
        "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 2",
        "paper_inspiration": ["arXiv:2606.03234", "arXiv:2611.30627"],
        "config": vars(args),
        "device": str(device),
        "selected_neurons": selected,
        "baseline_sft": sft_eval,
        "anchored_edit_only": edit_eval,
        "arms": {},
    }

    step_log_path = out_dir / "step_metrics.jsonl"
    with step_log_path.open("w", encoding="utf-8") as step_log:
        for arm_name, align_layer, lam in arms:
            print(f"[Run] {arm_name} (align_layer={align_layer}, lambda={lam})", flush=True)
            trained, stats = train_hidden_align_arm(
                arm_name,
                edited,
                ref_model,
                tokenizer,
                prompt_stream,
                device,
                vec_rl.DISJOINT_LAYERS,
                align_layer,
                lam,
                args.steps,
                args.group_size,
                args.lr,
                args.beta,
                args.seed,
            )
            for rec in stats["trace"]:
                step_log.write(json.dumps(rec) + "\n")
            step_log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(
                f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | "
                f"unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%",
                flush=True,
            )
            results["arms"][arm_name] = {"train_stats": stats, "eval": ev}
            del trained
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    summary_path = out_dir / "hidden_align_rl_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 51 DONE] wrote {summary_path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
