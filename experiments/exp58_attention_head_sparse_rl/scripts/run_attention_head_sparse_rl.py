#!/usr/bin/env python3
"""Experiment 58: Attention-Head Sparse RL Complementary to MLP Surgery.

Implements SUBAGENT1 rank-8: rank attention heads by initial action-decision
gradient norm, then run exact-compliance RL only through q_proj/output-projection
slices for selected heads while all MLPs, embeddings, and surgically edited
layers remain frozen. Compares top-ranked, bottom-ranked, random, and dense
matched-budget controls.

Research anchors: arXiv:2610.21336 and arXiv:2611.10866.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402


def first_action_pos(ids: torch.Tensor, prompt_len: int, action_ids: set[int]) -> int | None:
    seq = ids.tolist()
    for pos in range(prompt_len, len(seq)):
        if seq[pos] in action_ids:
            return pos
    return None


def rank_heads(model, tokenizer, train_rows, device, n_rows: int, seed: int) -> list[dict]:
    probe = copy.deepcopy(model).to(device)
    vec_rl.freeze_weights(probe)
    trainable = []
    for li, layer in enumerate(probe.model.layers):
        if li in vec_rl.EDIT_LAYERS:
            continue
        layer.self_attn.q_proj.weight.requires_grad_(True)
        layer.self_attn.o_proj.weight.requires_grad_(True)
        trainable.extend([layer.self_attn.q_proj.weight, layer.self_attn.o_proj.weight])
    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[0] for x in ("call", "ref"))
    rows = train_rows[:n_rows]
    losses = []
    for row in rows:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        full = prompt + row["assistant_text"]
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=320).to(device)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=256)["input_ids"])
        pos = first_action_pos(enc["input_ids"][0], prompt_len, action_ids)
        if pos is None or pos == 0:
            continue
        out = probe(input_ids=enc["input_ids"])
        logits = out.logits[0, pos - 1, :].float()
        target = enc["input_ids"][0, pos]
        losses.append(-F.log_softmax(logits, dim=-1)[target])
    if not losses:
        raise RuntimeError("no action-decision positions found for head ranking")
    probe.zero_grad(set_to_none=True)
    torch.stack(losses).mean().backward()

    head_dim = probe.config.hidden_size // probe.config.num_attention_heads
    records = []
    for li, layer in enumerate(probe.model.layers):
        if li in vec_rl.EDIT_LAYERS:
            continue
        qg = layer.self_attn.q_proj.weight.grad
        og = layer.self_attn.o_proj.weight.grad
        if qg is None or og is None:
            continue
        for head in range(probe.config.num_attention_heads):
            lo, hi = head * head_dim, (head + 1) * head_dim
            norm = float(qg[lo:hi, :].norm().item() + og[:, lo:hi].norm().item())
            records.append({"layer": li, "head": head, "grad_norm": norm})
    records.sort(key=lambda x: -x["grad_norm"])
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def select_heads(ranked: list[dict], n_select: int, mode: str, seed: int, dense_layers: list[int]) -> list[dict]:
    if mode == "top":
        return ranked[:n_select]
    if mode == "bottom":
        return list(reversed(ranked[-n_select:]))
    if mode == "random":
        rng = random.Random(seed)
        pool = ranked.copy()
        rng.shuffle(pool)
        return pool[:n_select]
    if mode == "dense":
        n_heads = max(int(x["head"]) for x in ranked) + 1
        lookup = {(int(x["layer"]), int(x["head"])): x for x in ranked}
        out = []
        for li in dense_layers:
            for head in range(n_heads):
                rec = lookup.get((li, head))
                if rec is not None:
                    out.append(rec)
                    if len(out) >= n_select:
                        return out
        return (out + ranked[: max(0, n_select - len(out))])[:n_select]
    raise ValueError(mode)


def install_head_masks(model, selected: list[dict]) -> tuple[list[torch.nn.Parameter], int]:
    vec_rl.freeze_weights(model)
    hidden = model.config.hidden_size
    head_dim = hidden // model.config.num_attention_heads
    by_layer: dict[int, list[int]] = {}
    for rec in selected:
        by_layer.setdefault(int(rec["layer"]), []).append(int(rec["head"]))
    params = []
    effective = 0
    for li, heads in by_layer.items():
        q = model.model.layers[li].self_attn.q_proj.weight
        o = model.model.layers[li].self_attn.o_proj.weight
        q_mask = torch.zeros_like(q)
        o_mask = torch.zeros_like(o)
        for head in heads:
            lo, hi = head * head_dim, (head + 1) * head_dim
            q_mask[lo:hi, :] = 1.0
            o_mask[:, lo:hi] = 1.0
            effective += 2 * head_dim * hidden
        q.requires_grad_(True)
        o.requires_grad_(True)
        q.register_hook(lambda grad, mask=q_mask: grad * mask)
        o.register_hook(lambda grad, mask=o_mask: grad * mask)
        params.extend([q, o])
    return params, effective


def train_sparse_attention_arm(arm_name, selected, init_model, ref_model, tokenizer, prompt_stream, device, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    params, effective = install_head_masks(model, selected)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    trace = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval()
        groups = []
        rewards_all = []
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
                rewards_all.append(float(reward))
                items.append({"seq": seq, "reward": float(reward), "exact": info["exact"]})
            r = torch.tensor([x["reward"] for x in items], dtype=torch.float32, device=device)
            if float(r.std(unbiased=False).item()) < 1e-6:
                continue
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
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
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
            opt.step()
            loss_val = float(loss.detach().item())
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(rewards_all) / max(len(rewards_all), 1), 4), "active_groups": len(groups), "dead_groups": 2 - len(groups), "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0, "loss": round(loss_val, 5), "grad_norm": round(grad_norm, 5)}
        trace.append(rec)
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']}", flush=True)
    stats = {"trace": trace, "train_wall_sec": round(time.perf_counter() - t0, 2), "optimizer_tensor_params": sum(p.numel() for p in params), "effective_trainable_slice_params": effective, "selected_heads": selected}
    model.eval()
    return model, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-fraction", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp58_attention_head_sparse_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 58] Attention-Head Sparse RL on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, dtype)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval()
    vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [validated edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)

    print("[Stage 1] Ranking attention heads by action-decision gradient norm...", flush=True)
    ranked = rank_heads(edited, tokenizer, train_rows, device, n_rows=8, seed=args.seed)
    n_select = max(1, int(round(len(ranked) * args.head_fraction)))
    dense_layers = [24, 25, 26, 27]
    selections = {
        "top_ranked_heads": select_heads(ranked, n_select, "top", args.seed, dense_layers),
        "bottom_ranked_heads": select_heads(ranked, n_select, "bottom", args.seed, dense_layers),
        "random_ranked_heads": select_heads(ranked, n_select, "random", args.seed, dense_layers),
        "dense_complementary_control": select_heads(ranked, n_select, "dense", args.seed, dense_layers),
    }
    print(f"  selected {n_select}/{len(ranked)} heads per arm; top={[(x['layer'],x['head']) for x in selections['top_ranked_heads'][:6]]}", flush=True)

    results = {"experiment": "exp58_attention_head_sparse_rl", "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 8", "paper_inspiration": ["arXiv:2610.21336", "arXiv:2611.10866"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "baseline_sft": sft_eval, "anchored_edit_only": edit_eval, "n_candidate_heads": len(ranked), "n_selected_heads": n_select, "top_ranked_head_preview": ranked[:12], "arms": {}}
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm_name, selected in selections.items():
            print(f"[Run] {arm_name}", flush=True)
            trained, stats = train_sparse_attention_arm(arm_name, selected, edited, ref_model, tokenizer, prompt_stream, device, args.steps, args.group_size, args.lr, args.beta, args.seed)
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
    path = out_dir / "attention_head_sparse_rl_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 58 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
