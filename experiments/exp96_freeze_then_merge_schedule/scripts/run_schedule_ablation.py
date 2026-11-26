#!/usr/bin/env python3
"""Experiment 96: Freeze-then-merge vs joint training for the v2 multi-layer bank.

Both arms use the same v2 architecture, seed, data stream, total optimizer-step
budget, learning rate, alphas, and evaluation. Only the causal training schedule
changes: all four heads jointly for 32 steps, or each head alone for 8 steps in
layer order with prior heads frozen but merged into the forward pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import dumps_safe, write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from methodical_slider import IndependentLayerSliderBank, head_parameter_count, install_independent_layer_bank, ordered_fork_rows, probe_layers_geometry, train_independent_layer_bank  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402

LAYERS = [24, 25, 26, 27]
WIDTH = 64
TOTAL_STEPS = 32


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def rollout_loss(model, tokenizer, row, device, group_size: int, max_new_tokens: int = 48):
    ids = tokenizer(base_rl.build_prompt(tokenizer, row["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
    prompt_len = int(ids["input_ids"].shape[1])
    model.eval()
    with torch.no_grad():
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id,
        )
    rewards = []
    for gi in range(group_size):
        text = tokenizer.decode(gen[gi][prompt_len:], skip_special_tokens=True)
        rew, _ = base_rl.score_completion(text, row["target"])
        rewards.append(float(rew))
    rew_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    adv = rew_t - rew_t.mean()
    model.train()
    loss = torch.zeros((), device=device)
    for gi in range(group_size):
        lp = base_rl.completion_logprobs(model, gen[gi].unsqueeze(0), prompt_len)
        loss = loss - adv[gi] * lp
    return loss / group_size, rew_t


def train_freeze_then_merge(model, tokenizer, rows, device, bank, group_size: int, lr: float, steps_per_head: int):
    handles = install_independent_layer_bank(model, bank)
    log = []
    global_step = 0
    for layer in bank.layers:
        head = bank.layer_heads[str(layer)]
        for p in head.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(head.parameters(), lr=lr)
        for local_step in range(steps_per_head):
            t0 = time.perf_counter()
            row = rows[global_step % len(rows)]
            loss, rew_t = rollout_loss(model, tokenizer, row, device, group_size)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
            opt.step()
            rec = {
                "step": global_step,
                "active_layer": layer,
                "local_step": local_step,
                "mean_reward": round(float(rew_t.mean().item()), 4),
                "alpha": round(float(head.alpha.detach().item()), 6),
                "loss": round(float(loss.detach().item()), 5),
                "grad_norm": round(grad_norm, 5),
                "step_wall_sec": round(time.perf_counter() - t0, 3),
            }
            log.append(rec)
            global_step += 1
        for p in head.parameters():
            p.requires_grad_(False)
    return handles, log


def build_bank(sft, device, seed: int):
    torch.manual_seed(seed)
    bank = IndependentLayerSliderBank(sft.config.hidden_size, WIDTH, LAYERS).to(device)
    with torch.no_grad():
        probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
        identity_diff = max(float((head(probe) - probe).abs().max().item()) for head in bank.layer_heads.values())
    if identity_diff != 0.0:
        raise RuntimeError(f"zero-init identity failed: {identity_diff}")
    return bank, identity_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp96_freeze_then_merge_schedule/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 96] Freeze-then-merge schedule ablation on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = ordered_fork_rows(train_rows, 12)
    probe_rows = rows[:4] + rows[-4:]
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline benchmark] {baseline_bench}", flush=True)
    try:
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, LAYERS)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp96_freeze_then_merge_schedule",
        "controlled_variable": "training schedule only: joint 32 steps vs sequential frozen 8 steps/head",
        "fixed_conditions": {
            "layers": LAYERS,
            "head_dim": WIDTH,
            "total_optimizer_steps": args.steps,
            "seed": args.seed,
            "architecture": "v2 width-64 independent-alpha multi-layer bank",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "transformerlens_probe_by_layer": tl_probe,
        "arms": {},
    }

    for schedule in ["joint_32", "freeze_then_merge_8x4"]:
        print(f"[Run] {schedule}: same v2 architecture and optimizer-step budget", flush=True)
        bank, identity_diff = build_bank(sft, device, args.seed)
        if schedule == "joint_32":
            handles, trace = train_independent_layer_bank(
                sft, tokenizer, rows, device, bank,
                steps=args.steps, group_size=args.group_size, lr=args.lr,
            )
        else:
            handles, trace = train_freeze_then_merge(
                sft, tokenizer, rows, device, bank,
                group_size=args.group_size, lr=args.lr,
                steps_per_head=max(1, args.steps // len(LAYERS)),
            )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        for handle in handles:
            handle.remove()
        final_alphas = {str(k): round(v, 6) for k, v in bank.alphas().items()}
        print(f"  [{schedule}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alphas={final_alphas} bench={bench}", flush=True)
        results["arms"][schedule] = {
            "parameter_count": head_parameter_count(bank),
            "zero_init_identity_max_diff": identity_diff,
            "final_alphas": final_alphas,
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": metric_delta(bench, baseline_bench),
            "trace": trace,
        }
        del bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "freeze_then_merge_schedule_summary.json"
    write_json(path, results)
    print(f"[Exp 96 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
