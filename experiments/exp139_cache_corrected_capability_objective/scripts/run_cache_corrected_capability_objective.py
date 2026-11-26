#!/usr/bin/env python3
"""Experiment 139: Cache-corrected capability-objective slider replication.

Iteration-4 priority test: can an inserted zero-init attention head acquire or
amplify capability when its training signal is language-model quality rather than
policy behavior? One fixed L27 width-64 head is trained under three objectives:
pure capability next-token loss, pure behavioral grouped REINFORCE, and a hybrid
capability-primary objective. All arms use identical architecture, seed, optimizer,
step budget, behavior prompt stream, and train-split capability corpus.

Contamination control: capability text comes only from MMLU validation, ARC-Easy
train, and HellaSwag train. The benchmark harness measures MMLU test, ARC-Easy
test, and HellaSwag validation, so no measured benchmark items are trained on.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
from capability_objective import causal_lm_loss, corpus_batch, evaluate_lm_loss, load_benchmark_style_train_corpus  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, install_plain_slider, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
HEAD_DIM = 64
OBJECTIVES = ["capability_only", "behavior_only", "hybrid_capability_primary"]


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def continuation_delta(after: dict, before: dict) -> dict:
    out = {}
    for bench, metrics in after.items():
        if bench not in before or not isinstance(metrics, dict) or not isinstance(before[bench], dict):
            continue
        out[bench] = {
            k: round(metrics[k] - before[bench][k], 2)
            for k in ("sum_acc", "token_norm_acc", "byte_norm_acc")
            if isinstance(metrics.get(k), (int, float)) and isinstance(before[bench].get(k), (int, float))
        }
    return out


def behavior_grouped_loss(model, tokenizer, row: dict, device: torch.device, group_size: int, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
    """Exact slider-fork grouped REINFORCE loss for one behavior prompt."""
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
            use_cache=False,
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
    loss = loss / group_size
    metrics = {
        "mean_reward": round(float(rew_t.mean().item()), 4),
        "reward_std": round(float(rew_t.std(unbiased=False).item()), 4),
        "raw_behavior_loss": round(float(loss.detach().item()), 5),
    }
    return loss, metrics


def train_objective_arm(
    objective: str,
    model,
    tokenizer,
    behavior_rows: list[dict],
    capability_rows: list[dict],
    device: torch.device,
    head: PlainSliderHead,
    layer: int,
    steps: int,
    group_size: int,
    batch_size: int,
    lr: float,
    behavior_weight: float,
    max_new_tokens: int,
) -> tuple[object, list[dict]]:
    handle = install_plain_slider(model, layer, head)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        total_loss = torch.zeros((), device=device)
        rec = {"step": step, "objective": objective}
        if objective in {"capability_only", "hybrid_capability_primary"}:
            batch = corpus_batch(capability_rows, step, batch_size)
            cap_loss = causal_lm_loss(model, tokenizer, [r["text"] for r in batch], device)
            total_loss = total_loss + cap_loss
            rec["capability_loss"] = round(float(cap_loss.detach().item()), 5)
            rec["capability_sources"] = sorted({r["source"] for r in batch})
        else:
            rec["capability_loss"] = None
            rec["capability_sources"] = []
        if objective in {"behavior_only", "hybrid_capability_primary"}:
            row = behavior_rows[step % len(behavior_rows)]
            behavior_loss, behavior_metrics = behavior_grouped_loss(model, tokenizer, row, device, group_size, max_new_tokens)
            total_loss = total_loss + behavior_weight * behavior_loss
            rec.update(behavior_metrics)
            rec["weighted_behavior_loss"] = round(float((behavior_weight * behavior_loss).detach().item()), 5)
            rec["prompt_action"] = row["target"]["action"]
        else:
            rec["mean_reward"] = None
            rec["reward_std"] = None
            rec["raw_behavior_loss"] = None
            rec["weighted_behavior_loss"] = None
            rec["prompt_action"] = None
        model.train()
        head.train()
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
        opt.step()
        rec.update(
            {
                "alpha": round(float(head.alpha.detach().item()), 6),
                "total_loss": round(float(total_loss.detach().item()), 5),
                "grad_norm": round(grad_norm, 5),
                "step_wall_sec": round(time.perf_counter() - step_t0, 3),
            }
        )
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(
                f"    step {step:02d}: cap={rec['capability_loss']} rew={rec['mean_reward']} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}",
                flush=True,
            )
    return handle, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--cap-items-per-source", type=int, default=64)
    parser.add_argument("--behavior-weight", type=float, default=10.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp139_cache_corrected_capability_objective/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 139] Capability-objective slider ablation on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(train_rows, 12)
    corpus = load_benchmark_style_train_corpus(args.cap_items_per_source, args.seed)
    print(
        f"  capability corpus: train={len(corpus['train'])} validation={len(corpus['validation'])} sha256={corpus['sha256'][:12]} errors={list(corpus['errors'])}",
        flush=True,
    )

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, use_cache=False)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline legacy benchmark] {baseline_bench}", flush=True)
    baseline_continuation = run_continuation_benchmark(sft, tokenizer, device, args.bench_n)
    print(f"  [baseline continuation benchmark] {baseline_continuation}", flush=True)
    baseline_cap_loss = evaluate_lm_loss(sft, tokenizer, corpus["validation"], device, batch_size=args.batch_size)
    print(f"  [baseline heldout capability loss] {baseline_cap_loss}", flush=True)
    probe_rows = behavior_rows[:4] + behavior_rows[-4:]
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, probe_rows, device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp139_cache_corrected_capability_objective",
        "controlled_variable": "training objective only: pure capability vs pure behavioral vs hybrid capability-primary; cache-corrected replication of exp132",
        "scientific_question": "Can an inserted zero-init slider head add measurable capability in a saturated 135M residual stream when trained with a capability signal?",
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": HEAD_DIM,
            "head_count": 1,
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas/weight decay",
            "steps": args.steps,
            "capability_batch_size": args.batch_size,
            "behavior_group_size": args.group_size,
            "behavior_prompt_stream": "12 benign then 12 unsafe, fixed order",
            "hybrid_loss": "capability_loss + behavior_weight * behavior_loss",
            "generation_cache_mode": "use_cache=False for inserted-head generation and behavioral evaluation",
        },
        "objective_scaling": {"capability_weight": 1.0, "behavior_weight": args.behavior_weight},
        "capability_corpus": {
            "train_count": len(corpus["train"]),
            "validation_count": len(corpus["validation"]),
            "sha256": corpus["sha256"],
            "provenance": corpus["provenance"],
            "load_errors": corpus["errors"],
            "contamination_control": "uses only MMLU validation, ARC-Easy train, HellaSwag train; benchmark uses MMLU test, ARC-Easy test, HellaSwag validation",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_continuation_benchmark": baseline_continuation,
        "baseline_capability_validation_loss": baseline_cap_loss,
        "transformerlens_probe": tl_probe,
        "arms": {},
    }

    for objective in OBJECTIVES:
        print(f"[Run] {objective}: fixed L27 width64 head, objective only", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity_diff = float((head(probe) - probe).abs().max().item())
        if identity_diff != 0.0:
            raise RuntimeError(f"zero-init identity failed for {objective}: {identity_diff}")
        handle, trace = train_objective_arm(
            objective,
            sft,
            tokenizer,
            behavior_rows,
            corpus["train"],
            device,
            head,
            LAYER,
            steps=args.steps,
            group_size=args.group_size,
            batch_size=args.batch_size,
            lr=args.lr,
            behavior_weight=args.behavior_weight,
            max_new_tokens=args.max_new_tokens,
        )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, use_cache=False)
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        continuation = run_continuation_benchmark(sft, tokenizer, device, args.bench_n)
        cap_val_loss = evaluate_lm_loss(sft, tokenizer, corpus["validation"], device, batch_size=args.batch_size)
        handle.remove()
        print(
            f"  [{objective}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} continuation={continuation} cap_val={cap_val_loss} alpha={head.alpha.item():.5f}",
            flush=True,
        )
        results["arms"][objective] = {
            "objective": objective,
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity_diff,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": metric_delta(bench, baseline_bench),
            "continuation_benchmark": continuation,
            "continuation_delta_vs_baseline": continuation_delta(continuation, baseline_continuation),
            "capability_validation_loss": cap_val_loss,
            "capability_validation_loss_delta_vs_baseline": round(cap_val_loss - baseline_cap_loss, 6),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "cache_corrected_capability_objective_summary.json"
    write_json(path, results)
    print(f"[Exp 139 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
