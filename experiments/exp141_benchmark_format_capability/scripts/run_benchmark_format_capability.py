#!/usr/bin/env python3
"""Experiment 141: benchmark-format capability training.

Exp140 isolates answer-token weighting within the original lettered corpus format.
This experiment isolates the remaining format mismatch: the continuation benchmark
scores `context + " " + answer` where context is the bare question/passage, while
the original corpus trains on lettered options followed by an explicit marker.
Both arms use answer-only loss, identical underlying rows, ordering, architecture,
seed, optimizer, and budget; only surface format varies.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp139_cache_corrected_capability_objective/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
from capability_objective import corpus_batch, evaluate_lm_loss, evaluate_span_weighted_lm_loss, load_benchmark_style_train_corpus, span_weighted_causal_lm_loss  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_cache_corrected_capability_objective as caprep  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, install_plain_slider, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
HEAD_DIM = 64
ARMS = ["lettered_answer_only_reference", "benchmark_direct_answer_only"]


def direct_record(source: str, context: str, answer: str) -> dict:
    context = context.strip()
    answer = answer.strip()
    return {
        "source": source,
        "text": f"{context} {answer}",
        "answer_start_char": len(context) + 1,
        "answer_text": answer,
    }


def split_round_robin(by_source: dict[str, list[dict]], seed: int) -> tuple[list[dict], list[dict]]:
    train_parts: dict[str, list[dict]] = {}
    validation_parts: dict[str, list[dict]] = {}
    for source, items in by_source.items():
        if len(items) >= 4:
            n_val = max(1, len(items) // 4)
        elif len(items) == 2:
            n_val = 1
        else:
            n_val = 0
        validation_parts[source] = items[-n_val:] if n_val else []
        train_parts[source] = items[: len(items) - n_val] if n_val else list(items)

    def round_robin(parts: dict[str, list[dict]]) -> list[dict]:
        remaining = {k: list(v) for k, v in parts.items() if v}
        out = []
        while remaining:
            for source in sorted(list(remaining)):
                if remaining[source]:
                    out.append(remaining[source].pop(0))
                if not remaining[source]:
                    del remaining[source]
        return out

    train = round_robin(train_parts)
    validation = round_robin(validation_parts)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def load_benchmark_direct_corpus(per_source: int, seed: int) -> dict:
    """Load the same split rows as capability_objective, but in continuation-benchmark format."""
    by_source: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    provenance: list[dict] = []
    try:
        from datasets import load_dataset

        try:
            rows = load_dataset("cais/mmlu", "all", split=f"validation[:{per_source}]")
            items = [direct_record("mmlu_validation", r["question"], r["choices"][int(r["answer"])]) for r in rows]
            by_source["mmlu_validation"] = items
            provenance.append({"source": "mmlu_validation", "dataset": "cais/mmlu/all", "split": "validation", "format": "question + space + answer", "rows_loaded": len(items)})
        except Exception as exc:
            errors["mmlu_validation"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"train[:{per_source}]")
            items = []
            for r in rows:
                choices = list(r["choices"]["text"])
                labels = list(r["choices"]["label"])
                items.append(direct_record("arc_easy_train", r["question"], choices[labels.index(r["answerKey"])]))
            by_source["arc_easy_train"] = items
            provenance.append({"source": "arc_easy_train", "dataset": "allenai/ai2_arc/ARC-Easy", "split": "train", "format": "question + space + answer", "rows_loaded": len(items)})
        except Exception as exc:
            errors["arc_easy_train"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = load_dataset("Rowan/hellaswag", split=f"train[:{per_source}]")
            items = [direct_record("hellaswag_train", r["ctx"], r["endings"][int(r["label"])]) for r in rows]
            by_source["hellaswag_train"] = items
            provenance.append({"source": "hellaswag_train", "dataset": "Rowan/hellaswag", "split": "train", "format": "context + space + continuation", "rows_loaded": len(items)})
        except Exception as exc:
            errors["hellaswag_train"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        errors["datasets_import"] = f"{type(exc).__name__}: {exc}"

    if not any(by_source.values()):
        fallback = [
            ("If a train travels 45 miles in one hour, how far does it travel in three hours at the same speed?", "135 miles"),
            ("Which process allows plants to convert light energy into chemical energy?", "photosynthesis"),
            ("A store reduces a $40 item by 25%. What is the sale price?", "$30"),
            ("Water is heated in a closed pot. As the temperature rises, more molecules escape the liquid phase.", "The vapor pressure therefore increases."),
            ("In a fair coin toss, heads and tails are equally likely.", "Across many independent tosses, the fraction of heads usually approaches one half."),
            ("Which gas makes up most of Earth's atmosphere?", "nitrogen"),
        ]
        by_source["synthetic_fallback"] = [direct_record("synthetic_fallback", c, a) for c, a in fallback]
        errors["fallback_used"] = "no external benchmark-format corpus loaded"
    train, validation = split_round_robin(by_source, seed)
    payload = "\n".join(f"{r['source']}::{r['text']}" for r in train + validation)
    return {
        "train": train,
        "validation": validation,
        "provenance": provenance,
        "errors": errors,
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def answer_payload(text: str) -> str:
    for marker in ("\nAnswer: ", "\nContinuation: "):
        idx = text.rfind(marker)
        if idx >= 0:
            return text[idx + len(marker) :].strip()
    return ""


def validation_metrics(model, tokenizer, records: list[dict], device: torch.device, batch_size: int) -> dict:
    starts = [r.get("answer_start_char") for r in records]
    explicit = all(x is not None for x in starts)
    return {
        "full_validation_loss": evaluate_lm_loss(model, tokenizer, records, device, batch_size=batch_size),
        "answer_only_validation_loss": evaluate_span_weighted_lm_loss(
            model,
            tokenizer,
            records,
            device,
            context_weight=0.0,
            answer_weight=1.0,
            answer_starts=starts if explicit else None,
            batch_size=batch_size,
        ),
    }


def measure_state(label, model, tokenizer, eval_rows, lettered_validation, direct_validation, device, bench_n, continuation_n, batch_size, max_new_tokens):
    behavior = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device, max_new_tokens=max_new_tokens, use_cache=False)
    bench = run_benchmark(model, tokenizer, device, bench_n)
    continuation = run_continuation_benchmark(model, tokenizer, device, continuation_n)
    lettered_val = validation_metrics(model, tokenizer, lettered_validation, device, batch_size)
    direct_val = validation_metrics(model, tokenizer, direct_validation, device, batch_size)
    print(
        f"  [{label}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} lettered_answer={lettered_val['answer_only_validation_loss']} direct_answer={direct_val['answer_only_validation_loss']}",
        flush=True,
    )
    return {
        "behavior_eval": behavior,
        "benchmark": bench,
        "continuation_benchmark": continuation,
        "lettered_validation": lettered_val,
        "direct_validation": direct_val,
    }


def train_format_arm(arm, records, model, tokenizer, device, head, steps, batch_size, lr):
    handle = install_plain_slider(model, LAYER, head)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    log = []
    direct = arm == "benchmark_direct_answer_only"
    for step in range(steps):
        step_t0 = time.perf_counter()
        batch = corpus_batch(records, step, batch_size)
        starts = [r.get("answer_start_char") for r in batch] if direct else None
        loss = span_weighted_causal_lm_loss(
            model,
            tokenizer,
            [r["text"] for r in batch],
            device,
            context_weight=0.0,
            answer_weight=1.0,
            answer_starts=starts,
        )
        model.train(); head.train(); opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
        opt.step()
        rec = {
            "step": step,
            "objective": arm,
            "capability_sources": sorted({r["source"] for r in batch}),
            "answer_only_loss": round(float(loss.detach().item()), 5),
            "alpha": round(float(head.alpha.detach().item()), 6),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(f"    {arm} step {step:02d}: loss={rec['answer_only_loss']:.3f} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}", flush=True)
    return handle, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--continuation-n", type=int, default=150)
    parser.add_argument("--cap-items-per-source", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp141_benchmark_format_capability/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 141] Benchmark-format capability training on {device}", flush=True)
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(base_rl.load_rows(ROOT / "data/policy_train.jsonl"), 12)
    lettered = load_benchmark_style_train_corpus(args.cap_items_per_source, args.seed)
    direct = load_benchmark_direct_corpus(args.cap_items_per_source, args.seed)
    source_order_match = [r["source"] for r in lettered["train"]] == [r["source"] for r in direct["train"]]
    payload_mismatches = sum(
        int(answer_payload(a["text"]) != b["text"][int(b["answer_start_char"]) :].strip())
        for a, b in zip(lettered["train"], direct["train"])
    )
    print(f"  pairing: source_order_match={source_order_match} payload_mismatches={payload_mismatches}/{len(direct['train'])}", flush=True)
    if not source_order_match:
        raise RuntimeError("lettered/direct corpus ordering mismatch; format comparison would be confounded")
    if payload_mismatches:
        raise RuntimeError("lettered/direct answer payload mismatch; format comparison would be confounded")

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    baseline = measure_state("baseline", sft, tokenizer, eval_rows, lettered["validation"], direct["validation"], device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, behavior_rows[:4] + behavior_rows[-4:], device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp141_benchmark_format_capability",
        "controlled_variable": "training-text surface format only, with answer-only token weighting fixed",
        "scientific_question": "Does exact continuation-benchmark formatting improve industry-benchmark transfer relative to lettered-marker capability format?",
        "fixed_conditions": {
            "layer": LAYER,
            "head_dim": HEAD_DIM,
            "objective": "answer-only weighted causal LM loss",
            "seed": args.seed,
            "steps": args.steps,
            "capability_batch_size": args.batch_size,
            "same_underlying_rows": True,
            "source_order_match": source_order_match,
            "answer_payload_mismatches": payload_mismatches,
            "generation_cache_mode": "use_cache=False for behavioral evaluation; no behavioral generation is used in training",
        },
        "baseline": baseline,
        "lettered_corpus": {"train_count": len(lettered["train"]), "validation_count": len(lettered["validation"]), "sha256": lettered["sha256"], "provenance": lettered["provenance"], "load_errors": lettered["errors"]},
        "direct_corpus": {"train_count": len(direct["train"]), "validation_count": len(direct["validation"]), "sha256": direct["sha256"], "provenance": direct["provenance"], "load_errors": direct["errors"]},
        "contamination_control": "uses only MMLU validation, ARC-Easy train, HellaSwag train; benchmark uses MMLU test, ARC-Easy test, HellaSwag validation",
        "transformerlens_probe": tl_probe,
        "config": vars(args),
        "arms": {},
    }

    for arm in ARMS:
        direct_arm = arm == "benchmark_direct_answer_only"
        corpus = direct if direct_arm else lettered
        print(f"[Arm] {arm}: format only", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity = float((head(probe) - probe).abs().max().item())
        if identity != 0.0:
            raise RuntimeError(f"zero-init identity failed for {arm}: {identity}")
        handle, trace = train_format_arm(arm, corpus["train"], sft, tokenizer, device, head, args.steps, args.batch_size, args.lr)
        state = measure_state(arm, sft, tokenizer, eval_rows, lettered["validation"], direct["validation"], device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
        handle.remove()
        validation_key = "direct_validation" if direct_arm else "lettered_validation"
        results["arms"][arm] = {
            **state,
            "behavior_delta_vs_baseline": caprep.metric_delta(state["behavior_eval"], baseline["behavior_eval"]),
            "benchmark_delta_vs_baseline": caprep.metric_delta(state["benchmark"], baseline["benchmark"]),
            "continuation_delta_vs_baseline": caprep.continuation_delta(state["continuation_benchmark"], baseline["continuation_benchmark"]),
            "matched_answer_validation_delta_vs_baseline": round(state[validation_key]["answer_only_validation_loss"] - baseline[validation_key]["answer_only_validation_loss"], 6),
            "matched_full_validation_delta_vs_baseline": round(state[validation_key]["full_validation_loss"] - baseline[validation_key]["full_validation_loss"], 6),
            "training_format": "benchmark_direct" if direct_arm else "lettered_marker",
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "benchmark_format_capability_summary.json"
    write_json(path, results)
    print(f"[Exp 141 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
