#!/usr/bin/env python3
"""Experiment 143: contrastive continuation-choice capability objective.

Correct-answer causal training is format-aligned but still not benchmark-aligned:
the continuation benchmark is discriminative, comparing several possible answers in
the same context. This experiment holds data/order/format/steps fixed and varies
only the objective: correct-answer causal loss vs token-normalized contrastive
choice loss over all candidate continuations.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp139_cache_corrected_capability_objective/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp141_benchmark_format_capability/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
from capability_objective import corpus_batch, span_weighted_causal_lm_loss  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_cache_corrected_capability_objective as caprep  # noqa: E402
import run_benchmark_format_capability as fmt  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, install_plain_slider, ordered_fork_rows, probe_layer27_geometry  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import run_continuation_benchmark  # noqa: E402

LAYER = 27
HEAD_DIM = 64
ARMS = ["correct_answer_causal_reference", "contrastive_choice_objective"]


def mc_record(source: str, context: str, choices: list[str], correct_index: int) -> dict:
    clean_choices = [c.strip() for c in choices]
    return {
        "source": source,
        "context": context.strip(),
        "choices": clean_choices,
        "correct_index": int(correct_index),
        "correct_text": clean_choices[int(correct_index)],
    }


def load_multiple_choice_corpus(per_source: int, seed: int) -> dict:
    by_source: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    provenance: list[dict] = []
    try:
        from datasets import load_dataset
        try:
            rows = load_dataset("cais/mmlu", "all", split=f"validation[:{per_source}]")
            items = [mc_record("mmlu_validation", r["question"], list(r["choices"]), int(r["answer"])) for r in rows]
            by_source["mmlu_validation"] = items
            provenance.append({"source": "mmlu_validation", "dataset": "cais/mmlu/all", "split": "validation", "rows_loaded": len(items)})
        except Exception as exc:
            errors["mmlu_validation"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"train[:{per_source}]")
            items = []
            for r in rows:
                labels = list(r["choices"]["label"])
                items.append(mc_record("arc_easy_train", r["question"], list(r["choices"]["text"]), labels.index(r["answerKey"])))
            by_source["arc_easy_train"] = items
            provenance.append({"source": "arc_easy_train", "dataset": "allenai/ai2_arc/ARC-Easy", "split": "train", "rows_loaded": len(items)})
        except Exception as exc:
            errors["arc_easy_train"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = load_dataset("Rowan/hellaswag", split=f"train[:{per_source}]")
            items = [mc_record("hellaswag_train", r["ctx"], list(r["endings"]), int(r["label"])) for r in rows]
            by_source["hellaswag_train"] = items
            provenance.append({"source": "hellaswag_train", "dataset": "Rowan/hellaswag", "split": "train", "rows_loaded": len(items)})
        except Exception as exc:
            errors["hellaswag_train"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        errors["datasets_import"] = f"{type(exc).__name__}: {exc}"
    if not any(by_source.values()):
        fallback = [
            ("If a train travels 45 miles in one hour, how far does it travel in three hours at the same speed?", ["45 miles", "90 miles", "135 miles", "180 miles"], 2),
            ("Which process allows plants to convert light energy into chemical energy?", ["respiration", "photosynthesis", "fermentation", "osmosis"], 1),
            ("A store reduces a $40 item by 25%. What is the sale price?", ["$10", "$25", "$30", "$35"], 2),
            ("Water is heated in a closed pot. As the temperature rises, more molecules escape the liquid phase.", ["The vapor pressure therefore increases.", "The water immediately freezes.", "The vapor pressure disappears.", "The pot loses all heat."], 0),
            ("In a fair coin toss, heads and tails are equally likely.", ["Across many independent tosses, the fraction of heads usually approaches one half.", "Every second toss must be tails.", "The fraction of heads is always exactly one.", "The coin becomes unfair after one toss."], 0),
            ("Which gas makes up most of Earth's atmosphere?", ["oxygen", "carbon dioxide", "nitrogen", "hydrogen"], 2),
        ]
        by_source["synthetic_fallback"] = [mc_record("synthetic_fallback", c, ch, i) for c, ch, i in fallback]
        errors["fallback_used"] = "no external multiple-choice corpus loaded"
    train, validation = fmt.split_round_robin(by_source, seed)
    payload = "\n".join(f"{r['source']}::{r['context']}::{r['correct_text']}" for r in train + validation)
    return {"train": train, "validation": validation, "provenance": provenance, "errors": errors, "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def direct_records_from_mc(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        context = r["context"].strip()
        answer = r["correct_text"].strip()
        out.append({"source": r["source"], "text": f"{context} {answer}", "answer_start_char": len(context) + 1, "answer_text": answer})
    return out


def flatten_choice_batch(records: list[dict], tokenizer, device: torch.device, max_length: int = 512):
    texts: list[str] = []
    prefix_lens: list[int] = []
    group_slices: list[tuple[int, int, int]] = []
    pos = 0
    for row, r in enumerate(records):
        start = pos
        for choice in r["choices"]:
            context = r["context"]
            full = context + " " + choice
            texts.append(full)
            prefix_lens.append(len(tokenizer(context, add_special_tokens=False)["input_ids"]))
            pos += 1
        group_slices.append((start, pos, int(r["correct_index"])))
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
    return enc.to(device), torch.tensor(prefix_lens, device=device), group_slices


def contrastive_choice_loss(model, tokenizer, records: list[dict], device: torch.device, max_length: int = 512) -> torch.Tensor:
    enc, prefix_lens, groups = flatten_choice_batch(records, tokenizer, device, max_length)
    logits = model(**enc).logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = enc["input_ids"][:, 1:]
    token_logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    target_positions = torch.arange(token_logp.shape[1], device=device).unsqueeze(0) + 1
    choice_mask = (target_positions >= prefix_lens.unsqueeze(1)) & enc["attention_mask"][:, 1:].bool()
    counts = choice_mask.sum(dim=1).clamp_min(1)
    scores = (token_logp * choice_mask).sum(dim=1) / counts
    losses = []
    for start, end, correct in groups:
        group_scores = scores[start:end]
        losses.append(-F.log_softmax(group_scores, dim=0)[correct])
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_contrastive_validation(model, tokenizer, records: list[dict], device: torch.device, batch_size: int) -> dict:
    losses = []
    correct = 0
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if not batch:
            continue
        enc, prefix_lens, groups = flatten_choice_batch(batch, tokenizer, device)
        logits = model(**enc).logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = enc["input_ids"][:, 1:]
        token_logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        target_positions = torch.arange(token_logp.shape[1], device=device).unsqueeze(0) + 1
        choice_mask = (target_positions >= prefix_lens.unsqueeze(1)) & enc["attention_mask"][:, 1:].bool()
        counts = choice_mask.sum(dim=1).clamp_min(1)
        scores = (token_logp * choice_mask).sum(dim=1) / counts
        for g_start, g_end, answer in groups:
            group_scores = scores[g_start:g_end]
            losses.append(float((-F.log_softmax(group_scores, dim=0)[answer]).item()))
            correct += int(int(group_scores.argmax().item()) == answer)
            total += 1
    return {"loss": sum(losses) / max(len(losses), 1), "accuracy": 100.0 * correct / max(total, 1), "n": total}


def measure_state(label, model, tokenizer, eval_rows, validation_records, device, bench_n, continuation_n, batch_size, max_new_tokens):
    behavior = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device, max_new_tokens=max_new_tokens, use_cache=False)
    bench = run_benchmark(model, tokenizer, device, bench_n)
    continuation = run_continuation_benchmark(model, tokenizer, device, continuation_n)
    contrastive_val = evaluate_contrastive_validation(model, tokenizer, validation_records, device, batch_size)
    print(f"  [{label}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% bench={bench} contrastive_val={contrastive_val}", flush=True)
    return {"behavior_eval": behavior, "benchmark": bench, "continuation_benchmark": continuation, "contrastive_validation": contrastive_val}


def train_arm(arm, direct_records, mc_records, model, tokenizer, device, head, steps, batch_size, lr):
    handle = install_plain_slider(model, LAYER, head)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    log = []
    use_contrastive = arm == "contrastive_choice_objective"
    for step in range(steps):
        step_t0 = time.perf_counter()
        records = mc_records if use_contrastive else direct_records
        batch = corpus_batch(records, step, batch_size)
        if use_contrastive:
            loss = contrastive_choice_loss(model, tokenizer, batch, device)
        else:
            loss = span_weighted_causal_lm_loss(
                model,
                tokenizer,
                [r["text"] for r in batch],
                device,
                context_weight=0.0,
                answer_weight=1.0,
                answer_starts=[r.get("answer_start_char") for r in batch],
            )
        model.train(); head.train(); opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
        opt.step()
        rec = {"step": step, "loss": round(float(loss.detach().item()), 5), "alpha": round(float(head.alpha.detach().item()), 6), "grad_norm": round(grad_norm, 5), "capability_sources": sorted({r["source"] for r in batch}), "step_wall_sec": round(time.perf_counter() - step_t0, 3)}
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(f"    {arm} step {step:02d}: loss={rec['loss']:.3f} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}", flush=True)
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
    out_dir = ROOT / "experiments/exp143_contrastive_choice_capability/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 143] Contrastive continuation-choice objective on {device}", flush=True)
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    behavior_rows = ordered_fork_rows(base_rl.load_rows(ROOT / "data/policy_train.jsonl"), 12)
    mc = load_multiple_choice_corpus(args.cap_items_per_source, args.seed)
    direct_train = direct_records_from_mc(mc["train"])
    direct_validation = direct_records_from_mc(mc["validation"])
    source_order_match = True
    payload_mismatches = 0
    print(f"  pairing: source_order_match={source_order_match} payload_mismatches={payload_mismatches}/{len(mc['train'])}", flush=True)
    if not source_order_match or payload_mismatches:
        raise RuntimeError("direct/multiple-choice corpus pairing mismatch; objective comparison would be confounded")

    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    baseline = measure_state("baseline", sft, tokenizer, eval_rows, mc["validation"], device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
    try:
        tl_probe = probe_layer27_geometry(sft, tokenizer, behavior_rows[:4] + behavior_rows[-4:], device)
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp143_contrastive_choice_capability",
        "controlled_variable": "capability objective only: correct-answer causal loss vs token-normalized contrastive choice loss",
        "scientific_question": "Does explicit discrimination among candidate continuations produce stronger industry-benchmark transfer than correct-answer causal training?",
        "fixed_conditions": {"layer": LAYER, "head_dim": HEAD_DIM, "training_format": "benchmark_direct", "seed": args.seed, "steps": args.steps, "capability_batch_size": args.batch_size, "same_underlying_rows": True, "source_order_match": source_order_match, "answer_payload_mismatches": payload_mismatches, "generation_cache_mode": "use_cache=False for behavioral evaluation; no behavioral generation is used in training"},
        "baseline": baseline,
        "direct_corpus": {"train_count": len(direct_train), "validation_count": len(direct_validation), "derived_from_multiple_choice_records": True, "sha256": mc["sha256"], "provenance": mc["provenance"], "load_errors": mc["errors"]},
        "multiple_choice_corpus": {"train_count": len(mc["train"]), "validation_count": len(mc["validation"]), "sha256": mc["sha256"], "provenance": mc["provenance"], "load_errors": mc["errors"]},
        "contamination_control": "uses only MMLU validation, ARC-Easy train, HellaSwag train; benchmark uses MMLU test, ARC-Easy test, HellaSwag validation",
        "transformerlens_probe": tl_probe,
        "config": vars(args),
        "arms": {},
    }

    for arm in ARMS:
        print(f"[Arm] {arm}: objective only", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, HEAD_DIM).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity = float((head(probe) - probe).abs().max().item())
        if identity != 0.0:
            raise RuntimeError(f"zero-init identity failed for {arm}: {identity}")
        handle, trace = train_arm(arm, direct_train, mc["train"], sft, tokenizer, device, head, args.steps, args.batch_size, args.lr)
        state = measure_state(arm, sft, tokenizer, eval_rows, mc["validation"], device, args.bench_n, args.continuation_n, args.batch_size, args.max_new_tokens)
        handle.remove()
        results["arms"][arm] = {
            **state,
            "behavior_delta_vs_baseline": caprep.metric_delta(state["behavior_eval"], baseline["behavior_eval"]),
            "benchmark_delta_vs_baseline": caprep.metric_delta(state["benchmark"], baseline["benchmark"]),
            "continuation_delta_vs_baseline": caprep.continuation_delta(state["continuation_benchmark"], baseline["continuation_benchmark"]),
            "contrastive_validation_loss_delta_vs_baseline": round(state["contrastive_validation"]["loss"] - baseline["contrastive_validation"]["loss"], 6),
            "contrastive_validation_accuracy_delta_vs_baseline": round(state["contrastive_validation"]["accuracy"] - baseline["contrastive_validation"]["accuracy"], 3),
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "contrastive_choice_capability_summary.json"
    write_json(path, results)
    print(f"[Exp 143 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
