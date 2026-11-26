"""Capability-objective data and loss utilities for slider-head experiments.

Uses only train/validation splits of benchmark-style datasets. The benchmark harness
evaluates MMLU test, ARC-Easy test, and HellaSwag validation, so these corpora are
split-disjoint from the measured items by construction.
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable

import torch
import torch.nn.functional as F


def _lettered_qa(question: str, choices: list[str], answer_index: int) -> str:
    letters = ["A", "B", "C", "D", "E", "F"]
    rows = [question.strip()]
    for i, choice in enumerate(choices):
        rows.append(f"{letters[i]}. {choice.strip()}")
    rows.append(f"Answer: {choices[answer_index].strip()}")
    return "\n".join(rows)


def _records(source: str, texts: Iterable[str]) -> list[dict]:
    return [{"source": source, "text": text} for text in texts]


def _fallback_records() -> dict[str, list[dict]]:
    texts = [
        "Question: If a train travels 45 miles in one hour, how far does it travel in three hours at the same speed?\nA. 45 miles\nB. 90 miles\nC. 135 miles\nD. 180 miles\nAnswer: 135 miles",
        "Question: Which process allows plants to convert light energy into chemical energy?\nA. respiration\nB. photosynthesis\nC. fermentation\nD. digestion\nAnswer: photosynthesis",
        "Question: A store reduces a $40 item by 25%. What is the sale price?\nA. $10\nB. $25\nC. $30\nD. $35\nAnswer: $30",
        "Passage: Water is heated in a closed pot. As the temperature rises, more molecules escape the liquid phase. The vapor pressure therefore increases.",
        "Passage: In a fair coin toss, heads and tails are equally likely. Across many independent tosses, the fraction of heads usually approaches one half.",
        "Question: Which gas makes up most of Earth's atmosphere?\nA. oxygen\nB. carbon dioxide\nC. nitrogen\nD. hydrogen\nAnswer: nitrogen",
    ]
    return {"synthetic_fallback": _records("synthetic_fallback", texts)}


def load_benchmark_style_train_corpus(per_source: int = 64, seed: int = 42) -> dict:
    """Load split-disjoint capability-training text and a held-out validation slice.

    Sources:
    - MMLU validation (benchmark scores MMLU test)
    - ARC-Easy train (benchmark scores ARC-Easy test)
    - HellaSwag train (benchmark scores HellaSwag validation)
    """
    per_source = max(int(per_source), 1)
    by_source: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    provenance: list[dict] = []
    try:
        from datasets import load_dataset

        try:
            rows = load_dataset("cais/mmlu", "all", split=f"validation[:{per_source}]")
            texts = [_lettered_qa(r["question"], list(r["choices"]), int(r["answer"])) for r in rows]
            by_source["mmlu_validation"] = _records("mmlu_validation", texts)
            provenance.append({"source": "mmlu_validation", "dataset": "cais/mmlu/all", "split": "validation", "eval_split_excluded": "test", "rows_loaded": len(texts)})
        except Exception as exc:
            errors["mmlu_validation"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"train[:{per_source}]")
            texts = []
            for r in rows:
                choices = list(r["choices"]["text"])
                labels = list(r["choices"]["label"])
                answer_index = labels.index(r["answerKey"])
                texts.append(_lettered_qa(r["question"], choices, answer_index))
            by_source["arc_easy_train"] = _records("arc_easy_train", texts)
            provenance.append({"source": "arc_easy_train", "dataset": "allenai/ai2_arc/ARC-Easy", "split": "train", "eval_split_excluded": "test", "rows_loaded": len(texts)})
        except Exception as exc:
            errors["arc_easy_train"] = f"{type(exc).__name__}: {exc}"
        try:
            rows = load_dataset("Rowan/hellaswag", split=f"train[:{per_source}]")
            texts = []
            for r in rows:
                answer = r["endings"][int(r["label"])]
                texts.append(f"Passage: {r['ctx'].strip()}\nContinuation: {answer.strip()}")
            by_source["hellaswag_train"] = _records("hellaswag_train", texts)
            provenance.append({"source": "hellaswag_train", "dataset": "Rowan/hellaswag", "split": "train", "eval_split_excluded": "validation", "rows_loaded": len(texts)})
        except Exception as exc:
            errors["hellaswag_train"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        errors["datasets_import"] = f"{type(exc).__name__}: {exc}"

    if not any(by_source.values()):
        by_source = _fallback_records()
        provenance.append({"source": "synthetic_fallback", "dataset": "embedded fallback", "split": "not applicable", "eval_split_excluded": "all benchmark items", "rows_loaded": sum(len(v) for v in by_source.values())})
        errors["fallback_used"] = "no external train/validation corpus loaded"

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

    rng = random.Random(seed)
    train = round_robin(train_parts)
    validation = round_robin(validation_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    fingerprint_payload = "\n".join(f"{r['source']}::{r['text']}" for r in train + validation)
    return {
        "train": train,
        "validation": validation,
        "provenance": provenance,
        "errors": errors,
        "sha256": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
    }


def causal_lm_loss(model: torch.nn.Module, tokenizer, texts: list[str], device: torch.device, max_length: int = 384) -> torch.Tensor:
    """Token-normalized causal language-model loss on a batch of plain texts."""
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_lm_loss(model: torch.nn.Module, tokenizer, records: list[dict], device: torch.device, batch_size: int = 4, max_length: int = 384) -> float:
    if not records:
        return 0.0
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(records), batch_size):
        texts = [r["text"] for r in records[start : start + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid = shift_labels.view(-1) != -100
        total_loss += float(losses[valid].sum().item())
        total_tokens += int(valid.sum().item())
    return round(total_loss / max(total_tokens, 1), 6)


def corpus_batch(records: list[dict], step: int, batch_size: int) -> list[dict]:
    if not records:
        raise ValueError("capability training corpus is empty")
    start = (step * batch_size) % len(records)
    return [records[(start + i) % len(records)] for i in range(batch_size)]
