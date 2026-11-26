#!/usr/bin/env python3
"""Build a broad, benchmark-independent direct-continuation capability corpus.

Unlike capability_train_scaled.jsonl (MMLU validation + ARC-Easy train + HellaSwag train),
this corpus intentionally excludes those benchmark train sources. It uses train splits from
SciQ, CommonsenseQA, QASC, and a length-filtered Dolly closed-QA subset, then decontaminates
against every in-domain and transfer evaluation context used by the OLMoE track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import build_capability_corpus as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUOTAS = {"sciq_train": 1500, "commonsense_qa_train": 1200, "qasc_train": 1200, "dolly_closed_qa_train": 900}


def load_general_eval_indexes() -> dict:
    from datasets import load_dataset

    indexes = base.load_eval_indexes()
    specs = [
        ("arc_challenge", [r["question"] for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")]),
        ("openbookqa", [r["question_stem"] for r in load_dataset("allenai/openbookqa", "main", split="test")]),
        ("winogrande", [r["sentence"] for r in load_dataset("allenai/winogrande", "winogrande_xl", split="validation")]),
        ("truthfulqa_mc1", [r["question"] for r in load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")]),
    ]
    for name, contexts in specs:
        exact, grams = set(), set()
        for context in contexts:
            exact.add(base.text_hash(context))
            grams.update(base.ngram_hashes(context))
        indexes[name] = {"exact": exact, "ngrams": grams, "eval_rows": len(contexts)}
    return indexes


def load_science_mc() -> dict[str, list[dict]]:
    from datasets import load_dataset

    out = {}
    rows = load_dataset("allenai/sciq", split="train")
    out["sciq_train"] = [
        base.direct_record("sciq_train", r["question"], r["correct_answer"], f"sciq_train_{i}", "train")
        for i, r in enumerate(rows)
    ]
    rows = load_dataset("tau/commonsense_qa", split="train")
    vals = []
    for r in rows:
        choices = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        vals.append(base.direct_record("commonsense_qa_train", r["question"], choices[labels.index(r["answerKey"])], str(r["id"]), "train"))
    out["commonsense_qa_train"] = vals
    rows = load_dataset("allenai/qasc", split="train")
    vals = []
    for r in rows:
        choices = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        vals.append(base.direct_record("qasc_train", r["question"], choices[labels.index(r["answerKey"])], str(r["id"]), "train"))
    out["qasc_train"] = vals
    return out


def dolly_words(text: str) -> int:
    return len(text.split())


def load_dolly() -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("databricks/databricks-dolly-15k", split="train")
    out = []
    for i, r in enumerate(rows):
        if r.get("category") != "closed_qa":
            continue
        context = r["instruction"].strip()
        if r.get("context", "").strip():
            context = context + "\n\n" + r["context"].strip()
        answer = r["response"].strip()
        # Keep direct-continuation rows in the same token scale as the existing corpus.
        if dolly_words(answer) > 90 or dolly_words(context) > 260 or not answer:
            continue
        out.append(base.direct_record("dolly_closed_qa_train", context, answer, f"dolly_closedqa_{i}", "train", {"category": r["category"]}))
    return out


def select_clean(records: list[dict], source: str, quota: int, indexes: dict, source_reports: dict, seed: int) -> list[dict]:
    rng = random.Random(seed + sum(ord(c) for c in source))
    rows = list(records)
    rng.shuffle(rows)
    clean = base.filter_unique_clean(rows, indexes, source, source_reports)
    source_reports[source]["kept_before_quota"] = len(clean)
    selected = clean[:quota]
    source_reports[source]["quota"] = quota
    source_reports[source]["kept"] = len(selected)
    source_reports[source]["unused_clean_pool"] = max(len(clean) - len(selected), 0)
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-fraction", type=float, default=0.10)
    args = ap.parse_args()
    print("[general capability corpus] loading evaluation contamination indexes", flush=True)
    indexes = load_general_eval_indexes()
    report = {
        "generator": "scripts/build_capability_general_corpus.py",
        "seed": args.seed,
        "format": "benchmark-independent direct answer-only continuation",
        "normalization": "lowercase, alphanumeric tokens, whitespace canonicalization",
        "ngram_n": base.NGRAM_N,
        "contamination_method": "train contexts exact/hash and 13-gram compared against MMLU/ARC-Easy/HellaSwag plus ARC-Challenge/OpenBookQA/Winogrande/TruthfulQA evaluation contexts",
        "eval_reference": {k: {"rows": v["eval_rows"], "exact_hashes": len(v["exact"]), "ngram_hashes": len(v["ngrams"])} for k, v in indexes.items()},
        "sources": {},
    }
    raw = load_science_mc()
    raw["dolly_closed_qa_train"] = load_dolly()
    by_source = {}
    for source, quota in DEFAULT_QUOTAS.items():
        rows = raw[source]
        report["sources"][source] = base.new_source_report(len(rows))
        print(f"[general capability corpus] filtering {source}: loaded={len(rows)} quota={quota}", flush=True)
        by_source[source] = select_clean(rows, source, quota, indexes, report["sources"], args.seed)
    train, validation = base.round_robin_split(by_source, args.seed, args.validation_fraction)
    write_jsonl(ROOT / "data/capability_train_general_v1.jsonl", train)
    write_jsonl(ROOT / "data/capability_validation_general_v1.jsonl", validation)
    full = train + validation
    write_jsonl(ROOT / "data/capability_corpus_general_v1.jsonl", full)
    for source in report["sources"]:
        report["sources"][source]["matches_by_eval_benchmark"] = dict(report["sources"][source]["matches_by_eval_benchmark"])
    report["counts"] = {
        "full_rows": len(full),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_source_counts": dict(Counter(r["source"] for r in train)),
        "validation_source_counts": dict(Counter(r["source"] for r in validation)),
    }
    report["file_sha256"] = {
        "capability_train_general_v1.jsonl": hashlib.sha256((ROOT / "data/capability_train_general_v1.jsonl").read_bytes()).hexdigest(),
        "capability_validation_general_v1.jsonl": hashlib.sha256((ROOT / "data/capability_validation_general_v1.jsonl").read_bytes()).hexdigest(),
        "capability_corpus_general_v1.jsonl": hashlib.sha256((ROOT / "data/capability_corpus_general_v1.jsonl").read_bytes()).hexdigest(),
    }
    report["passed"] = all(report["sources"][s]["kept"] == q for s, q in DEFAULT_QUOTAS.items())
    (ROOT / "data/capability_general_corpus_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2), flush=True)
    print(f"[general capability corpus] passed={report['passed']}", flush=True)


if __name__ == "__main__":
    main()
