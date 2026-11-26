#!/usr/bin/env python3
"""Build the scaled decontaminated benchmark-direct capability corpus.

Sources are train/validation splits only; evaluation uses disjoint benchmark splits.
The generated rows use exp141's direct continuation format: context + space + answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGRAM_N = 13
TARGET_TOTAL = 5280


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def text_hash(text: str) -> int:
    return digest(normalize_text(text))


def ngram_hashes(text: str, n: int = NGRAM_N) -> set[int]:
    words = normalize_text(text).split()
    return {digest(" ".join(words[i:i+n])) for i in range(max(0, len(words)-n+1))}


def direct_record(source: str, context: str, answer: str, dataset_id: str, split: str, extra: dict | None = None) -> dict:
    context = " ".join(context.strip().split())
    answer = " ".join(answer.strip().split())
    rec = {
        "source": source,
        "dataset_id": dataset_id,
        "split": split,
        "context": context,
        "answer_text": answer,
        "text": f"{context} {answer}",
        "answer_start_char": len(context) + 1,
        "format": "benchmark_direct_answer_only",
    }
    if extra:
        rec.update(extra)
    return rec


def load_eval_indexes() -> dict:
    from datasets import load_dataset

    indexes = {}
    specs = [
        ("mmlu", lambda: [r["question"] for r in load_dataset("cais/mmlu", "all", split="test")]),
        ("arc_easy", lambda: [r["question"] for r in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")]),
        ("hellaswag", lambda: [r["ctx"] for r in load_dataset("Rowan/hellaswag", split="validation")]),
    ]
    for name, loader in specs:
        contexts = loader()
        exact = set()
        grams = set()
        for context in contexts:
            exact.add(text_hash(context))
            grams.update(ngram_hashes(context))
        indexes[name] = {"exact": exact, "ngrams": grams, "eval_rows": len(contexts)}
    return indexes


def contamination(rec: dict, indexes: dict) -> dict:
    context = rec["context"]
    exact_hash = text_hash(context)
    grams = ngram_hashes(context)
    exact_matches = [name for name, idx in indexes.items() if exact_hash in idx["exact"]]
    ngram_matches = [name for name, idx in indexes.items() if grams & idx["ngrams"]]
    return {
        "exact_matches": exact_matches,
        "ngram_matches": ngram_matches,
        "contaminated": bool(exact_matches or ngram_matches),
    }


def filter_unique_clean(records: list[dict], indexes: dict, source: str, report: dict, examples_limit: int = 5) -> list[dict]:
    seen = set()
    clean = []
    for rec in records:
        h = text_hash(rec["text"])
        if h in seen:
            report[source]["dropped_duplicate"] += 1
            continue
        seen.add(h)
        cont = contamination(rec, indexes)
        if cont["contaminated"]:
            if cont["exact_matches"]:
                report[source]["dropped_exact_context_match"] += 1
            if cont["ngram_matches"]:
                report[source]["dropped_ngram_overlap"] += 1
            for bench in set(cont["exact_matches"] + cont["ngram_matches"]):
                report[source]["matches_by_eval_benchmark"][bench] += 1
            if len(report[source]["contamination_examples"]) < examples_limit:
                report[source]["contamination_examples"].append({
                    "dataset_id": rec["dataset_id"],
                    "exact_matches": cont["exact_matches"],
                    "ngram_matches": cont["ngram_matches"],
                    "context_prefix": rec["context"][:180],
                })
            continue
        clean.append(rec)
    report[source]["kept"] = len(clean)
    return clean


def load_mmlu() -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("cais/mmlu", "all", split="validation")
    return [
        direct_record(
            "mmlu_validation",
            r["question"],
            list(r["choices"])[int(r["answer"])],
            f"mmlu_validation_{i}_{r.get('subject','unknown')}",
            "validation",
            {"subject": r.get("subject", "unknown"), "correct_choice_index": int(r["answer"])},
        )
        for i, r in enumerate(rows)
    ]


def load_arc() -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    out = []
    for r in rows:
        choices = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        ai = labels.index(r["answerKey"])
        out.append(direct_record(
            "arc_easy_train",
            r["question"],
            choices[ai],
            str(r["id"]),
            "train",
            {"correct_choice_label": r["answerKey"], "correct_choice_index": ai},
        ))
    return out


def hellaswag_record(r: dict) -> dict:
    label = int(r["label"])
    return direct_record(
        "hellaswag_train",
        r["ctx"],
        list(r["endings"])[label],
        str(r.get("source_id") or f"hellaswag_train_{r['ind']}") ,
        "train",
        {"activity_label": r.get("activity_label", "unknown"), "correct_choice_index": label, "split_type": r.get("split_type", "unknown")},
    )


def select_hellaswag(target: int, indexes: dict, report: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows = list(load_dataset("Rowan/hellaswag", split="train"))
    report["hellaswag_train"]["loaded"] = len(rows)
    by_label = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)
    rng = random.Random(seed)
    for vals in by_label.values():
        rng.shuffle(vals)
    quotas = {0: target // 4, 1: target // 4, 2: target // 4, 3: target // 4}
    for label in range(target % 4):
        quotas[label] += 1
    selected = []
    seen = set()
    positions = defaultdict(int)
    label_counts = Counter()
    # Round-robin labels; if a candidate is contaminated or duplicated, keep scanning that label.
    while sum(label_counts.values()) < target:
        progressed = False
        for label in sorted(quotas):
            if label_counts[label] >= quotas[label]:
                continue
            vals = by_label[label]
            while positions[label] < len(vals):
                raw = vals[positions[label]]
                positions[label] += 1
                rec = hellaswag_record(raw)
                h = text_hash(rec["text"])
                if h in seen:
                    report["hellaswag_train"]["dropped_duplicate"] += 1
                    continue
                cont = contamination(rec, indexes)
                if cont["contaminated"]:
                    if cont["exact_matches"]:
                        report["hellaswag_train"]["dropped_exact_context_match"] += 1
                    if cont["ngram_matches"]:
                        report["hellaswag_train"]["dropped_ngram_overlap"] += 1
                    for bench in set(cont["exact_matches"] + cont["ngram_matches"]):
                        report["hellaswag_train"]["matches_by_eval_benchmark"][bench] += 1
                    if len(report["hellaswag_train"]["contamination_examples"]) < 5:
                        report["hellaswag_train"]["contamination_examples"].append({
                            "dataset_id": rec["dataset_id"],
                            "exact_matches": cont["exact_matches"],
                            "ngram_matches": cont["ngram_matches"],
                            "context_prefix": rec["context"][:180],
                        })
                    continue
                seen.add(h)
                selected.append(rec)
                label_counts[label] += 1
                progressed = True
                break
        if not progressed:
            break
    report["hellaswag_train"]["kept"] = len(selected)
    report["hellaswag_train"]["label_balance"] = dict(sorted(label_counts.items()))
    report["hellaswag_train"]["scanned_train_rows"] = sum(positions.values())
    return selected


def round_robin_split(by_source: dict[str, list[dict]], seed: int, validation_fraction: float) -> tuple[list[dict], list[dict]]:
    train_parts = {}
    validation_parts = {}
    for source, items in by_source.items():
        n_val = max(1, int(round(len(items) * validation_fraction))) if len(items) >= 10 else max(0, len(items) - 1)
        validation_parts[source] = items[-n_val:] if n_val else []
        train_parts[source] = items[: len(items) - n_val] if n_val else list(items)

    def rr(parts):
        remaining = {k: list(v) for k, v in parts.items() if v}
        out = []
        while remaining:
            for source in sorted(list(remaining)):
                if remaining[source]:
                    out.append(remaining[source].pop(0))
                if not remaining[source]:
                    del remaining[source]
        return out

    train = rr(train_parts)
    validation = rr(validation_parts)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_source_report(loaded: int = 0) -> dict:
    return {
        "loaded": loaded,
        "kept": 0,
        "dropped_duplicate": 0,
        "dropped_exact_context_match": 0,
        "dropped_ngram_overlap": 0,
        "matches_by_eval_benchmark": Counter(),
        "contamination_examples": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation-fraction", type=float, default=0.10)
    ap.add_argument("--full-out", type=str, default="data/capability_corpus_scaled.jsonl")
    ap.add_argument("--train-out", type=str, default="data/capability_train_scaled.jsonl")
    ap.add_argument("--validation-out", type=str, default="data/capability_validation_scaled.jsonl")
    ap.add_argument("--report-out", type=str, default="data/capability_corpus_report.json")
    args = ap.parse_args()

    print("[capability corpus] loading eval contamination indexes", flush=True)
    indexes = load_eval_indexes()
    report = {
        "generator": "scripts/build_capability_corpus.py",
        "seed": args.seed,
        "target_total": args.target_total,
        "format": "exp141 benchmark-direct: context + single space + answer",
        "normalization": "lowercase, alphanumeric tokens, whitespace canonicalization",
        "ngram_n": NGRAM_N,
        "contamination_method": "train contexts are exact/hash and 13-gram compared against evaluation question/passage contexts; answer choices are excluded because shared short answer strings are not item leakage",
        "dedupe_method": "exact digest of normalized direct context+answer text",
        "eval_reference": {k: {"rows": v["eval_rows"], "exact_hashes": len(v["exact"]), "ngram_hashes": len(v["ngrams"])} for k, v in indexes.items()},
        "sources": {},
    }

    print("[capability corpus] loading MMLU validation and ARC-Easy train", flush=True)
    mmlu = load_mmlu()
    arc = load_arc()
    report["sources"]["mmlu_validation"] = new_source_report(len(mmlu))
    report["sources"]["arc_easy_train"] = new_source_report(len(arc))
    mmlu_clean = filter_unique_clean(mmlu, indexes, "mmlu_validation", report["sources"])
    arc_clean = filter_unique_clean(arc, indexes, "arc_easy_train", report["sources"])

    hella_target = max(args.target_total - len(mmlu_clean) - len(arc_clean), 0)
    report["sources"]["hellaswag_train"] = new_source_report(39905)
    report["sources"]["hellaswag_train"]["target_after_mmlu_arc"] = hella_target
    print(f"[capability corpus] selecting balanced HellaSwag train rows target={hella_target}", flush=True)
    hella_clean = select_hellaswag(hella_target, indexes, report["sources"], args.seed)

    by_source = {"mmlu_validation": mmlu_clean, "arc_easy_train": arc_clean, "hellaswag_train": hella_clean}
    full = [rec for source in sorted(by_source) for rec in by_source[source]]
    # Deduplicate globally as a final invariant; source-level dedupe already makes collisions rare.
    seen = set(); deduped = []
    for rec in full:
        h = text_hash(rec["text"])
        if h not in seen:
            seen.add(h); deduped.append(rec)
    full = deduped
    train, validation = round_robin_split(by_source, args.seed, args.validation_fraction)

    full_path = ROOT / args.full_out
    train_path = ROOT / args.train_out
    validation_path = ROOT / args.validation_out
    report_path = ROOT / args.report_out
    write_jsonl(full_path, full)
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    source_counts = {k: len(v) for k, v in by_source.items()}
    train_counts = Counter(r["source"] for r in train)
    val_counts = Counter(r["source"] for r in validation)
    all_text_hashes = [text_hash(r["text"]) for r in full]
    report.update({
        "full_path": args.full_out,
        "train_path": args.train_out,
        "validation_path": args.validation_out,
        "full_sha256": sha256_file(full_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "full_rows": len(full),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "source_counts": source_counts,
        "train_source_counts": dict(train_counts),
        "validation_source_counts": dict(val_counts),
        "unique_normalized_texts": len(set(all_text_hashes)),
        "duplicate_normalized_texts": len(all_text_hashes) - len(set(all_text_hashes)),
        "hellaswag_label_balance": report["sources"]["hellaswag_train"].get("label_balance", {}),
        "hellaswag_activity_count": len({r.get("activity_label", "unknown") for r in by_source["hellaswag_train"]}),
    })
    # JSON-safe counters.
    for source_report in report["sources"].values():
        source_report["matches_by_eval_benchmark"] = dict(source_report["matches_by_eval_benchmark"])
    report["passed"] = (
        len(full) >= int(args.target_total * 0.98)
        and report["unique_normalized_texts"] == len(full)
        and len(train) + len(validation) == len(full)
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "full_rows": len(full),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "source_counts": source_counts,
        "hellaswag_label_balance": report["hellaswag_label_balance"],
        "dropped": {k: {x: v[x] for x in ["dropped_duplicate", "dropped_exact_context_match", "dropped_ngram_overlap"]} for k, v in report["sources"].items()},
    }, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
