#!/usr/bin/env python3
"""Experiment 148: legacy benchmark scorer and logit-flattening audit.

Measurement-only controlled audit. No training and no intervention: the same merged
SFT artifact is evaluated while its output logits are scaled toward uniformity. This
isolates whether the legacy real_benchmark scoring rule can manufacture accuracy
movement as distributions flatten, independently of any particular collapsed arm.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import write_json  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402

SCALES = [1.0, 0.5, 0.25, 0.1, 0.0]


class LogitScaledCausalLM(torch.nn.Module):
    """Read-only wrapper that scales logits; scale=0 is exactly uniform."""

    def __init__(self, base_model: torch.nn.Module, scale: float):
        super().__init__()
        self.base_model = base_model
        self.scale = float(scale)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits * self.scale
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def load_items(n: int) -> dict[str, list[dict]]:
    from datasets import load_dataset

    benches: dict[str, list[dict]] = {}
    rows = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
    benches["mmlu"] = [{"context": r["question"], "choices": list(r["choices"]), "answer": r["choices"][int(r["answer"])]} for r in rows]
    rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{n}]")
    benches["arc_easy"] = []
    for r in rows:
        choices = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        benches["arc_easy"].append({"context": r["question"], "choices": choices, "answer": choices[labels.index(r["answerKey"])]})
    rows = load_dataset("Rowan/hellaswag", split=f"validation[:{n}]")
    benches["hellaswag"] = [{"context": r["ctx"], "choices": list(r["endings"]), "answer": r["endings"][int(r["label"])]} for r in rows]
    return benches


def static_length_profile(items: list[dict]) -> dict:
    correct_longest = correct_shortest = correct_unique_longest = correct_unique_shortest = 0
    correct_chars = []
    longest_chars = []
    shortest_chars = []
    correct_length_ranks = []
    for q in items:
        by_chars = sorted(q["choices"], key=len)
        longest = by_chars[-1]
        shortest = by_chars[0]
        correct_longest += int(q["answer"] == longest)
        correct_shortest += int(q["answer"] == shortest)
        lens = [len(c) for c in q["choices"]]
        correct_unique_longest += int(len(q["answer"]) == max(lens) and lens.count(max(lens)) == 1)
        correct_unique_shortest += int(len(q["answer"]) == min(lens) and lens.count(min(lens)) == 1)
        correct_chars.append(len(q["answer"]))
        longest_chars.append(len(longest))
        shortest_chars.append(len(shortest))
        correct_length_ranks.append(sum(len(c) < len(q["answer"]) for c in q["choices"]) / max(len(q["choices"]) - 1, 1))
    n = max(len(items), 1)
    return {
        "n": len(items),
        "correct_is_longest_rate_tie_inclusive": round(100.0 * correct_longest / n, 2),
        "correct_is_shortest_rate_tie_inclusive": round(100.0 * correct_shortest / n, 2),
        "correct_is_unique_longest_rate": round(100.0 * correct_unique_longest / n, 2),
        "correct_is_unique_shortest_rate": round(100.0 * correct_unique_shortest / n, 2),
        "mean_correct_answer_chars": round(sum(correct_chars) / n, 3),
        "mean_longest_answer_chars": round(sum(longest_chars) / n, 3),
        "mean_shortest_answer_chars": round(sum(shortest_chars) / n, 3),
        "mean_correct_length_rank": round(sum(correct_length_ranks) / n, 4),
    }


@torch.no_grad()
def combined_choice_scores(model, tokenizer, device, context: str, choice: str) -> dict:
    """One forward pass yielding exact legacy score plus clean continuation scores."""
    full_text = context + " " + choice
    ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=384).to(device)
    out = model(**ids, labels=ids["input_ids"])
    seq_len = int(ids["input_ids"].shape[1])
    legacy_score = -float(out.loss.item()) * seq_len / max(len(choice), 1)

    prefix_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])
    logits = out.logits[:, :-1, :]
    targets = ids["input_ids"][:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp = log_probs[0, max(prefix_len - 1, 0) :]
    total = float(comp.sum().item())
    n_tokens = int(comp.numel())
    n_bytes = max(len(choice.encode("utf-8")), 1)
    return {
        "choice": choice,
        "legacy_score": legacy_score,
        "continuation_sum": total,
        "continuation_token_mean": total / max(n_tokens, 1),
        "continuation_byte_mean": total / n_bytes,
        "n_tokens": n_tokens,
        "chars": len(choice),
    }


def mean(xs: list[float]) -> float:
    return round(sum(xs) / max(len(xs), 1), 5)


@torch.no_grad()
def evaluate_flattening_scale(model, tokenizer, device, benches: dict[str, list[dict]], scale: float) -> dict:
    scaled = LogitScaledCausalLM(model, scale).to(device)
    scaled.eval()
    out = {}
    for bench, items in benches.items():
        legacy_correct = legacy_longest = legacy_shortest = legacy_unique_longest = legacy_unique_shortest = 0
        sum_correct = token_correct = byte_correct = 0
        sum_shortest = sum_unique_shortest = 0
        legacy_margins = []
        correct_legacy_scores = []
        incorrect_legacy_scores = []
        correct_chars = []
        selected_chars = []
        correct_tokens = []
        selected_tokens = []
        sum_margins = []
        token_margins = []
        for q in items:
            scored = [combined_choice_scores(scaled, tokenizer, device, q["context"], ch) for ch in q["choices"]]
            correct_row = next(x for x in scored if x["choice"] == q["answer"])
            best_legacy = max(scored, key=lambda x: x["legacy_score"])
            best_wrong_legacy = max((x for x in scored if x["choice"] != q["answer"]), key=lambda x: x["legacy_score"])
            best_sum = max(scored, key=lambda x: x["continuation_sum"])
            best_token = max(scored, key=lambda x: x["continuation_token_mean"])
            best_byte = max(scored, key=lambda x: x["continuation_byte_mean"])
            best_wrong_sum = max((x for x in scored if x["choice"] != q["answer"]), key=lambda x: x["continuation_sum"])
            best_wrong_token = max((x for x in scored if x["choice"] != q["answer"]), key=lambda x: x["continuation_token_mean"])
            longest = max(scored, key=lambda x: x["chars"])["choice"]
            shortest = min(scored, key=lambda x: x["chars"])["choice"]
            chars = [x["chars"] for x in scored]
            unique_longest = max(chars) if chars.count(max(chars)) == 1 else None
            unique_shortest = min(chars) if chars.count(min(chars)) == 1 else None
            token_lens = [x["n_tokens"] for x in scored]
            token_shortest = min(scored, key=lambda x: x["n_tokens"])["choice"]
            unique_token_shortest = min(token_lens) if token_lens.count(min(token_lens)) == 1 else None
            legacy_correct += int(best_legacy["choice"] == q["answer"])
            legacy_longest += int(best_legacy["choice"] == longest)
            legacy_shortest += int(best_legacy["choice"] == shortest)
            sum_correct += int(best_sum["choice"] == q["answer"])
            token_correct += int(best_token["choice"] == q["answer"])
            byte_correct += int(best_byte["choice"] == q["answer"])
            sum_shortest += int(best_sum["choice"] == token_shortest)
            legacy_unique_longest += int(unique_longest is not None and best_legacy["chars"] == unique_longest)
            legacy_unique_shortest += int(unique_shortest is not None and best_legacy["chars"] == unique_shortest)
            sum_unique_shortest += int(unique_token_shortest is not None and best_sum["n_tokens"] == unique_token_shortest)
            legacy_margins.append(correct_row["legacy_score"] - best_wrong_legacy["legacy_score"])
            correct_legacy_scores.append(correct_row["legacy_score"])
            incorrect_legacy_scores.extend(x["legacy_score"] for x in scored if x["choice"] != q["answer"])
            correct_chars.append(correct_row["chars"])
            selected_chars.append(best_legacy["chars"])
            correct_tokens.append(correct_row["n_tokens"])
            selected_tokens.append(best_sum["n_tokens"])
            sum_margins.append(correct_row["continuation_sum"] - best_wrong_sum["continuation_sum"])
            token_margins.append(correct_row["continuation_token_mean"] - best_wrong_token["continuation_token_mean"])
        n = max(len(items), 1)
        out[bench] = {
            "n": len(items),
            "legacy_accuracy": round(100.0 * legacy_correct / n, 2),
            "legacy_best_is_longest_rate_tie_inclusive": round(100.0 * legacy_longest / n, 2),
            "legacy_best_is_shortest_rate_tie_inclusive": round(100.0 * legacy_shortest / n, 2),
            "legacy_best_is_unique_longest_rate": round(100.0 * legacy_unique_longest / n, 2),
            "legacy_best_is_unique_shortest_rate": round(100.0 * legacy_unique_shortest / n, 2),
            "mean_correct_minus_best_incorrect_legacy_score": mean(legacy_margins),
            "mean_correct_legacy_score": mean(correct_legacy_scores),
            "mean_incorrect_legacy_score": mean(incorrect_legacy_scores),
            "mean_correct_answer_chars": mean(correct_chars),
            "mean_legacy_selected_answer_chars": mean(selected_chars),
            "continuation_sum_accuracy": round(100.0 * sum_correct / n, 2),
            "continuation_token_norm_accuracy": round(100.0 * token_correct / n, 2),
            "continuation_byte_norm_accuracy": round(100.0 * byte_correct / n, 2),
            "continuation_sum_best_is_shortest_rate_tie_inclusive": round(100.0 * sum_shortest / n, 2),
            "continuation_sum_best_is_unique_shortest_rate": round(100.0 * sum_unique_shortest / n, 2),
            "mean_correct_answer_tokens": mean(correct_tokens),
            "mean_sum_selected_answer_tokens": mean(selected_tokens),
            "mean_correct_minus_best_incorrect_sum": mean(sum_margins),
            "mean_correct_minus_best_incorrect_token": mean(token_margins),
        }
    del scaled
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-n", type=int, default=32)
    parser.add_argument("--scale-count", type=int, default=len(SCALES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp148_legacy_scorer_flattening_audit/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 148] legacy scorer flattening audit on {device}", flush=True)

    model, tokenizer = vec_rl.load_sft(device, torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    benches = load_items(args.diagnostic_n)
    static_profiles = {name: static_length_profile(items) for name, items in benches.items()}
    scales = SCALES[: max(1, min(args.scale_count, len(SCALES)))]
    scale_results = {}
    for scale in scales:
        print(f"[Scale] {scale}", flush=True)
        scale_results[str(scale)] = evaluate_flattening_scale(model, tokenizer, device, benches, scale)
        summary = {k: v["legacy_accuracy"] for k, v in scale_results[str(scale)].items()}
        print(f"  legacy accuracy at scale {scale}: {summary}", flush=True)

    results = {
        "experiment": "exp148_legacy_scorer_flattening_audit",
        "measurement_only": True,
        "scientific_question": "Can exact legacy real_benchmark scoring manufacture benchmark gains as logits flatten toward uniformity?",
        "legacy_scoring_rule": "score = -mean_full_sequence_loss * sequence_token_length / choice_character_length",
        "scales": scales,
        "scale_interpretation": "1.0 is the unmodified merged SFT artifact; 0.0 is exactly uniform next-token logits",
        "config": vars(args),
        "static_choice_length_profiles": static_profiles,
        "scale_results": scale_results,
        "decision_guide": "If uniform or near-uniform logits retain/inflate legacy accuracy while clean token-normalized continuation collapses or ties, legacy benchmark deltas are measurement artifacts rather than capability evidence.",
    }
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "legacy_scorer_flattening_audit_summary.json"
    write_json(path, results)
    print(f"[Exp 148 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
