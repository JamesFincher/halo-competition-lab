"""Continuation-scored MMLU/ARC-Easy/HellaSwag benchmark.

This is a measurement-companion to real_benchmark.py. It scores only answer
continuation tokens conditioned on the question/context instead of including the
shared prompt loss, which makes small capability/collateral changes visible. It
reports raw summed-logprob accuracy plus token- and byte-normalized accuracy.
"""
from __future__ import annotations

import os
import torch


@torch.no_grad()
def continuation_logprob(model, tok, device, context: str, choice: str, max_length: int = 384) -> dict:
    full_text = context + " " + choice
    full = tok(full_text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False).to(device)
    prefix_len = len(tok(context, add_special_tokens=False)["input_ids"])
    full_len = int(full["input_ids"].shape[1])
    if full_len <= prefix_len or prefix_len < 1:
        return {"sum": -1e18, "token_mean": -1e18, "byte_mean": -1e18, "n_tokens": 0}
    logits = model(**full).logits[:, :-1, :]
    targets = full["input_ids"][:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp = log_probs[0, max(prefix_len - 1, 0) :]
    total = float(comp.sum().item())
    n_tokens = int(comp.numel())
    n_bytes = max(len(choice.encode("utf-8")), 1)
    return {
        "sum": total,
        "token_mean": total / max(n_tokens, 1),
        "byte_mean": total / n_bytes,
        "n_tokens": n_tokens,
    }


@torch.no_grad()
def continuation_logprobs_batch(model, tok, device, context: str, choices: list[str], max_length: int = 384, batch_size: int = 4) -> list[dict]:
    """Batched equivalent of continuation_logprob for same-context answer choices.

    Right-padded positions are masked and excluded explicitly. Batching preserves the
    causal prefix/answer token positions while reducing benchmark scoring forwards by
    roughly the choice batch size on the large-model track.
    """
    prefix_len = len(tok(context, add_special_tokens=False)["input_ids"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoded = []
    for choice in choices:
        ids = tok(context + " " + choice, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        encoded.append((choice, ids))
    out = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        width = max(len(ids) for _, ids in chunk)
        input_ids = torch.tensor([ids + [pad_id] * (width - len(ids)) for _, ids in chunk], device=device)
        attention_mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for _, ids in chunk], device=device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for row, (choice, ids) in enumerate(chunk):
            full_len = len(ids)
            if full_len <= prefix_len or prefix_len < 1:
                out.append({"sum": -1e18, "token_mean": -1e18, "byte_mean": -1e18, "n_tokens": 0})
                continue
            comp = log_probs[row, max(prefix_len - 1, 0):full_len - 1]
            total = float(comp.sum().item())
            n_tokens = int(comp.numel())
            n_bytes = max(len(choice.encode("utf-8")), 1)
            out.append({"sum": total, "token_mean": total / max(n_tokens, 1), "byte_mean": total / n_bytes, "n_tokens": n_tokens})
    return out


@torch.no_grad()
def score_mc_continuation(model, tok, device, items) -> dict:
    correct_sum = correct_token = correct_byte = 0
    details = []
    for q in items:
        choice_lps = continuation_logprobs_batch(model, tok, device, q["context"], q["choices"])
        scored = list(zip(q["choices"], choice_lps))
        best_sum = max(scored, key=lambda x: x[1]["sum"])[0]
        best_token = max(scored, key=lambda x: x[1]["token_mean"])[0]
        best_byte = max(scored, key=lambda x: x[1]["byte_mean"])[0]
        correct_sum += int(best_sum == q["answer"])
        correct_token += int(best_token == q["answer"])
        correct_byte += int(best_byte == q["answer"])
        details.append({"context": q["context"][:160], "answer": q["answer"], "best_sum": best_sum, "best_token": best_token, "best_byte": best_byte})
    n = max(len(items), 1)
    return {
        "sum_acc": round(100.0 * correct_sum / n, 2),
        "token_norm_acc": round(100.0 * correct_token / n, 2),
        "byte_norm_acc": round(100.0 * correct_byte / n, 2),
        "n": len(items),
        "details": details,
    }


def run_continuation_benchmark(model, tok, device, n: int = 100, include_details: bool = False) -> dict:
    """Score an already-loaded causal LM by answer-continuation likelihood only."""
    if os.environ.get("HALO_SMOKE_TEST") == "1":
        n = min(n, 2)
    from datasets import load_dataset

    res = {}
    try:
        rows = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
        items = [{"context": r["question"], "choices": list(r["choices"]), "answer": r["choices"][r["answer"]]} for r in rows]
        out = score_mc_continuation(model, tok, device, items)
        if not include_details:
            out.pop("details", None)
        res["mmlu"] = out
        print(f"  continuation MMLU sum/token/byte: {out['sum_acc']}/{out['token_norm_acc']}/{out['byte_norm_acc']}%", flush=True)
    except Exception as exc:
        res["mmlu"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  continuation MMLU failed: {str(exc)[:80]}", flush=True)
    try:
        rows = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{n}]")
        items = []
        for r in rows:
            choices = list(r["choices"]["text"])
            labels = list(r["choices"]["label"])
            items.append({"context": r["question"], "choices": choices, "answer": choices[labels.index(r["answerKey"])]})
        out = score_mc_continuation(model, tok, device, items)
        if not include_details:
            out.pop("details", None)
        res["arc_easy"] = out
        print(f"  continuation ARC-Easy sum/token/byte: {out['sum_acc']}/{out['token_norm_acc']}/{out['byte_norm_acc']}%", flush=True)
    except Exception as exc:
        res["arc_easy"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  continuation ARC failed: {str(exc)[:80]}", flush=True)
    try:
        rows = load_dataset("Rowan/hellaswag", split=f"validation[:{n}]")
        items = [{"context": r["ctx"], "choices": list(r["endings"]), "answer": r["endings"][int(r["label"])]} for r in rows]
        out = score_mc_continuation(model, tok, device, items)
        if not include_details:
            out.pop("details", None)
        res["hellaswag"] = out
        print(f"  continuation HellaSwag sum/token/byte: {out['sum_acc']}/{out['token_norm_acc']}/{out['byte_norm_acc']}%", flush=True)
    except Exception as exc:
        res["hellaswag"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  continuation HellaSwag failed: {str(exc)[:80]}", flush=True)
    return res

def run_transfer_benchmark(model, tok, device, n: int = 50, include_details: bool = False) -> dict:
    """Score held-out transfer benchmarks whose train splits are absent from the capability corpus.

    These are still continuation-scored rather than generation benchmarks. They test
    whether benchmark-specific training generalizes beyond MMLU/ARC-Easy/HellaSwag
    in-domain formats instead of merely measuring train-split distribution adaptation.
    """
    if os.environ.get("HALO_SMOKE_TEST") == "1":
        n = min(n, 2)
    from datasets import load_dataset

    res = {}

    def finish(name: str, items):
        out = score_mc_continuation(model, tok, device, items)
        if not include_details:
            out.pop("details", None)
        res[name] = out
        print(f"  transfer {name} sum/token/byte: {out['sum_acc']}/{out['token_norm_acc']}/{out['byte_norm_acc']}%", flush=True)

    try:
        rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"test[:{n}]")
        items = []
        for r in rows:
            choices = list(r["choices"]["text"])
            labels = list(r["choices"]["label"])
            items.append({"context": r["question"], "choices": choices, "answer": choices[labels.index(r["answerKey"])]})
        finish("arc_challenge", items)
    except Exception as exc:
        res["arc_challenge"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  transfer ARC-Challenge failed: {str(exc)[:80]}", flush=True)
    try:
        rows = load_dataset("allenai/openbookqa", "main", split=f"test[:{n}]")
        items = []
        for r in rows:
            choices = list(r["choices"]["text"])
            labels = list(r["choices"]["label"])
            items.append({"context": r["question_stem"], "choices": choices, "answer": choices[labels.index(r["answerKey"])]})
        finish("openbookqa", items)
    except Exception as exc:
        res["openbookqa"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  transfer OpenBookQA failed: {str(exc)[:80]}", flush=True)
    try:
        rows = load_dataset("allenai/winogrande", "winogrande_xl", split=f"validation[:{n}]")
        items = [
            {"context": r["sentence"], "choices": [r["option1"], r["option2"]], "answer": r["option1"] if str(r["answer"]) == "1" else r["option2"]}
            for r in rows
        ]
        finish("winogrande", items)
    except Exception as exc:
        res["winogrande"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  transfer Winogrande failed: {str(exc)[:80]}", flush=True)
    try:
        rows = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=f"validation[:{n}]")
        items = []
        for r in rows:
            choices = list(r["mc1_targets"]["choices"])
            labels = list(r["mc1_targets"]["labels"])
            items.append({"context": r["question"], "choices": choices, "answer": choices[labels.index(1)]})
        finish("truthfulqa_mc1", items)
    except Exception as exc:
        res["truthfulqa_mc1"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  transfer TruthfulQA MC1 failed: {str(exc)[:80]}", flush=True)
    return res

