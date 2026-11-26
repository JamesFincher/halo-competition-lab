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
def score_mc_continuation(model, tok, device, items) -> dict:
    correct_sum = correct_token = correct_byte = 0
    details = []
    for q in items:
        scored = []
        for ch in q["choices"]:
            lp = continuation_logprob(model, tok, device, q["context"], ch)
            scored.append((ch, lp))
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
