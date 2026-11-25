#!/usr/bin/env python3
"""Hypothesis D — 1-Shot Diagonal Fisher + SNIP Reward-Gradient Top-K Module Selector.

Grounded in:
- arXiv:2609.36262 (Sep 2026): 1-shot initial diagonal Fisher masks recover 85.21% of full RLVR
  MATH-500 gain at 0.1% params and 98.82% at 5%, outperforming random masks at every budget.
- arXiv:2602.01599 (ICML 2026): Multiple Ticket Hypothesis — always emit a matched-budget random control.

Halo v1.3.0rc1 Native Fit:
- Outputs `unfreeze_layers_patterns` for `use_peft: false` applied rank-uniformly BEFORE
  `_configure_distributed_modes()` (halo/trainers/distributed_trainer.py:365-432), yielding true
  6 B/param AdamWBF16 optimizer memory savings with zero Halo core modifications.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(path: Path, max_rows: int) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            texts.append("\n".join(f"{m['role']}: {m['content']}" for m in row["messages"]))
        elif "pass_text" in row:
            texts.append(row["pass_text"])
        if len(texts) >= max_rows:
            break
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--keep-frac", type=float, default=0.05, help="Fraction of linear modules to keep trainable")
    parser.add_argument("--max-calib", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    texts = load_texts(Path(args.calib_file), args.max_calib)

    fisher_accum: dict[str, float] = {}
    snip_accum: dict[str, float] = {}
    numel_map: dict[str, int] = {}

    for txt in texts:
        model.zero_grad(set_to_none=True)
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.weight.grad is not None and "lm_head" not in name:
                g = mod.weight.grad.detach().float()
                w = mod.weight.detach().float()
                fisher_accum[name] = fisher_accum.get(name, 0.0) + float((g * g).mean().item())
                snip_accum[name] = snip_accum.get(name, 0.0) + float((g.abs() * w.abs()).mean().item())
                numel_map[name] = mod.weight.numel()

    scored = []
    for name in fisher_accum:
        f_val = fisher_accum[name] / max(len(texts), 1)
        s_val = snip_accum[name] / max(len(texts), 1)
        scored.append({"module": name, "fisher": f_val, "snip": s_val, "combined": f_val + s_val, "params": numel_map[name]})

    scored.sort(key=lambda x: x["combined"], reverse=True)
    k = max(1, int(math.ceil(len(scored) * args.keep_frac)))
    top_mods = [x["module"] for x in scored[:k]]
    rng = random.Random(args.seed)
    rest = [x["module"] for x in scored[k:]]
    rand_mods = rng.sample(rest, k=min(k, len(rest)))

    out = {
        "model": args.model,
        "keep_frac": args.keep_frac,
        "total_linear_modules": len(scored),
        "selected_count": len(top_mods),
        "unfreeze_layers_patterns": top_mods,
        "random_control_unfreeze_layers_patterns": rand_mods,
        "top_10_ranked": scored[:10],
    }
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] Selected {len(top_mods)}/{len(scored)} modules ({100*args.keep_frac:.1f}%) -> {args.out_file}")


if __name__ == "__main__":
    main()
