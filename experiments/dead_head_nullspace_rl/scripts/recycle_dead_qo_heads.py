#!/usr/bin/env python3
"""Experiment 5: QO-Only Zero-Init Dead-Head Recycling (Zero Architecture / KV-Cache Changes).

Identifies the bottom-K% lowest-impact query heads across all transformer layers using:
- `o_proj` per-head column Frobenius norm (proxy for head write magnitude into residual stream)
- optional single-head output variance on calibration prompts

Surgically edits ONLY:
- `q_proj` rows for dead Q-heads -> small orthogonal re-initialization
- `o_proj` columns for dead Q-heads -> EXACT ZERO (guarantees bit-identical step-0 logprobs & KL=0!)
NEVER modifies `k_proj` or `v_proj` (protects 4:1 / 8:1 GQA shared KV heads).
If the model has `sinks` (e.g. gpt-oss-20b), resets `sinks[h]` to 0.0 for recycled heads offline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    for path in ("model.layers", "transformer.h", "language_model.model.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise RuntimeError("Could not locate transformer layers.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--recycle-frac", type=float, default=0.05, help="Fraction of Q-heads to recycle (default 5%)")
    parser.add_argument("--q-init-scale", type=float, default=0.02, help="Scale for orthogonal q_proj re-init")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    layers = find_layers(model)

    num_q_heads = int(getattr(model.config, "num_attention_heads"))
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // num_q_heads))

    head_scores = []
    for li, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "o_proj") or not hasattr(attn, "q_proj"):
            continue
        o_w = attn.o_proj.weight.detach().float()  # [hidden, num_q_heads * head_dim]
        if o_w.shape[1] != num_q_heads * head_dim:
            continue
        for h in range(num_q_heads):
            c0, c1 = h * head_dim, (h + 1) * head_dim
            col_norm = float(o_w[:, c0:c1].norm().item())
            head_scores.append({"layer": li, "head": h, "o_proj_col_norm": round(col_norm, 6)})

    head_scores.sort(key=lambda x: x["o_proj_col_norm"])
    k = max(1, int(math.ceil(len(head_scores) * args.recycle_frac)))
    selected = head_scores[:k]

    recycled_by_layer: dict[int, list[int]] = {}
    for item in selected:
        recycled_by_layer.setdefault(item["layer"], []).append(item["head"])

    with torch.no_grad():
        for li, heads in recycled_by_layer.items():
            attn = layers[li].self_attn
            q_w = attn.q_proj.weight
            o_w = attn.o_proj.weight
            for h in heads:
                s0, s1 = h * head_dim, (h + 1) * head_dim
                # Orthogonal re-init for Q-head rows; exact zero for O-proj cols (step-0 bit-identical!)
                rand_block = torch.empty((head_dim, q_w.shape[1]), dtype=torch.float32, device=q_w.device)
                torch.nn.init.orthogonal_(rand_block, gain=args.q_init_scale)
                q_w.data[s0:s1, :].copy_(rand_block.to(q_w.dtype))
                o_w.data[:, s0:s1].zero_()
                if hasattr(attn, "sinks") and attn.sinks is not None and h < attn.sinks.numel():
                    attn.sinks.data[h] = 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "num_q_heads_per_layer": num_q_heads,
        "head_dim": head_dim,
        "total_q_heads_profiled": len(head_scores),
        "recycled_count": len(selected),
        "recycled_by_layer": {str(k): sorted(v) for k, v in sorted(recycled_by_layer.items())},
        "kv_heads_modified": 0,
    }
    if args.save_checkpoint:
        ckpt_dir = out_dir / "checkpoint_recycled"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        report["saved_checkpoint"] = str(ckpt_dir)

    (out_dir / "dead_qo_heads.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Recycled {len(selected)}/{len(head_scores)} dead QO heads (0 KV heads touched) -> {out_dir}")


if __name__ == "__main__":
    main()
