#!/usr/bin/env python3
"""Variants V4 (Dual-Edit Synergy, arXiv:2611.33133) & V6 (Iterative Gated SELFWRITE, arXiv:2611.29513).

- `--mode dual_bias_edit` (V4):
  Implements the Nov 24, 2026 'A Bias in Disguise' finding: takes an already-edited (or base)
  checkpoint, computes gradients on a tiny set of correct examples (e.g. 16-50), selects the
  top-K largest-gradient weights concentrated in the first 3 layers (layers 0, 1, 2), and applies
  a half-strength sparse gradient edit (`--strength 0.5`) that complements mid-layer weight-space edits.

- `--mode iterative_gated` (V6):
  Implements SELFWRITE (arXiv:2611.29513): iteratively proposes rank-1 contrastive MLP down_proj
  projection edits across `--rounds` generations, evaluating each candidate on a hidden gate set
  and ONLY merging edits that strictly reduce gate cross-entropy loss on correct completions.
"""

from __future__ import annotations

import argparse
import copy
import json
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


def load_texts(path: Path, max_rows: int) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "pass_text" in row:
            texts.append(row["pass_text"])
        elif "messages" in row:
            msgs = row["messages"]
            texts.append("\n".join(f"{m['role']}: {m['content']}" for m in msgs))
        if len(texts) >= max_rows:
            break
    return texts


@torch.no_grad()
def eval_gate_loss(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    gate_texts: list[str],
    device: torch.device,
    max_length: int,
) -> float:
    total_loss = 0.0
    for txt in gate_texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(**enc, labels=enc["input_ids"])
        total_loss += float(out.loss.item())
    return total_loss / max(len(gate_texts), 1)


def run_dual_bias_edit(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_texts: list[str],
    gate_texts: list[str],
    device: torch.device,
    max_length: int,
    top_k_weights: int,
    step_size: float,
    strength: float,
    early_layers: int = 3,
) -> dict:
    layers = find_layers(model)
    target_layers = list(range(min(early_layers, len(layers))))
    loss_before = eval_gate_loss(model, tokenizer, gate_texts, device, max_length)

    model.zero_grad(set_to_none=True)
    for txt in train_texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(**enc, labels=enc["input_ids"])
        (out.loss / max(len(train_texts), 1)).backward()

    # Collect absolute gradients across early layers (0..early_layers-1)
    grads_flat = []
    param_refs = []
    for li in target_layers:
        for name, p in layers[li].named_parameters():
            if p.grad is not None and p.ndim >= 2:
                g = p.grad.detach().float()
                grads_flat.append(g.abs().reshape(-1))
                param_refs.append((li, name, p, g))

    all_abs = torch.cat(grads_flat)
    k = min(top_k_weights, all_abs.numel())
    threshold = torch.topk(all_abs, k=k).values[-1].item()

    edited_count = 0
    with torch.no_grad():
        for _li, _name, p, g in param_refs:
            mask = g.abs() >= threshold
            n_sel = int(mask.sum().item())
            if n_sel > 0:
                update = (strength * step_size) * g.sign() * mask.float()
                p.data.copy_((p.data.float() - update.to(p.device)).to(p.dtype))
                edited_count += n_sel

    model.zero_grad(set_to_none=True)
    loss_after = eval_gate_loss(model, tokenizer, gate_texts, device, max_length)
    return {
        "mode": "dual_bias_edit",
        "early_layers": target_layers,
        "edited_weights": edited_count,
        "strength": strength,
        "step_size": step_size,
        "gate_loss_before": round(loss_before, 5),
        "gate_loss_after": round(loss_after, 5),
        "improved": loss_after < loss_before,
    }


def run_iterative_gated(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_texts: list[str],
    gate_texts: list[str],
    device: torch.device,
    max_length: int,
    rounds: int,
    step_size: float,
    top_k_per_round: int,
) -> dict:
    layers = find_layers(model)
    best_loss = eval_gate_loss(model, tokenizer, gate_texts, device, max_length)
    history = [{"round": 0, "gate_loss": round(best_loss, 5), "accepted": True}]

    for r in range(1, rounds + 1):
        # Snapshot candidate target layer (cycle through middle 40-60% depth)
        mid_start = int(len(layers) * 0.4)
        mid_end = max(mid_start + 1, int(len(layers) * 0.65))
        li = mid_start + ((r - 1) % (mid_end - mid_start))
        down_proj = getattr(getattr(layers[li], "mlp", None), "down_proj", None)
        if down_proj is None:
            continue

        backup = copy.deepcopy(down_proj.weight.data)
        model.zero_grad(set_to_none=True)
        for txt in train_texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            out = model(**enc, labels=enc["input_ids"])
            (out.loss / max(len(train_texts), 1)).backward()

        g = down_proj.weight.grad.detach().float()
        k = min(top_k_per_round, g.numel())
        thresh = torch.topk(g.abs().reshape(-1), k=k).values[-1].item()
        mask = g.abs() >= thresh
        with torch.no_grad():
            down_proj.weight.data.copy_(
                (down_proj.weight.data.float() - step_size * g.sign() * mask.float()).to(down_proj.weight.dtype)
            )
        model.zero_grad(set_to_none=True)

        cand_loss = eval_gate_loss(model, tokenizer, gate_texts, device, max_length)
        accepted = cand_loss < best_loss
        if accepted:
            best_loss = cand_loss
        else:
            down_proj.weight.data.copy_(backup)

        history.append(
            {
                "round": r,
                "layer": li,
                "edited_weights": int(mask.sum().item()),
                "gate_loss": round(cand_loss, 5),
                "accepted": accepted,
            }
        )

    return {"mode": "iterative_gated", "final_gate_loss": round(best_loss, 5), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dual_bias_edit", "iterative_gated"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-train", type=int, default=16)
    parser.add_argument("--max-gate", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k-weights", type=int, default=2048)
    parser.add_argument("--step-size", type=float, default=5e-4)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    train_texts = load_texts(Path(args.train_file), args.max_train)
    gate_texts = load_texts(Path(args.gate_file), args.max_gate)

    if args.mode == "dual_bias_edit":
        res = run_dual_bias_edit(
            model, tokenizer, train_texts, gate_texts, device, args.max_length,
            args.top_k_weights, args.step_size, args.strength,
        )
    else:
        res = run_iterative_gated(
            model, tokenizer, train_texts, gate_texts, device, args.max_length,
            args.rounds, args.step_size, args.top_k_weights,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoint_edited"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    res["saved_checkpoint"] = str(ckpt_dir)
    (out_dir / "edit_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
