#!/usr/bin/env python3
"""Hypothesis C — GRAIL-Orth: Reward-Aligned & Off-Principal SVD LoRA Initialization.

Grounded in:
- GRAIL (EMNLP 2026 Findings): estimate reward-aligned gradients before RL to initialize LoRA A
- GCPO (arXiv:2608.11674): project updates out of top-k pretrained principal singular vectors
- LoRA-RLPO (arXiv:2606.31813) & PeRL (arXiv:2512.23165): avoid PiSSA/MiLoRA spectral collapse

Halo v1.3.0rc1 Compatibility:
- Keeps `init_lora_weights: true` (required by Halo ExpertLoraSpec) and overwrites `lora_A.data`
  with orthonormal off-principal right-singular vectors and zeros `lora_B.data` (zero base drift).
- 100% compatible with Halo's folded vLLM NCCL weight sync (`rollout_adapter_sync: false`).
- Optional `--freeze-a` sets `lora_A.requires_grad_(False)` (hard subspace mode).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(path: Path, max_rows: int) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            out.append("\n".join(f"{m['role']}: {m['content']}" for m in row["messages"]))
        elif "pass_text" in row:
            out.append(row["pass_text"])
        if len(out) >= max_rows:
            break
    return out


def compute_grail_orth_A(
    weight: torch.Tensor,
    grad: torch.Tensor,
    rank: int,
    gcpo_drop_top_k: int,
) -> torch.Tensor:
    """Project calibration gradient G out of top-k pretrained right-singular subspace of W,
    then take top-r orthonormal right-singular vectors as lora_A [r, in_features]."""
    w_f32 = weight.detach().float()
    g_f32 = grad.detach().float()

    # Truncated SVD on pretrained weight W to get principal subspace to avoid (GCPO)
    q_proj = min(gcpo_drop_top_k + 8, min(w_f32.shape) - 1)
    if gcpo_drop_top_k > 0 and q_proj > 0:
        _u, _s, v = torch.svd_lowrank(w_f32, q=q_proj, niter=2)
        v_top = v[:, : min(gcpo_drop_top_k, v.shape[1])]  # [in_features, k]
        # Orthogonal complement projection: G_orth = G - (G V_top) V_top^T
        g_orth = g_f32 - (g_f32 @ v_top) @ v_top.T
    else:
        g_orth = g_f32

    q_grad = min(rank + 8, min(g_orth.shape) - 1)
    _ug, _sg, vg = torch.svd_lowrank(g_orth, q=max(q_grad, rank), niter=2)
    a_orth = vg[:, :rank].T.contiguous()  # [rank, in_features], rows orthonormal
    return a_orth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gcpo-drop-top-k", type=int, default=32)
    parser.add_argument("--max-calib", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--freeze-a", action="store_true")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    texts = load_texts(Path(args.calib_file), args.max_calib)

    model.zero_grad(set_to_none=True)
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        out = model(**enc, labels=enc["input_ids"])
        (out.loss / max(len(texts), 1)).backward()

    grad_cache = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and any(name.endswith(t) for t in args.target_modules):
            if mod.weight.grad is not None:
                grad_cache[name] = (mod.weight.detach().cpu(), mod.weight.grad.detach().cpu())

    model.zero_grad(set_to_none=True)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=args.target_modules,
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_cfg)

    initialized = 0
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and "default" in mod.lora_A:
            base_key = name.replace("base_model.model.", "")
            if base_key in grad_cache:
                w_cpu, g_cpu = grad_cache[base_key]
                a_init = compute_grail_orth_A(w_cpu, g_cpu, args.lora_r, args.gcpo_drop_top_k)
                mod.lora_A["default"].weight.data.copy_(
                    a_init.to(device=mod.lora_A["default"].weight.device, dtype=mod.lora_A["default"].weight.dtype)
                )
                mod.lora_B["default"].weight.data.zero_()
                if args.freeze_a:
                    mod.lora_A["default"].weight.requires_grad_(False)
                initialized += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    meta = {
        "model": args.model,
        "initialized_modules": initialized,
        "lora_r": args.lora_r,
        "gcpo_drop_top_k": args.gcpo_drop_top_k,
        "freeze_a": args.freeze_a,
    }
    (out_dir / "grail_orth_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Initialized {initialized} GRAIL-Orth LoRA adapters -> {out_dir}")


if __name__ == "__main__":
    main()
