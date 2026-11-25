#!/usr/bin/env python3
"""Stage 1 of Edit-Then-Target RL: Contrastive + LS-LoRA + 1-Shot Fisher Weight Localization
and Optional Surgical Weight-Projection Pre-Edit.

Emulates MIT CSAIL's Weight-Space Programming (Nov 2, 2026) combined with:
- CircuitLens (arXiv:2609.07183): contrastive forward-pass localization
- LS-LoRA (arXiv:2610.11620): forward-only input-output cosine similarity s_l
- Fisher Sparsity (arXiv:2609.36262): 1-shot diagonal Fisher + SNIP |grad * w| sensitivity

Outputs:
1. Ranked per-layer localization report (`localized_targets.json`) combining:
   - contrastive residual-stream divergence ||mean(h_fail) - mean(h_pass)||
   - LS-LoRA forward sensitivity (1 - cos_sim(h_in, h_out))
   - optional 1-shot diagonal Fisher / SNIP module sensitivity
2. Ready-to-use Halo v1.3.0rc1 config snippets:
   - bare YAML regex `lora_target_modules` (for dense models / non-EP linears)
   - `unfreeze_layers_patterns` (for MoE partial full-FT with `use_peft: false`)
   - matched-budget random-layer control regex (mandatory per arXiv:2602.01599)
3. Optional surgical weight-projection pre-edit (`--apply-surgical-edit`):
   - projects out the dominant failure direction from the top-edit MLP `down_proj` output columns
     (touching ~0.01-0.1% of model parameters) and saves `checkpoint_edited/` for Stage-2 Halo GRPO.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_transformer_layers(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
    for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers", "language_model.model.layers"):
        obj = model
        ok = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList):
            return attr_path, obj
    raise RuntimeError("Could not locate transformer layer ModuleList on model.")


def load_contrastive_pairs(path: Path, max_pairs: int) -> tuple[list[str], list[str]]:
    """Load contrastive (pass_texts, fail_texts) from either:
    - JSONL with {"pass_text": ..., "fail_text": ...}
    - Halo messages JSONL (constructs paired correct vs wrong-action completions for local dry-runs).
    """
    pass_texts: list[str] = []
    fail_texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "pass_text" in row and "fail_text" in row:
            pass_texts.append(row["pass_text"])
            fail_texts.append(row["fail_text"])
        elif "messages" in row:
            msgs = row["messages"]
            sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), "")
            usr_txt = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_txt = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prompt_prefix = f"{sys_txt}\nUser: {usr_txt}\nAssistant: "
            pass_texts.append(prompt_prefix + asst_txt)
            # Construct a contrastive failure completion (wrong action / over-refusal or unsafe allow)
            try:
                parsed = json.loads(asst_txt)
                flipped_action = "refuse" if parsed.get("action") == "call_tool" else "call_tool"
                flipped = {
                    "action": flipped_action,
                    "tool": None if flipped_action == "refuse" else "run_shell",
                    "policy_code": "DESTRUCTIVE_OP" if flipped_action == "refuse" else "ALLOW",
                    "reason": "Contrastive failure trajectory.",
                }
                fail_texts.append(prompt_prefix + json.dumps(flipped, ensure_ascii=True))
            except Exception:
                fail_texts.append(prompt_prefix + "ERROR: repeated failure loop " * 6)
        if len(pass_texts) >= max_pairs:
            break
    if not pass_texts:
        raise SystemExit(f"No contrastive pairs found in {path}")
    return pass_texts, fail_texts


@torch.no_grad()
def collect_layer_stats(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    layers: torch.nn.ModuleList,
    texts: list[str],
    device: torch.device,
    max_length: int,
    tail_tokens: int,
) -> tuple[list[torch.Tensor], list[float]]:
    """Return per-layer mean tail hidden state [D] and LS-LoRA mean input-output cosine similarity."""
    n_layers = len(layers)
    sum_hidden = [None for _ in range(n_layers)]
    cos_accum = [0.0 for _ in range(n_layers)]
    count = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        in_states: list[torch.Tensor] = []
        out_states: list[torch.Tensor] = []
        hooks = []

        for idx, layer in enumerate(layers):
            def _make_hook(layer_idx: int):
                def _hook(_mod, inp, out):
                    h_in = inp[0] if isinstance(inp, tuple) else inp
                    h_out = out[0] if isinstance(out, tuple) else out
                    in_states.append((layer_idx, h_in.detach().float()))
                    out_states.append((layer_idx, h_out.detach().float()))
                return _hook
            hooks.append(layer.register_forward_hook(_make_hook(idx)))

        try:
            _ = model(**enc)
        finally:
            for h in hooks:
                h.remove()

        in_map = dict(in_states)
        out_map = dict(out_states)
        for idx in range(n_layers):
            h_in = in_map[idx][0, -tail_tokens:, :]
            h_out = out_map[idx][0, -tail_tokens:, :]
            mean_tail = h_out.mean(dim=0).cpu()
            sum_hidden[idx] = mean_tail if sum_hidden[idx] is None else sum_hidden[idx] + mean_tail
            cos = F.cosine_similarity(h_in, h_out, dim=-1).mean().item()
            cos_accum[idx] += cos
        count += 1

    mean_hiddens = [h / max(count, 1) for h in sum_hidden]
    mean_cos = [c / max(count, 1) for c in cos_accum]
    return mean_hiddens, mean_cos


def compute_one_shot_fisher(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    layers: torch.nn.ModuleList,
    pass_texts: list[str],
    device: torch.device,
    max_length: int,
    fisher_samples: int,
) -> list[float]:
    """1-shot diagonal Fisher + SNIP |grad * w| per-layer sensitivity (arXiv:2609.36262)."""
    n_layers = len(layers)
    scores = [0.0 for _ in range(n_layers)]
    if fisher_samples <= 0:
        return scores

    model.zero_grad(set_to_none=True)
    used = 0
    for text in pass_texts[:fisher_samples]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        for idx, layer in enumerate(layers):
            layer_score = 0.0
            n_params = 0
            for p in layer.parameters():
                if p.grad is not None:
                    # SNIP + Fisher diagonal proxy: mean(|g * w| + g^2)
                    g = p.grad.detach().float()
                    w = p.detach().float()
                    layer_score += (g.abs() * w.abs()).sum().item() + (g * g).sum().item()
                    n_params += p.numel()
            scores[idx] += layer_score / max(n_params, 1)
        model.zero_grad(set_to_none=True)
        used += 1

    return [s / max(used, 1) for s in scores]


def normalize(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def apply_surgical_projection_edit(
    model: torch.nn.Module,
    layers: torch.nn.ModuleList,
    edit_layer_indices: list[int],
    failure_dirs: dict[int, torch.Tensor],
    strength: float,
    top_columns: int,
) -> dict:
    """Zero-gradient surgical pre-edit (Weight-Space Programming style):
    For each target layer's MLP down_proj [D_out, D_in], identify the top `top_columns`
    columns most aligned with the failure direction `u_l` and subtract `strength * u_l (u_l^T w_j)`.
    """
    total_params = sum(p.numel() for p in model.parameters())
    edited_weights = 0
    edited_details = []

    with torch.no_grad():
        for idx in edit_layer_indices:
            layer = layers[idx]
            down_proj = getattr(getattr(layer, "mlp", None), "down_proj", None)
            if down_proj is None or not hasattr(down_proj, "weight"):
                continue
            w = down_proj.weight  # [hidden_size, intermediate_size]
            u = failure_dirs[idx].to(device=w.device, dtype=torch.float32)
            u_norm = u.norm()
            if u_norm < 1e-8:
                continue
            u = u / u_norm

            w_f32 = w.data.float()
            # Alignment of each column j with failure direction u: |u^T w_j|
            proj = torch.matmul(u, w_f32)  # [intermediate_size]
            k = min(top_columns, proj.numel())
            top_idx = torch.topk(proj.abs(), k=k).indices

            # Subtract strength * (u^T w_j) * u from selected columns only
            delta = strength * torch.outer(u, proj[top_idx])
            w_f32[:, top_idx] = w_f32[:, top_idx] - delta
            w.data.copy_(w_f32.to(w.dtype))

            n_mod = int(w.shape[0] * k)
            edited_weights += n_mod
            edited_details.append(
                {
                    "layer": idx,
                    "module": "mlp.down_proj",
                    "columns_edited": k,
                    "weights_edited": n_mod,
                    "max_col_proj": round(float(proj[top_idx].abs().max().item()), 6),
                }
            )

    return {
        "total_model_params": total_params,
        "edited_weights": edited_weights,
        "edited_fraction_pct": round(100.0 * edited_weights / max(total_params, 1), 5),
        "strength": strength,
        "layers": edited_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--pairs-file", required=True, help="JSONL of contrastive pairs or Halo messages")
    parser.add_argument("--out-dir", required=True, help="Output directory for report & optional edited checkpoint")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--tail-tokens", type=int, default=32)
    parser.add_argument("--top-k-frac", type=float, default=0.25, help="Fraction of layers to localize (default 0.25)")
    parser.add_argument("--fisher-samples", type=int, default=4, help="1-shot Fisher/SNIP samples (0 to disable)")
    parser.add_argument("--apply-surgical-edit", action="store_true", help="Save surgical pre-edited checkpoint")
    parser.add_argument("--edit-layers", type=int, default=2, help="Number of top failure-direction layers to edit")
    parser.add_argument("--edit-columns", type=int, default=64, help="MLP down_proj columns to edit per layer")
    parser.add_argument("--edit-strength", type=float, default=0.5, help="Projection subtraction strength (0..1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    layer_prefix, layers = find_transformer_layers(model)
    n_layers = len(layers)
    pass_texts, fail_texts = load_contrastive_pairs(Path(args.pairs_file), args.max_pairs)

    h_pass, cos_pass = collect_layer_stats(
        model, tokenizer, layers, pass_texts, device, args.max_length, args.tail_tokens
    )
    h_fail, cos_fail = collect_layer_stats(
        model, tokenizer, layers, fail_texts, device, args.max_length, args.tail_tokens
    )

    failure_dirs = {i: (h_fail[i] - h_pass[i]) for i in range(n_layers)}
    contrastive_div = [float(failure_dirs[i].norm().item()) for i in range(n_layers)]
    ls_lora_sens = [1.0 - 0.5 * (cos_pass[i] + cos_fail[i]) for i in range(n_layers)]
    fisher_sens = compute_one_shot_fisher(
        model, tokenizer, layers, pass_texts, device, args.max_length, args.fisher_samples
    )

    n_div = normalize(contrastive_div)
    n_ls = normalize(ls_lora_sens)
    n_fish = normalize(fisher_sens) if args.fisher_samples > 0 else [0.0] * n_layers

    combined = []
    for i in range(n_layers):
        if args.fisher_samples > 0:
            score = 0.40 * n_div[i] + 0.20 * n_ls[i] + 0.40 * n_fish[i]
        else:
            score = 0.65 * n_div[i] + 0.35 * n_ls[i]
        combined.append(
            {
                "layer": i,
                "depth_frac": round((i + 0.5) / n_layers, 3),
                "combined_score": round(score, 4),
                "contrastive_l2": round(contrastive_div[i], 5),
                "ls_lora_1_minus_cos": round(ls_lora_sens[i], 5),
                "fisher_snip": round(fisher_sens[i], 8),
            }
        )

    ranked = sorted(combined, key=lambda x: x["combined_score"], reverse=True)
    k_layers = max(1, int(math.ceil(n_layers * args.top_k_frac)))
    top_layers = sorted([r["layer"] for r in ranked[:k_layers]])

    rng = random.Random(args.seed)
    non_top = [i for i in range(n_layers) if i not in top_layers]
    random_layers = sorted(rng.sample(non_top, k=min(k_layers, len(non_top))))

    top_pattern = "|".join(str(i) for i in top_layers)
    rand_pattern = "|".join(str(i) for i in random_layers)

    halo_snippets = {
        "dense_bare_yaml_regex_lora_target_modules": (
            f".*\\.layers\\.({top_pattern})\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"
        ),
        "random_control_regex_lora_target_modules": (
            f".*\\.layers\\.({rand_pattern})\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"
        ),
        "moe_unfreeze_layers_patterns": [f"*{layer_prefix}.{i}.*" for i in top_layers],
        "random_control_unfreeze_layers_patterns": [f"*{layer_prefix}.{i}.*" for i in random_layers],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edit_report = None
    if args.apply_surgical_edit:
        top_div_layers = [
            r["layer"] for r in sorted(combined, key=lambda x: x["contrastive_l2"], reverse=True)[: args.edit_layers]
        ]
        edit_report = apply_surgical_projection_edit(
            model, layers, top_div_layers, failure_dirs, args.edit_strength, args.edit_columns
        )
        ckpt_dir = out_dir / "checkpoint_edited"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        edit_report["saved_checkpoint"] = str(ckpt_dir)

    report = {
        "model": args.model,
        "device": str(device),
        "n_layers": n_layers,
        "n_pairs": len(pass_texts),
        "fisher_samples": args.fisher_samples,
        "top_k_layers": top_layers,
        "random_control_layers": random_layers,
        "halo_config_snippets": halo_snippets,
        "surgical_edit": edit_report,
        "layer_scores_ranked": ranked,
    }
    report_path = out_dir / "localized_targets.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Localized top-{k_layers}/{n_layers} layers: {top_layers}")
    print(f"[OK] Random control layers: {random_layers}")
    print(f"[OK] Halo dense regex: {halo_snippets['dense_bare_yaml_regex_lora_target_modules']}")
    if edit_report:
        print(
            f"[OK] Surgical pre-edit modified {edit_report['edited_weights']:,} weights "
            f"({edit_report['edited_fraction_pct']}%) -> {edit_report['saved_checkpoint']}"
        )
    print(f"[OK] Wrote {report_path}")


if __name__ == "__main__":
    main()
