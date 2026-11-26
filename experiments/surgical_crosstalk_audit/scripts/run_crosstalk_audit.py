#!/usr/bin/env python3
"""Experiment 19: 7-Behavior Surgical Cross-Talk & Hub-Neuron Protection Audit.

Tests Experiment 18's cartography prediction:
  - Editing one behavior via DLA (False Over-Refusal, Sycophancy, or Verbosity) should leave
    low-Jaccard behaviors (Math, Facts, JSON) largely untouched while high-Jaccard neighbors
    (Sycophancy <-> Verbosity via L29 #778) co-move.
  - Compares Unprotected DLA vs Hub-Protected DLA (masking polysemantic hubs: L29#1512, L28#1095, L29#1022).
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/behavior_fingerprint_atlas/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_behavior_atlas as atlas_mod  # noqa: E402

HUB_NEURONS = {(29, 1512), (28, 1095), (29, 1022), (27, 110)}


@torch.no_grad()
def eval_all_7_margins(model: torch.nn.Module, tokenizer: AutoTokenizer, specs: list[dict], device: torch.device) -> dict[str, float]:
    model.eval()
    margins = {}
    for sp in specs:
        tok_pos = tokenizer.encode(sp["pos_tok"], add_special_tokens=False)[0]
        tok_neg = tokenizer.encode(sp["neg_tok"], add_special_tokens=False)[0]
        vals = []
        for txt in sp["prompts"]:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
            lg = model(**enc).logits[0, -1, :].float()
            vals.append(float((lg[tok_pos] - lg[tok_neg]).item()))
        margins[sp["id"]] = round(sum(vals) / max(len(vals), 1), 4)
    return margins


@torch.no_grad()
def apply_dla_behavior_suppress(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    spec: dict,
    device: torch.device,
    edit_layers: tuple[int, ...] = (29, 28),
    top_k: int = 64,
    strength: float = 0.65,
    protect_hubs: bool = False,
) -> int:
    """Suppress a target behavior's top-K DLA neurons via directional down_proj rank-1 projection."""
    model.eval()
    layers = model.model.layers
    tok_pos = tokenizer.encode(spec["pos_tok"], add_special_tokens=False)[0]
    tok_neg = tokenizer.encode(spec["neg_tok"], add_special_tokens=False)[0]
    lm_w = model.lm_head.weight.data.float().to(device)
    w_diff = lm_w[tok_pos] - lm_w[tok_neg]
    u = w_diff / (w_diff.norm() + 1e-8)

    inter_dim = int(model.config.intermediate_size)
    act_acc = {li: torch.zeros(inter_dim, device=device) for li in edit_layers}
    for txt in spec["prompts"]:
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
        cap = {}
        hooks = []
        for li in edit_layers:
            def _mk(i: int):
                def _hk(_m, inp, _out):
                    cap[i] = inp[0][0, -1, :].detach().float()
                return _hk
            hooks.append(layers[li].mlp.down_proj.register_forward_hook(_mk(li)))
        _ = model(**enc)
        for h in hooks:
            h.remove()
        for li in edit_layers:
            act_acc[li] += cap[li]

    inv = 1.0 / max(len(spec["prompts"]), 1)
    edited_neurons = 0
    for li in edit_layers:
        w = layers[li].mlp.down_proj.weight.data.float()  # [D, I]
        contrib = (act_acc[li] * inv) * torch.matmul(w_diff, w)
        if protect_hubs:
            for h_li, h_ni in HUB_NEURONS:
                if h_li == li and h_ni < contrib.numel():
                    contrib[h_ni] = -1e9
        idxs = torch.topk(contrib, k=top_k).indices
        # Rank-1 projection along u ONLY on the selected top-K neuron columns
        cols = w[:, idxs]  # [D, K]
        proj = torch.outer(u, torch.matmul(u, cols))
        w[:, idxs] = cols - strength * proj
        layers[li].mlp.down_proj.weight.data.copy_(w.to(layers[li].mlp.down_proj.weight.dtype))
        edited_neurons += int(idxs.numel())
    return edited_neurons


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = ROOT / "experiments/surgical_crosstalk_audit/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 19] Running 7-Behavior Surgical Cross-Talk Audit on {device} ({dtype})", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    base_model.eval()
    sft_raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft_model = PeftModel.from_pretrained(sft_raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft_model.eval()

    specs = atlas_mod.build_behavior_probes(tokenizer, train_rows)
    spec_by_id = {s["id"]: s for s in specs}

    base_margins = eval_all_7_margins(base_model, tokenizer, specs, device)
    sft_margins = eval_all_7_margins(sft_model, tokenizer, specs, device)

    targets = [
        ("2_false_over_refusal", True),
        ("4_sycophancy_bias", False),
        ("7_verbosity_hedging", False),
    ]
    results = []
    for tid, use_sft in targets:
        ref_m = sft_model if use_sft else base_model
        ref_margins = sft_margins if use_sft else base_margins
        for protect in (False, True):
            m = copy.deepcopy(ref_m)
            n_ed = apply_dla_behavior_suppress(m, tokenizer, spec_by_id[tid], device, protect_hubs=protect)
            after = eval_all_7_margins(m, tokenizer, specs, device)
            deltas = {k: round(after[k] - ref_margins[k], 4) for k in ref_margins}
            target_drop = deltas[tid]
            other_abs = [abs(v) for k, v in deltas.items() if k != tid]
            mean_collateral = round(sum(other_abs) / len(other_abs), 4)
            rec = {
                "target_behavior": tid,
                "protect_hubs": protect,
                "edited_neurons": n_ed,
                "target_margin_delta": target_drop,
                "mean_collateral_abs_delta": mean_collateral,
                "all_deltas": deltas,
            }
            results.append(rec)
            print(
                f"  [target={tid:22s} | hub_protect={str(protect):5s}] "
                f"target_delta={target_drop:+.3f} | mean_collateral={mean_collateral:.3f} | "
                f"syc={deltas['4_sycophancy_bias']:+.3f} verb={deltas['7_verbosity_hedging']:+.3f} "
                f"math={deltas['5_arithmetic_reasoning']:+.3f} fact={deltas['6_factual_recall']:+.3f}",
                flush=True,
            )
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "device": str(device),
        "wall_sec": round(time.time() - t0, 2),
        "base_margins": base_margins,
        "sft_margins": sft_margins,
        "runs": results,
    }
    (out_dir / "crosstalk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 19 DONE] Saved crosstalk_summary.json in {summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
