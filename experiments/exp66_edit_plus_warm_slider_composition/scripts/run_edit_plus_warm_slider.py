#!/usr/bin/env python3
"""Experiment 66: Composition of validated surgical edit and warm GQA slider lane.

The lab's best validated interventions are subtractive residual-subspace surgery and
additive zero-init attention inserts, but they have not been composed directly. This
experiment measures edit-only versus edit+warm-slider trained with insert-only BC+RL.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from gqa_slider import GqaSliderHead, balanced_rows, count_parameters, train_slider_rl, warm_copy_from_gqa_group  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402

INSERT_LAYER = 27
SOURCE_LAYER = 29


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--bc-lambda", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=0.90)
    parser.add_argument("--top-columns", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp66_edit_plus_warm_slider_composition/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 66] Surgical edit + warm slider composition on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = balanced_rows(train_rows, n_each=12)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    baseline = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [baseline] exact={baseline['exact_compliance_rate']:.1f}% unsafe={baseline['unsafe_refusal_rate']:.1f}% benign_fr={baseline['benign_false_refusal_rate']:.1f}%", flush=True)

    edited = copy.deepcopy(sft).to(device).eval()
    edit_info = apply_validated_rank1_edit(
        edited, tokenizer, train_rows, device,
        edit_layers=(29, 28), top_columns=args.top_columns, strength=args.strength,
    )
    edit_only = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [edit_only] exact={edit_only['exact_compliance_rate']:.1f}% unsafe={edit_only['unsafe_refusal_rate']:.1f}% benign_fr={edit_only['benign_false_refusal_rate']:.1f}%", flush=True)

    ref_edit = copy.deepcopy(edited).to(device).eval()
    for p in ref_edit.parameters():
        p.requires_grad_(False)
    torch.manual_seed(args.seed)
    head = GqaSliderHead(hidden_size=edited.config.hidden_size).to(device)
    warm_copy_from_gqa_group(head, edited, SOURCE_LAYER, 0)
    with torch.no_grad():
        probe = torch.randn(1, 5, edited.config.hidden_size, device=device)
        identity_max_diff = float((head(probe) - probe).abs().max().item())
    if identity_max_diff != 0.0:
        raise RuntimeError(f"zero-init identity failed: {identity_max_diff}")
    handle, trace = train_slider_rl(
        "edit_plus_warm_slider",
        edited,
        ref_edit,
        tokenizer,
        rows,
        device,
        head,
        INSERT_LAYER,
        steps=args.steps,
        group_size=args.group_size,
        lr=args.lr,
        beta=args.beta,
        bc_lambda=args.bc_lambda,
        seed=args.seed,
    )
    composite = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    handle.remove()
    print(f"  [composite] exact={composite['exact_compliance_rate']:.1f}% unsafe={composite['unsafe_refusal_rate']:.1f}% benign_fr={composite['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.5f}", flush=True)

    results = {
        "experiment": "exp66_edit_plus_warm_slider_composition",
        "research_source": "research/HYPOTHESES.md operating context: compose subtractive residual edits with additive zero-damage inserts",
        "config": vars(args),
        "insert_layer": INSERT_LAYER,
        "baseline": baseline,
        "validated_edit": {"operator_info": edit_info, "eval": edit_only},
        "composite": {
            "eval": composite,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "parameter_count": count_parameters(head),
            "zero_init_identity_max_diff": identity_max_diff,
            "delta_exact_vs_edit_only": round(composite["exact_compliance_rate"] - edit_only["exact_compliance_rate"], 2),
            "delta_unsafe_refusal_vs_edit_only": round(composite["unsafe_refusal_rate"] - edit_only["unsafe_refusal_rate"], 2),
            "delta_benign_false_refusal_vs_edit_only": round(composite["benign_false_refusal_rate"] - edit_only["benign_false_refusal_rate"], 2),
            "trace": trace[::3] + ([trace[-1]] if trace and (len(trace) - 1) % 3 != 0 else []),
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    path = out_dir / "edit_plus_warm_slider_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 66 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
