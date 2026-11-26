#!/usr/bin/env python3
"""Experiment 65: Insertion-layer sweep for warm zero-init GQA slider lanes.

The first slider-head studies inserted after layer 27 by convention. This experiment
compares early-late (L24), canonical (L27), and final-layer (L29) insertion at fixed
architecture, warm start, RL budget, and parameter count.
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

SOURCE_LAYER = 29
KV_GROUP = 0
INSERT_LAYERS = (24, 27, 29)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp65_slider_insertion_layer_sweep/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 65] Warm GQA slider insertion-layer sweep on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = balanced_rows(train_rows, n_each=12)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    ref_model = copy.deepcopy(sft).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    baseline = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(
        f"  [baseline] exact={baseline['exact_compliance_rate']:.1f}% unsafe={baseline['unsafe_refusal_rate']:.1f}% benign_fr={baseline['benign_false_refusal_rate']:.1f}%",
        flush=True,
    )

    results = {
        "experiment": "exp65_slider_insertion_layer_sweep",
        "research_source": "Exp 42/59/61/62 attention-insert sequence + H3 layer/localization question",
        "config": vars(args),
        "source_layer": SOURCE_LAYER,
        "baseline": baseline,
        "arms": {},
    }
    for layer in INSERT_LAYERS:
        arm_name = f"insert_after_l{layer}"
        print(f"[Run] {arm_name}", flush=True)
        torch.manual_seed(args.seed)
        head = GqaSliderHead(hidden_size=sft.config.hidden_size).to(device)
        warm_copy_from_gqa_group(head, sft, SOURCE_LAYER, KV_GROUP)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity_max_diff = float((head(probe) - probe).abs().max().item())
        if identity_max_diff != 0.0:
            raise RuntimeError(f"zero-init identity failed: {identity_max_diff}")
        handle, trace = train_slider_rl(
            arm_name,
            sft,
            ref_model,
            tokenizer,
            rows,
            device,
            head,
            layer,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            beta=args.beta,
            bc_lambda=0.5,
            seed=args.seed,
        )
        ev = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        handle.remove()
        print(
            f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% policy={ev['policy_accuracy']:.1f}% "
            f"unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.5f}",
            flush=True,
        )
        results["arms"][arm_name] = {
            "insert_layer": layer,
            "bc_lambda": 0.5,
            "zero_init_identity_max_diff": identity_max_diff,
            "parameter_count": count_parameters(head),
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "eval": ev,
            "delta_exact_vs_baseline": round(ev["exact_compliance_rate"] - baseline["exact_compliance_rate"], 2),
            "delta_policy_accuracy_vs_baseline": round(ev["policy_accuracy"] - baseline["policy_accuracy"], 2),
            "delta_benign_false_refusal_vs_baseline": round(ev["benign_false_refusal_rate"] - baseline["benign_false_refusal_rate"], 2),
            "trace": trace[::3] + ([trace[-1]] if trace and (len(trace) - 1) % 3 != 0 else []),
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_name, best = max(
        results["arms"].items(),
        key=lambda item: (
            item[1]["eval"]["exact_compliance_rate"],
            item[1]["eval"]["unsafe_refusal_rate"],
            -item[1]["eval"]["benign_false_refusal_rate"],
        ),
    )
    results["diagnostic_best_exact_arm"] = {"arm": best_name, "eval": best["eval"]}
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "slider_insertion_layer_sweep_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 65 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
