#!/usr/bin/env python3
"""Experiment 62: Policy-preserving joint BC+RL warm GQA slider lane.

Exp 42 reduced false refusal but exact compliance and policy accuracy fell. This
experiment tests whether teacher-forced behavior cloning on correct Halo policy
completions can preserve decision structure while grouped RL trains a zero-init,
architecture-matched warm attention lane.
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

INSERT_LAYER = 27
SOURCE_LAYER = 29
KV_GROUP = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp62_policy_preserving_gqa_slider/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 62] Policy-preserving warm GQA slider on {device}", flush=True)

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
        "experiment": "exp62_policy_preserving_gqa_slider",
        "research_source": "Exp 42 failure analysis + research/HYPOTHESES.md H1/H3",
        "config": vars(args),
        "insert_layer": INSERT_LAYER,
        "source_layer": SOURCE_LAYER,
        "baseline": baseline,
        "arms": {},
    }
    for bc_lambda in (0.0, 0.5, 2.0):
        arm_name = f"bc_lambda_{bc_lambda:g}"
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
            INSERT_LAYER,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            beta=args.beta,
            bc_lambda=bc_lambda,
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
            "bc_lambda": bc_lambda,
            "zero_init_identity_max_diff": identity_max_diff,
            "parameter_count": count_parameters(head),
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "eval": ev,
            "delta_exact_vs_baseline": round(ev["exact_compliance_rate"] - baseline["exact_compliance_rate"], 2),
            "delta_policy_accuracy_vs_baseline": round(ev["policy_accuracy"] - baseline["policy_accuracy"], 2),
            "delta_benign_false_refusal_vs_baseline": round(ev["benign_false_refusal_rate"] - baseline["benign_false_refusal_rate"], 2),
            "trace": trace[::4] + ([trace[-1]] if trace and (len(trace) - 1) % 4 != 0 else []),
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "policy_preserving_slider_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 62 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
