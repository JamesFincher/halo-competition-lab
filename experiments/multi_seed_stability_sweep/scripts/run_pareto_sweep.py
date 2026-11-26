#!/usr/bin/env python3
"""Experiment 22: Neuron-Budget (K) x Edit-Strength Pareto Frontier Sweep (DLA).

Maps the exact 2D trade-off surface across:
  top_k in {16, 32, 64, 128, 256} neurons/layer x strength in {0.40, 0.60, 0.80, 1.00}
(20 full 40-task held-out evaluations = 800 unseen generations!) in seconds on H100.
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
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = ROOT / "experiments/multi_seed_stability_sweep/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 22] Running 20-Point Budget x Strength Pareto Sweep on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()

    dla_scores, u = rnd_mod.compute_dla_scores(sft, tokenizer, train_rows, device)

    grid = []
    best = None
    for top_k in (16, 32, 64, 128, 256):
        sel = {li: torch.topk(dla_scores[li], k=top_k).indices.tolist() for li in (29, 28)}
        for strength in (0.40, 0.60, 0.80, 1.00):
            m = copy.deepcopy(sft)
            rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=strength)
            ev = base_rl.evaluate_heldout(m, tokenizer, eval_rows, device)
            rec = {"top_k": top_k, "strength": strength, **ev}
            grid.append(rec)
            if best is None or (rec["exact_compliance_rate"], rec["unsafe_refusal_rate"]) > (best["exact_compliance_rate"], best["unsafe_refusal_rate"]):
                best = rec
            print(
                f"  [k={top_k:3d} st={strength:.2f}] exact={ev['exact_compliance_rate']:5.1f}% | "
                f"pol={ev['policy_accuracy']:5.1f}% | unsafe={ev['unsafe_refusal_rate']:5.1f}% | benign_fr={ev['benign_false_refusal_rate']:5.1f}%",
                flush=True,
            )
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {"device": str(device), "wall_sec": round(time.time() - t0, 2), "best": best, "grid": grid}
    (out_dir / "pareto_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 22 DONE] Best: k={best['top_k']} st={best['strength']} -> exact={best['exact_compliance_rate']:.1f}% unsafe={best['unsafe_refusal_rate']:.1f}% in {summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
