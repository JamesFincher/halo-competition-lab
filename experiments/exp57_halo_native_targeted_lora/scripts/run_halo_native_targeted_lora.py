#!/usr/bin/env python3
"""Experiment 57: Halo-Native Post-Edit Anchored Targeted LoRA.

Implements the top transfer recommendation from SUBAGENT4: initialize policy and
KL reference from the validated post-edit checkpoint, freeze every base weight
through PEFT, target only complementary layers with a bare regex, and use
weight_decay=0. Negative controls include edited-layer targeting and full-model
LoRA.

Research anchors: research/SUBAGENT4_HALO_NATIVE_RL.md plus arXiv:2403.10704
(PE-RLHF) and arXiv:2609.18587 (small-parameter-space RL).
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
from validated_edit import apply_validated_rank1_edit  # noqa: E402

COMPLEMENT_LAYERS = [24, 25, 26, 27]
EDITED_INCLUSIVE_LAYERS = [24, 25, 26, 27, 28, 29]


def layer_regex(layers: list[int], mlp_only: bool = False) -> str:
    layer_alt = "|".join(str(x) for x in layers)
    if mlp_only:
        return f".*\\.layers\\.({layer_alt})\\.mlp\\.(gate|up|down)_proj$"
    return f".*\\.layers\\.({layer_alt})\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp57_halo_native_targeted_lora/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 57] Halo-Native Post-Edit Anchored Targeted LoRA on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, dtype)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval()
    vec_rl.freeze_weights(ref_model)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [validated edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)

    full_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    arm_specs = [
        ("halo_regex_complementary_allproj_r16", layer_regex(COMPLEMENT_LAYERS, mlp_only=False), 16),
        ("halo_regex_complementary_mlp_r16", layer_regex(COMPLEMENT_LAYERS, mlp_only=True), 16),
        ("edited_layer_inclusion_negative_control_r16", layer_regex(EDITED_INCLUSIVE_LAYERS, mlp_only=False), 16),
        ("full_lora_control_r8", full_targets, 8),
    ]
    results = {
        "experiment": "exp57_halo_native_targeted_lora",
        "research_source": "research/SUBAGENT4_HALO_NATIVE_RL.md top transfer recommendation",
        "paper_inspiration": ["arXiv:2403.10704", "arXiv:2609.18587"],
        "config": vars(args),
        "device": str(device),
        "validated_edit_operator": edit_info,
        "complement_layers": COMPLEMENT_LAYERS,
        "baseline_sft": sft_eval,
        "anchored_edit_only": edit_eval,
        "arms": {},
    }

    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm_name, targets, rank in arm_specs:
            print(f"[Run] {arm_name}", flush=True)
            trained, stats = base_rl.run_grpo_arm(
                arm_name,
                edited,
                ref_model,
                tokenizer,
                prompt_stream,
                device,
                targets,
                rank,
                args.steps,
                args.group_size,
                args.lr,
                args.beta,
                log,
                args.seed,
            )
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm_name] = {"train_stats": stats, "eval": ev, "target_spec": targets if isinstance(targets, str) else "all_projection_module_names", "lora_r": rank}
            del trained
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "halo_native_targeted_lora_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 57 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
