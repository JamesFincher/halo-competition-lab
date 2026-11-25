# Experiment 1 (Flagship): Edit-Then-Target RL

**Priority:** #1 Flagship Track (**8.5/10 combined**)  
**Target Models:** `Qwen/Qwen3-4B` (cheap pre-Dec-1 pilot) → `unsloth/gpt-oss-20b-BF16` & `google/gemma-4-26B-A4B-it` (competition)

## Core Hypothesis
Combining **Stage-1 zero-gradient / surgical weight-space pre-edits** with **Stage-2 targeted Halo RL restricted to localized + 1-shot Fisher weight subsets** beats either half alone:
1. **Manual/surgical pre-edit unlocks RL:** Removing structural blockers (thinking/repetition loops, over-refusal, format drift) on ~0.01–0.1% of weights before step 0 eliminates GRPO's all-zero-reward dead groups.
2. **Targeted RL hardens the manual edit:** `arXiv:2611.26841` and MIT CSAIL's Gemma 3 270M control show zero-gradient edits degrade under prompt-wording changes and cannot create missing multi-step reasoning; restricting Halo GRPO to the top 5–25% localized layers hardens generalization while leaving 75–95% of the model frozen to prevent catastrophic forgetting.

## Supporting Literature (2026)
- **Weight-Space Programming** (Takehi et al., MIT CSAIL, Nov 2, 2026 — [blog](https://rikiyat.github.io/blog/weight-space-programming/)): Zero-gradient edit of 0.016% weights (131,072 weights in layers 15 & 19 MLPs) cut `Qwen3.5-0.8B` loops 93.0% → 33.1%, raising MATH-500 7.9% → 48.1%; non-looping `Gemma 3 270M` showed no gain (6.8% → 7.0%).
- **`arXiv:2611.33133` (*A Bias in Disguise*, Nov 24, 2026):** Combining a mid-layer weight-space edit with a half-strength early-layer (layers 0–2) sparse gradient edit boosted `Qwen3.5-0.8B` MATH-500 **52.1% → 62.2% (+10.1 pp, `p = 3e-9`)** with **half as many tokens**.
- **`arXiv:2611.32064` (*Fix How It Acts, Not How It Thinks*, Nov 24, 2026):** Standard LoRA loses ~2/3 of its update norm upon BF16 merge (93% in NVFP4); direct BF16/merge-aware surgical fine-tuning improved **Nemotron-3.5** by **+8.4 pts** on Terminal-Bench 2.1.
- **`arXiv:2611.29513` (*SELFWRITE*, Nov 22, 2026):** Iterative gate-verified white-box weight edits strictly improve held-out accuracy across generations without gradients.
- **`arXiv:2609.36262` (Sep 2026):** 1-shot initial diagonal Fisher masks recover 85.2% of full RLVR MATH-500 gain at 0.1% params and 98.8% at 5%.
- **`arXiv:2607.01232` (*Is One Layer Enough?*) & `arXiv:2610.11620` (*LS-LoRA*):** Middle-stack (40–60% depth) and low input-output cosine-similarity layers match or beat all-layer adaptation.
- **`arXiv:2602.01599` (*Multiple Ticket Hypothesis*, ICML 2026):** Random 1% sparse masks are strong in RLVR — making a matched-budget random-layer control mandatory.

## Halo v1.3.0rc1 Compatibility & Gotchas
- **No `layers_to_transform` flag:** Use a **bare YAML regex string** in `lora_target_modules` (never regex chars inside a YAML list, which raises `ValueError` at `halo/configs/script_arguments.py:120-127`).
- **EP MoE partial full-FT:** Halo's grouped EP `ExpertLoraSpec` has no per-layer filter; use `use_peft: false` + `unfreeze_layers_patterns` (Arm 5) to natively cover EP MoE experts AND avoid LoRA BF16 merge-rounding loss.
- **`AdamWBF16` trap:** Always set `weight_decay: 0.0` when masking gradients; `_adam_bf16_sr_kernel` (`halo/optimizers/adamw_bf16.py:192-195`) decays zero-grad elements otherwise.
- **vLLM NCCL sync:** Keep `vllm_mode: server` and `rollout_adapter_sync: false` so Halo folds adapters out-of-place via `lora_folded_data()` with 100% parity.

## Files & Ablation Arms

| Arm | Config / Script | Variant |
|---|---|---|
| Stage 1 | `scripts/localize_and_edit_weights.py` | Contrastive + LS-LoRA + 1-Shot Fisher localization & surgical `down_proj` pre-edit (verified live on `SmolLM2-135M`) |
| Stage 1b/1c | `scripts/dual_and_iterative_edit.py` | V4 Dual-Edit (`2611.33133`) & V6 Iterative Gated SELFWRITE (`2611.29513`) |
| **V0** | `configs/arm0_full_grpo.yaml` | Full all-layer LoRA GRPO baseline |
| **V2** | `configs/arm2_targeted_grpo_only.yaml` | Targeted Halo GRPO on localized top-25% layers (no pre-edit) |
| **V3** | `configs/arm3_edit_then_target_grpo.yaml` | **Flagship:** Stage-1 surgical pre-edit → Stage-2 localized Halo GRPO |
| **V4** | `configs/arm4_dual_edit_synergy_grpo.yaml` | Mid-layer edit + half-strength early-layer (0–2) edit → Targeted RL |
| **V5** | `configs/arm5_action_surgical_partial_ft.yaml` | Direct BF16 partial full-FT (`unfreeze_layers_patterns`, zero merge loss) |
| **V6** | `configs/arm6_iterative_selfwrite_rl.yaml` | Multi-round gate-verified edits → Targeted Halo GRPO |
| **V7** | `configs/arm7_random_control_grpo.yaml` | Mandatory matched-budget random-layer control (`2602.01599`) |

## Compute & Go/No-Go Criteria
- **Pilot cost (`Qwen3-4B`, 200 GRPO steps/arm):** ~8–16 GPU-hrs (~$25–$60).
- **GO:** V3, V4, or V5 beats V0 (Full RL) OR matches V0 in ≤60% steps while beating V7 (Random Control) by ≥2 pp and losing ≤1 pp on OOD retention.
