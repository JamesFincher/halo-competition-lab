# Experiment 2 (Hypothesis D): 1-Shot Fisher / Reward-Gradient Top-K Module RL

**Priority:** #1 Standalone Localization Method (**8.0/10**)  
**Target Models:** `unsloth/gpt-oss-20b-BF16`, `google/gemma-4-26B-A4B-it`, `Qwen/Qwen3-4B`

## Core Hypothesis
Running 1–5 calibration backward passes to rank modules by **diagonal Fisher + SNIP saliency (`g^2 + |g * w|`)** and freezing the bottom 95% of modules via `requires_grad=False` before Halo's FSDP2 wrap recovers ≥90–98% of full RLVR accuracy, beats zero-gradient localization on non-looping reasoning tasks, and yields **real 6 B/param `AdamWBF16` optimizer memory savings**.

## Supporting Literature
- **`arXiv:2609.36262` (Sep 2026):** Proves diagonal model Fisher bounds expected gradients across RL/OPD/SFT; 1-shot initial Fisher masks recover **85.21%** of full RLVR MATH-500 gain at **0.1%** params and **98.82%** at **5%**, outperforming random masks at every budget.
- **`arXiv:2609.15064` (*Fixed-SAE RL Tracking*, Sep 2026):** Confirms GRPO representation drift is concentrated in late layers (4.6× larger shift at layer 27 vs layer 6).
- **`arXiv:2505.11711` (Mukherjee et al.):** RL naturally updates 5–30% of weights.
- **`arXiv:2602.01599` (*Multiple Ticket Hypothesis*, ICML 2026):** Mandatory matched-budget random-mask control baseline.

## Halo v1.3.0rc1 Compatibility & Gotchas
- **Native FSDP2 / EP8 / GC / Optimizer fit:** `requires_grad=False` applied rank-uniformly BEFORE `_configure_distributed_modes()` (`halo/trainers/distributed_trainer.py:365-432`) is natively respected by FSDP2 (`halo/distributed/fsdp.py:135,205,279`), reentrant gradient checkpointing (`gradient_checkpointing.py:201-243`), `AdamWBF16` (`param_groups.py:33,60`), and EP8 grad sync (`grad_sync.py:222`).
- **EP8 3D expert tensor granularity:** Local experts are packed in fused 3D `[E_local, K, N]` tensors (`families/gpt_oss.py:40-56`), so `requires_grad=False` operates at whole-layer expert granularity (never per-expert).
- **Avoid element-wise masking with non-zero weight decay:** `AdamWBF16` applies decoupled weight decay even when `grad == 0`; keep `weight_decay: 0.0`.

## Files
- `scripts/fisher_module_select.py` — 1-shot diagonal Fisher + SNIP selector emitting `unfreeze_layers_patterns` and a matched-budget random control list.
- `configs/gptoss-20b-fisher-topk-grpo.yaml` — Halo EP4 online GRPO config (`use_peft: false`, `unfreeze_layers_patterns`).

## Compute & Go/No-Go Criteria
- **Calibration:** ~5–10 min on 1× GPU (8–64 prompts).
- **GO:** Top-5% Fisher modules recover ≥90% of Full RL pass@1 AND beat matched random modules by ≥2 pp with lower peak GPU memory.
