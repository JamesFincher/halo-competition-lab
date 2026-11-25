# Experiment 3 (Hypothesis C): GRAIL-Orth Behavioral Subspace-Targeted LoRA RL

**Priority:** #2 Cheapest Standalone Win (**7.5/10**, ~$12–$25 pilot)  
**Target Models:** `Qwen/Qwen3-4B` (1× GPU) → `unsloth/gpt-oss-20b-BF16` / `google/gemma-4-26B-A4B-it`

## Core Hypothesis
Instead of random Kaiming LoRA initialization (slow, noisy) or naive pretrained-weight SVD (`PiSSA`/`MiLoRA`, which causes spectral collapse in RLVR), initializing `lora_A` with the **top-r orthonormal right-singular vectors of calibration reward gradients projected OUT of the top-k pretrained principal directions (`GRAIL` + `GCPO`)** accelerates GRPO convergence, cuts KL drift, and preserves out-of-domain capability — with **zero Halo forks and 100% vLLM NCCL weight-sync parity**.

## Supporting Literature
- **GRAIL (EMNLP 2026 Findings):** Reward-aligned gradient initialization of LoRA factors before GRPO consistently beats standard LoRA-GRPO across Qwen2.5, Phi-3.5, and Gemma-2.
- **GCPO (`arXiv:2608.11674`, Aug 2026):** Projecting RL updates onto the orthogonal complement of pretrained principal subspaces eliminates length inflation and improves Qwen3-8B / GLM4-9B by +1.0–2.4 pts while preserving cross-task capabilities.
- **LoRA-RLPO (`arXiv:2606.31813`) & PeRL (`arXiv:2512.23165`):** Proves naive PiSSA/MiLoRA SVD init destabilizes RLVR, whereas orthonormal initialization closes the gap to full FT.
- **Why ASH (`arXiv:2611.23124`) is excluded here:** Runtime activation steering hooks are silently dropped by Halo's `_HubForwarder` (`weight_push.py:327-364`) when pushing weights to vLLM, breaking on-policy IS ratios and competition serving.

## Halo v1.3.0rc1 Compatibility & Gotchas
- Keep `init_lora_weights: true` in YAML (required by `ExpertLoraSpec.peft_config_conflicts` at `expert_lora_spec.py:49-52`), then overwrite `lora_A.data` (orthonormal) and zero `lora_B.data` post-wrap.
- Keep `rollout_adapter_sync: false` so `lora_folded_data()` folds adapters into BF16 hub weights for vLLM.
- Optional `--freeze-a` mode (`lora_A.requires_grad_(False)`) passes `verify_adapters_trainable()` (`peft_setup.py:353`) and `install_fsdp2_safe_disable_adapter()` (`fsdp.py:656-686`).

## Files
- `scripts/init_grail_orth_lora.py` — Calibration gradient capture + GCPO orthogonal projection + orthonormal SVD `lora_A` initialization (supports trainable `A+B` or `--freeze-a`).
- `configs/qwen3-4b-grail-orth-grpo.yaml` — Halo online GRPO config.

## Compute & Go/No-Go Criteria
- **Pilot cost (`Qwen3-4B`, 4 arms × 200 steps):** ~4–6 GPU-hrs (~$12–$25).
- **GO:** GRAIL-Orth beats standard Kaiming LoRA by ≥+2.0 pp pass@8 (or matches reward in ≤60% steps) with ≤50% KL divergence.
