# Experiment 7: Fine-Grained Token Credit Assignment & Adaptive GRPO Group Sizing

**Priority:** Plug-In GRPO Efficiency Upgrade (**7.5/10** — directly composable with Experiments 1–4)  
**Target Models:** `Qwen/Qwen3-4B`, `unsloth/gpt-oss-20b-BF16`, `google/gemma-4-26B-A4B-it`

## Core Hypothesis
Standard GRPO suffers from two variance bottlenecks: (1) a fixed rollout group size `G` wastes compute on easy/degenerate prompts while under-sampling high-variance prompts, and (2) sequence-level pass/fail rewards assign identical credit to both correct reasoning steps and wrong final-answer tokens. Combining **U-Statistics pessimistic jackknife adaptive group sizing (`arXiv:2611.33064`, Nov 24, 2026)** with **SAEPO-style token-level credit assignment (`arXiv:2611.29936`, Nov 23, 2026)** lowers policy-gradient MSE and accelerates early RLVR accuracy growth.

## Supporting Literature
- **`arXiv:2611.33064` (*U-Statistics-Powered LLM Reasoning*, Gong, Ye, Chernozhukov et al., Nov 24, 2026):** Recasts GRPO group size `G` as variance-ratio estimation using a jackknife-based pessimistic estimator, reducing policy-gradient MSE and improving math/code RL over fixed-`G` baselines.
- **`arXiv:2611.29936` (*SAEPO: Sparse-Autoencoder-Guided Credit Assignment for Policy Optimization*, Nov 23, 2026):** Plug-in token-level advantage shaping compatible with GRPO/GSPO; improves math averages by up to **+6.63%** and code averages by **+5.57%**.

## Halo v1.3.0rc1 Compatibility
- Drop-in pre-rollout / advantage-shaping utilities that require **zero changes to Halo's distributed parallelism or vLLM weight sync** (`rollout_adapter_sync: false`).
- Pairs directly with Experiment 1 (`Edit-Then-Target RL`): after Stage-1 removes structural loops, adaptive `G` + token-level credit focuses Stage-2 localized RL budget on the hardest non-degenerate prompts.

## Files
- `scripts/ustats_group_size_and_credit.py` — Implements the jackknife pessimistic variance-ratio adaptive group-size selector (`arXiv:2611.33064`) + token-span credit-shaping helper (`arXiv:2611.29936`).
- `configs/qwen3-4b-adaptive-credit-grpo.yaml` — Halo online GRPO config targeting localized layers with `num_generations: 8` baseline / adaptive prompt budget.

## Compute & Go/No-Go Criteria
- **Pilot cost (`Qwen3-4B`, 200 GRPO steps):** ~$15–$30.
- **GO:** Adaptive group sizing + token-level credit beats fixed-`G` GRPO by ≥2.0 pp pass@1 or reaches target accuracy in ≤70% of total generated tokens.
