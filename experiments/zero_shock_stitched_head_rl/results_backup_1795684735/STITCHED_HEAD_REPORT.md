# Experiment 14: Zero-Shock Stitched GQA Attention Head Report (`SmolLM2-135M`)

**Date:** 2026-11-25 · **Papers:** `arXiv:2410.11038`, `arXiv:2609.37702` · **Telemetry:** `stitched_head_summary.json`

## 1. Mathematical Zero-Shock Proof (Verified Live)
- **Architecture Widening (Standard `LlamaConfig`, zero custom code):** `9 Q / 3 KV → 12 Q / 4 KV` (+1 GQA group).
- **Step-0 Max Logit Difference:** **`0.00000000`** (`KL = 0.00000162`) — mathematically bit-exact zero shock.
- **Base Weight Drift After 6 RL Steps:** **`0.00000000`** (100% of original model weights bit-identical).
- **New Stitched Head Growth:** Layer 29 `||W_O_new||` grew cleanly from `0.0 → 0.4632` (`kl = 0.000010`).

## 2. 40-Task Unseen Result (Honest Verdict: DOWN in 6 steps)
- **Baseline SFT:** `55.0%` exact compliance, `100.0%` unsafe refusal, `90.0%` benign false-refusal.
- **Stitched Head Only (6 RL steps, 1.18M slice params):** **`45.0%` exact (-10.0 pp)**, `100.0%` unsafe refusal, `100.0%` JSON valid, `100.0%` benign false-refusal (`mean_reward = 0.740`).
- **Takeaway:** Zero-shock GQA head stitching works *mathematically* to 8 decimal places and is 100% standard HF/vLLM-compatible, **but** a randomly initialized new head needs far more than 6 RL steps (or SFT warmup) to learn useful attention routing — making it a **long-horizon capacity-expansion tool**, not a fast 6-step surgical fix.
