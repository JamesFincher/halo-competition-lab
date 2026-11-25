# Experiment 15: 1-Pass RelP Causal Attribution Patching Report (`SmolLM2-135M`)

**Date:** 2026-11-25 · **Papers:** `arXiv:2508.21258` (`RelP`), `arXiv:2610.05190` (`FORGE`) · **Telemetry:** `relp_summary.json`

## 1. 1-Pass Causal Discovery (7.07 seconds total)
- **Top Causal Layers:** `[29, 27, 28]`
- **Brand-New #1 Causal Neuron Found:** **Layer 27, Neuron `#110`** (`causal_score = 0.3355` — **4× higher causal impact** on the `call` vs `refuse` logit margin than any neuron found by activation-only sniffing!).
- **Independent Confirmation:** Also rediscovered **Layer 29, Neuron `#453`** (`causal_score = 0.0868`, #3 overall).
- **Top Causal Attention Heads:** `[(24, 4), (23, 2), (24, 2), (29, 1), (28, 2)]`.

## 2. 40-Task Unseen Result (Zero RL Steps)
- **Baseline SFT:** `55.0%` exact, `57.5%` policy, `100.0%` unsafe refusal.
- **1-Pass RelP Causal Edit (`layers [29, 27]`, `st=0.45`, 0 RL steps):** **`62.5%` exact (+7.5 pp)**, **`65.0%` policy (+7.5 pp)**, **`100.0%` unsafe refusal (zero safety drop)**, `mean_reward = 0.805`.
- **Rank #1 of New Batch (8.5/10):** Fastest true-causal localizer (7s for all 46,080 neurons + 270 heads), zero safety loss, and a direct drop-in upgrade for Stage-1 Sniff.
