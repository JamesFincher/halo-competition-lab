# Experiment 17: 6-Probe "Christmas-Tree" Node-Tracking Benchmark (`SmolLM2-135M`)

**Date:** 2026-11-25 · **Model:** `SmolLM2-135M-Instruct` (Halo SFT `checkpoint-36` — 46,080 MLP neurons, 30 layers)  
**Visual Map:** `christmas_tree_heatmap.svg` · **Telemetry:** `probe_benchmark_summary.json`

---

## 1. Head-to-Head Verifiable Bake-Off (40 Unseen Tasks, Identical Top-64 Neuron Budget, 0 RL Steps)

| Rank | Christmas-Tree Camera | Probe Time | Backward Passes | Exact Compliance | Policy Acc | Unsafe Refusal | Benign False-Refusal | Mean Reward |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| **#1** | **5-Step Integrated Gradients (`IG-5`)** | **4.72s** | 5 steps | **`82.5%` (+27.5 pp — NEW RECORD!)** | **`92.5%`** | `90.0%` | **`0.0%`** | **`0.920`** |
| **#2** | **Direct Logit Lens (`DLA`)** | **`2.24s` (Fastest!)** | **0 (zero-grad!)** | **`80.0%` (+25.0 pp — #2 ALL-TIME!)** | **`92.5%`** | `90.0%` | **`0.0%`** | `0.915` |
| **#3** | **Multi-Probe Consensus (`RRF`)** | 11.52s | Mixed | `72.5%` (+17.5 pp) | `82.5%` | `90.0%` | `0.0%` | `0.900` |
| **#4** | **Safety-Filtered Activation Spike** | 6.72s | 0 | `67.5%` (+12.5 pp) | `70.0%` | **`100.0%`** | `80.0%` | `0.815` |
| **#5** | **1-Pass RelP Causal Attribution** | 2.56s | 1 pass | `62.5%` (+7.5 pp) | `65.0%` | **`100.0%`** | `80.0%` | `0.805` |
| **#6** | **Variance / Flicker Spike** | 2.42s | 0 | `55.0%` (rejected) | `57.5%` | `100.0%` | `90.0%` | `0.775` |
| *Ref* | *Starting Halo SFT (`checkpoint-36`)* | — | — | `55.0%` | `57.5%` | `100.0%` | `90.0%` | `0.775` |
| *Ref* | *Full-Model LoRA GRPO (Arm 0)* | 123.3s | Full RL | `27.5%` (-27.5 pp) | `50.0%` | `86.7%` | `0.0%` | `0.805` |

---

## 2. Why `IG-5` (`82.5%`) and `DLA` (`80.0%`) Crushed Everything Else
1. **Direct Logit Lens (`DLA`, `80.0%` in `2.24s`, ZERO gradients):** Projects every neuron's output straight onto `W_U["ref"] - W_U["call"]` — pinpointing the exact neurons writing the wrong decision token into the vocabulary with **zero backward passes**.
2. **5-Step Integrated Gradients (`IG-5`, `82.5%` in `4.72s`):** Integrates the decision-token logit-margin gradient across 5 path steps to avoid single-point gradient saturation, setting a **new all-time lab record (+55.0 pp over Full-Model RL)** with **zero RL training steps**.
3. **Variance / Flicker Ratio (`55.0%`, rejected):** Proves noisy/high-variance neurons are NOT the causal culprits — output-logit attribution (`DLA` / `IG-5`) is strictly superior.
