# Experiment 18 Report: 7-Behavior Neural Fingerprint Atlas & Circuit-Overlap Cartography

**Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (30 layers, 46,080 MLP neurons, `d=576`)  
**Camera:** Zero-Gradient Direct Logit Attribution (`DLA`, `7.03s` total for all 7 behaviors)  
**Visual Atlas:** `behavior_fingerprint_atlas.svg`

## 1. 7-Behavior Location Fingerprint Table

| # | Behavior | Contrast Tokens | Depth CoM | Peak-3 Layers | #1 Signature Neuron (DLA) | Secondary Signature Neurons |
|---:|---|---|---:|---|---|---|
| **1** | **True Safety Refusal** | `'ref'` vs `'call'` | `L22.20` | `[29, 27, 28]` | **`L27 #110` (`44.08`)** | `L29 #1512`, `L29 #545`, `L29 #676`, `L29 #427` |
| **2** | **False Over-Refusal** | `'ref'` vs `'call'` | `L22.31` | `[29, 27, 28]` | **`L27 #110` (`50.44`)** | `L29 #1512`, `L29 #545`, `L29 #676`, `L29 #427` |
| **3** | **Strict JSON Syntax** | `'{'` vs `'Sure'` | `L21.90` | `[29, 22, 28]` | **`L29 #1128` (`21.33`)** | `L29 #1512`, `L29 #1022`, `L29 #347`, `L29 #748` |
| **4** | **Sycophancy Bias** | `'Yes'` vs `'No'` | `L21.26` | `[29, 28, 21]` | **`L28 #905` (`11.01`)** | `L21 #1314`, **`L29 #778`**, `L29 #1392`, `L28 #1095` |
| **5** | **Arithmetic Reasoning** | `'7'` vs `'9'` | `L21.06` | `[29, 28, 27]` | **`L29 #1022` (`2.44`)** | `L29 #1123`, `L29 #1513`, `L29 #427`, `L24 #467` |
| **6** | **Factual Recall** | `'Paris'` vs `'London'` | `L22.04` | `[29, 28, 27]` | **`L28 #1095` (`11.93`)** | `L29 #269`, `L29 #1040`, `L29 #1042`, `L29 #1498` |
| **7** | **Verbosity / Hedging** | `'Sure'` vs `'The'` | `L21.15` | `[29, 27, 23]` | **`L29 #778` (`15.43`)** | `L29 #684`, `L23 #87`, `L23 #863`, `L26 #306` |

## 2. Three Key Cartography Discoveries

1. **Same Building (Layers `L27–L29`), Different Rooms (Neurons):**
   - At the **layer** level, all 7 behaviors look almost identical (`layer_cosine = 0.90–0.999`, all peaking in `L27–L29` with Depth CoM `L21.1–L22.3`).
   - At the **neuron** level, unrelated behaviors are **nearly disjoint** (`top-128 neuron Jaccard = 1.6% – 4.9%`). This explains why **layer-level** full-model updates cause collateral forgetting while **neuron-level** surgical edits can isolate a single behavior.
2. **Two Real Circuit Collisions Discovered:**
   - **True Safety vs False Over-Refusal (`77.8%` neuron Jaccard, `0.999` layer cosine):** Share the exact same top-5 neurons led by `L27 #110` (`44.08` vs `50.44`). That is why binary neuron ablation destroys safety (Exp 13), whereas directional projection (`IG-5`/`DLA`, Exp 17) separates them to `82.5%` exact compliance.
   - **Sycophancy vs Verbosity / Hedging (`10.8%` neuron Jaccard — 2×–6× higher than any other distinct pair):** Both share signature neuron **`L29 #778`** (`15.43` on Verbosity, `9.19` on Sycophancy), showing that polite filler and agreeable sycophancy share an overlapping late-layer sub-circuit.
3. **Polysemantic "Hub" Neurons Flagged (Do Not Zero-Out!):**
   - `L29 #1512` (Refusal + Strict JSON), `L28 #1095` (Factual Recall + Sycophancy), `L29 #1022` (Arithmetic + Strict JSON).
