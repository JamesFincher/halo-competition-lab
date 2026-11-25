# Experiment 13: 1-Bit Sign-Flip & Zero-Mask False-Node Editing + Safety Bit-Flip Audit (`SmolLM2-135M`)

**Date:** 2026-11-25 · **Model:** `SmolLM2-135M-Instruct` (Halo SFT `checkpoint-36`)  
**Telemetry:** `bitflip_summary.json`, `step_metrics.jsonl`

---

## 1. Binary Ops vs Continuous Directional Projection (40 Unseen Tasks)

| Arm | Dev Gate Result | Policy Acc | Unsafe Refusal | Benign False-Refusal | Exact Compliance |
|---|---|---:|---:|---:|---:|
| Starting Halo SFT (`checkpoint-36`) | — | 57.5% | 100.0% | 90.0% | 55.0% |
| **Arm 1:** 1-Bit Zero-Mask (`w → 0`, `8–128` false nodes) | Rejected (`dev_rew 0.68→0.40–0.54`) | 57.5% | 100.0% | 90.0% | 55.0% |
| **Arm 2:** 1-Bit Sign-Flip (`w → -w`, `2–32` false nodes) | Rejected (`dev_rew 0.68→0.00` at just 2 nodes!) | 57.5% | 100.0% | 90.0% | 55.0% |
| **Arm 3:** **Continuous Rank-1 Projection** (`st=0.45`, top-64 false nodes) | **Accepted (`0.68→0.80`)** | **70.0%** | **100.0% (0 drop)** | 80.0% | **67.5% (+12.5 pp)** |
| **Arm 4:** Arm 3 + Early+Mid Disjoint RL (`[0,1,11,19,20,27]`) | `kl=0.0028` (re-entered refusal basin) | 57.5% | 100.0% | 90.0% | 55.0% |
| *Best Overall:* **Exp 11 Fork V3** (Full-strength projection `st=0.85` → Disjoint RL) | R1 accepted (`benign_fr 80%→0%`), R2 auto-stopped | **90.0%** | 90.0% | **0.0%** | **77.5%** |

**Why Projection Beats 1-Bit Ops (Superposition):** Each MLP neuron column encodes hundreds of orthogonal features. Zeroing or sign-flipping the whole column destroys all of them at once, whereas rank-1 projection (`w - s * u u^T w`) removes *only* the 1D failure direction while leaving the other 575 dimensions intact.

---

## 2. Security Audit: Safety Bit-Flip Attack (BFA) Curve (40 Unseen Tasks)

Sign-flipping (`w → -w`) the top true-safety MLP neurons in layers `[29, 28]`:

| Neurons Sign-Flipped | % of Model Weights | Unsafe Refusal Rate | JSON Valid Rate |
|---:|---:|---:|---:|
| `0` (Baseline) | `0.00000%` | `100.0%` | `100.0%` |
| `2` | `0.00086%` | `93.3%` | `95.0%` |
| `4` | `0.00171%` | `90.0%` | `92.5%` |
| **`8`** | **`0.00343%`** | **`53.3%`** | **`42.5%`** |
| **`16`** | **`0.00685%`** | **`0.0%`** | **`0.0%`** |
| `32` | `0.01370%` | `0.0%` | `0.0%` |

**Security Takeaway:** Flipping the sign bit of just **8 neurons (`0.0034%` of weights)** cuts refusal nearly in half (`53.3%`), and **16 neurons (`0.00685%`)** completely destroys both safety (`0.0%`) and JSON formatting (`0.0%`). This directly motivates **Experiment 9 (INSEP)** to distribute safety representations across layers.
