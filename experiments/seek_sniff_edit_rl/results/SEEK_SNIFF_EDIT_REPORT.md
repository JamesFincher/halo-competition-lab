# Experiment 11: Closed-Loop Seek → Offset-Sniff → Gated Edit → Disjoint RL Report (`SmolLM2-135M`)

**Date:** 2026-11-25 · **Hardware:** 8-Core CPU (4 parallel workers, `SmolLM2-135M-Instruct`, starting from Halo v1.3.0rc1 `checkpoint-36`)  
**Telemetry:** [`summary_v1.json`](summary_v1.json), [`summary_v2.json`](summary_v2.json), [`summary_v3.json`](summary_v3.json), [`summary_v4.json`](summary_v4.json), `step_metrics_v*.jsonl`

---

## 1. Key Scientific Discovery: Token-Aligned Canonical Sniff vs Noisy Generation Sniff

Running 4 forks in parallel immediately uncovered **why** some contrastive weight probes fail while others succeed:

| Sniff Contrast Method | Layer-29 Overlap with Real Safety (`safety_cos`) | Dev Gate Result (`dev_rew`) | Held-Out Exact Compliance |
|---|---:|---|---:|
| Raw variable-length generated refusals (`p_ben + gen`) | `0.815` (smeared across varying lengths) | Rejected (`0.680 → 0.680`) | `55.0%` (no gain) |
| **Token-aligned canonical contrast template** | **`0.561` (much cleaner separation!)** | **Accepted (`0.680 → 0.860`, `benign_fr: 80% → 0%`)** | **`77.5%` (+22.5 pp / +50 pp vs Full RL!)** |

---

## 2. Full 40-Task Unseen Held-Out Scoreboard

| Variant | Edit Gate Accepted | Final KL | Mean Reward | JSON Valid | Policy Acc | Unsafe Refusal | Benign False-Refusal | Exact Compliance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Starting Halo SFT (`checkpoint-36`) | — | — | 0.775 | 100.0% | 57.5% | 100.0% | 90.0% | 55.0% |
| Arm 0: Full-Model LoRA GRPO (baseline) | — | 0.0394 | 0.805 | 97.5% | 50.0% | 86.7% | 0.0% | 27.5% |
| Arm 4: Random-Layer Control (baseline) | — | 0.0123 | 0.905 | 100.0% | 70.0% | 100.0% | 10.0% | 60.0% |
| Arm 3b: 1-Round Fixed Edit-Then-Target | Manual | 0.0023 | **0.920** | 100.0% | **90.0%** | 93.3% | **0.0%** | 72.5% |
| **Fork V1 (Canonical):** 2-Round Pure Seek-Sniff-Edit (0 RL steps) | R1=Yes, R2=No | — | 0.890 | **100.0%** | 80.0% | 90.0% | **0.0%** | 67.5% |
| **Fork V3 (Flagship Closed Loop):** 2-Round `Seek → Sniff → Gated Edit → Disjoint RL` | R1=Yes (`st=0.85`), R2=No (`0/8` fails left!) | **0.0016** | **0.910** | **100.0%** | **90.0%** | 90.0% | **0.0%** | **77.5% (#1 RL!)** |

---

## 3. Why the Closed Loop Worked (Plain English)
1. **Round 1 Seek** caught **7/8** safe requests being falsely blocked.
2. **Round 1 Token-Aligned Sniff + Gate** automatically tested candidate edit strengths on a small dev slice and accepted `st=0.85` on layers `[29, 28]`, dropping dev false blocks **80% → 0%** with **100% unsafe refusal**.
3. **Round 1 Disjoint RL** (3 gentle steps on complementary layers, KL anchored to edited model) locked in the gain with tiny KL (`0.0019`).
4. **Round 2 Re-Seek** found **0/8 false blocks remaining** — so the gate automatically skipped extra weight surgery (`r2_edit_acc=False`) and ran 3 final polishing RL steps (`kl=0.0016`), setting our **new all-time RL exact-compliance record (`77.5%`, +50 pp over Full-Model RL, +17.5 pp over Random Control, +5 pp over Arm 3b)**.
