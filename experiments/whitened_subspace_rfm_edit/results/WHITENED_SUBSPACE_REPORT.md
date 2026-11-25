# Experiment 16: Safety-Whitened Rank-k Subspace Projection Report (`SmolLM2-135M`)

**Date:** 2026-11-25 · **Papers:** `arXiv:2607.02396` (`RFM-AGOP`), `arXiv:2603.27518`, `arXiv:2605.05220` (`MidSteer`), `arXiv:2610.05190` (`FORGE`) · **Telemetry:** `whitened_subspace_summary.json`

## 1. 40-Task Unseen Scoreboard (Zero RL Steps, `layers [29, 28]`)

| Mode | Rank `k` | Safety Nullspace `m` | Accepted `st` | Policy Acc | Unsafe Refusal | Benign False-Refusal | Exact Compliance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline SFT | — | — | — | 57.5% | 100.0% | 90.0% | 55.0% |
| **`raw_rank1`** | 1 | 0 | **0.90** | **80.0%** | 90.0% | **0.0%** | **65.0% (+10.0 pp)** |
| `safety_null_rank1` | 1 | 1 | 0.90 | 50.0% | 100.0% | 90.0% | 47.5% |
| `safety_null_rank2–4` | 2–4 | 1–2 | 0.00 (rejected) | 57.5% | 100.0% | 90.0% | 55.0% |

## 2. Key Geometric Finding
On a narrow 135M model (`d_model = 576`), hard Gram-Schmidt orthogonalization against the top true-safety subspace (`I - Q_s Q_s^T`) strips too much of the false-refusal direction because the two directions overlap heavily (`cos = 0.56–0.85`). **Soft false-node filtering (Exp 12: `67.5%` at `100%` safety) or raw Rank-1 + Disjoint RL (Exp 11 Fork V3: `77.5%`) beats hard nullspace projection at 135M**, while hard multi-rank nullspace (`RFM-AGOP`) should be re-tested on wider `Qwen3-4B` (`d = 2560`) where subspaces separate cleanly.
