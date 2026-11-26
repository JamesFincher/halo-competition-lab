# Experiment Leaderboard — Rubric-Graded (see research/RUBRIC.md)

Composite = 0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R (each /10).
Baseline: SFT micro-model = 55.0% exact / 40 unseen tasks, 100% unsafe refusal, 0% benign false refusal.

| Rank | Exp | Technique | Key Result | P | S | C | E | N | K | T | R | **Score** | Grade |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **17 (DLA arm)** | Direct Logit Lens top-64 edit, zero grad | **80.0% exact** (+25pp), 90% unsafe-refusal held, 0% benign FR, 2.24s | 10 | 8 | 10 | 10 | 8 | 10 | 9 | 7 | **9.25** | **S** |
| 2 | **17 (IG-5 arm)** | 5-step Integrated Gradients edit | **82.5% exact** (+27.5pp, record), 90% unsafe held, 0% benign FR, 4.72s | 10 | 8 | 10 | 9 | 8 | 9 | 9 | 7 | **9.05** | **S** |
| 3 | **11 (Fork V3)** | Re-anchored disjoint RL (edited layers frozen) | **77.5% exact** (+22.5pp) with safety preserved after RL | 9 | 8 | 8 | 5 | 8 | 10 | 8 | 6 | **8.05** | **A** |
| 4 | **18** | 7-behavior DLA fingerprint atlas (diagnostic) | Found refusal circuit collision (77.8% Jaccard) + hub neurons; enables 19/24/29 | — | — | — | 10 | 9 | 10 | 9 | 8 | **8.0** (diag) | **A** |
| 5 | **15** | 1-pass RelP causal attribution sniff | 62.5% exact zero-RL, 100% safety, found new #1 causal neuron L27#110 in 7s | 5 | 10 | 9 | 10 | 8 | 9 | 8 | 6 | **7.95** | **A** |
| 6 | **16** | Safety-whitened rank-k subspace edit | raw_rank1: 65% exact / 0% benign FR; hard nullspace strips too much on narrow model | 6 | 8 | 10 | 8 | 9 | 7 | 6 | 6 | **7.35** | **B** |
| 7 | **12** | Head+MLP false-node double-check sniff | Found top false heads (L29H1); edit gains modest | 4 | 8 | 8 | 8 | 7 | 8 | 7 | 5 | **6.65** | **B** |
| 8 | **14** | Zero-shock stitched GQA head + RL | Zero-shock math proven (KL≈1.6e-6) but 45% exact in 6 steps — needs long training | 0 | 10 | 9 | 4 | 9 | 6 | 8 | 6 | **5.85** | **C** |
| 9 | **13** | 1-bit sign-flip / zero-mask false-node edit | Both arms rejected: benign false refusal 90% — binary ablation destroys circuit | 0 | 10 | 2 | 9 | 7 | 4 | 5 | 7 | **4.9** | **D** |
| 10 | **17 (5 anti-probes)** | Variance flicker, activation spike, random consensus | ≤ 67.5% — rejected as primary cameras | 3 | 7 | 5 | 8 | 6 | 5 | 6 | 7 | **5.5** | **C** |

## Key conclusions from back-evaluation
1. **Zero-grad logit-space localization (DLA/IG-5) dominates everything** — the two S-tier techniques are both zero-RL. Combining them with re-anchored disjoint RL (Exp 21) is the flagship composition.
2. **Binary/hard structural edits fail** (Exp 13 D-grade, Exp 16 hard nullspace): continuous directional projection >> masking on narrow models.
3. **Diagnostic atlases pay off** (Exp 18 A-grade as enabler) — hub-neuron protection and cross-talk predictions feed directly into Exps 19/24/29.
4. **RL alone is cost-inefficient** (Exp 11 E=5) — but it is the only path past the ~82.5% edit ceiling, so its composability (K=10) keeps it A-tier.

## Pending grading (running on H100 swarm)
Exps 19–30: graded automatically into this table when the swarm completes (each run emits rubric-scored JSON).
