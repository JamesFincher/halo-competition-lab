# Experiment Leaderboard — Rubric-Graded (see research/RUBRIC.md)

Composite = 0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R (each /10).
Baseline: SFT micro-model = 55.0% exact / 40 unseen tasks, 100% unsafe refusal, 90% benign false refusal (raw base model: 0% exact, 0% unsafe refusal).

| Rank | Exp | Technique | Key Result | P | S | C | E | N | K | T | R | **Score** | Grade |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **17 (DLA arm)** | Direct Logit Lens top-64 edit, zero grad | **80.0% exact** (+25pp), 90% unsafe-refusal held, 0% benign FR, 2.24s | 10 | 8 | 10 | 10 | 8 | 10 | 9 | 7 | **9.25** | **S** |
| 2 | **17 (IG-5 arm)** | 5-step Integrated Gradients edit | **82.5% exact** (+27.5pp, record), 90% unsafe held, 0% benign FR, 4.72s | 10 | 8 | 10 | 9 | 8 | 9 | 9 | 7 | **9.05** | **S** |
| 3 | **11 (Fork V3)** | Re-anchored disjoint RL (edited layers frozen) | **77.5% exact** (+22.5pp) with safety preserved after RL | 9 | 8 | 8 | 5 | 8 | 10 | 8 | 6 | **8.05** | **A** |
| 4 | **18** | 7-behavior DLA fingerprint atlas (diagnostic) | Found refusal circuit collision (77.8% Jaccard) + hub neurons; enables 19/24/29 | — | — | — | 10 | 9 | 10 | 9 | 8 | **8.0** (diag) | **A** |
| 5 | **15** | 1-pass RelP causal attribution sniff | 62.5% exact zero-RL, 100% safety, found new #1 causal neuron L27#110 in 7s | 5 | 10 | 9 | 10 | 8 | 9 | 8 | 6 | **7.95** | **A** |
| 6 | **19** | 7-behavior cross-talk + hub-protection audit (diagnostic) | False-refusal edit moved true safety almost identically (-8.516 vs -8.516 protected); hub protection cut mean collateral only 8.4% — refusal circuits entangled | — | — | — | 8 | 8 | 9 | 8 | 4 | **7.6** (diag, provisional) | **A** |
| 7 | **16** | Safety-whitened rank-k subspace edit | raw_rank1: 65% exact / 0% benign FR; hard nullspace strips too much on narrow model | 6 | 8 | 10 | 8 | 9 | 7 | 6 | 6 | **7.35** | **B** |
| 8 | **12** | Head+MLP false-node double-check sniff | Found top false heads (L29H1); edit gains modest | 4 | 8 | 8 | 8 | 7 | 8 | 7 | 5 | **6.65** | **B** |
| 9 | **14** | Zero-shock stitched GQA head + RL | Zero-shock math proven (KL≈1.6e-6) but 45% exact in 6 steps — needs long training | 0 | 10 | 9 | 4 | 9 | 6 | 8 | 6 | **5.85** | **C** |
| 10 | **17 (5 anti-probes)** | Variance flicker, activation spike, random consensus | ≤ 67.5% — rejected as primary cameras | 3 | 7 | 5 | 8 | 6 | 5 | 6 | 7 | **5.5** | **C** |
| 11 | **13** | 1-bit sign-flip / zero-mask false-node edit | Both arms rejected: benign false refusal 90% — binary ablation destroys circuit | 0 | 10 | 2 | 9 | 7 | 4 | 5 | 7 | **4.9** | **D** |

## Key conclusions from back-evaluation
1. **Zero-grad logit-space localization (DLA/IG-5) dominates everything** — the two S-tier techniques are both zero-RL. Combining them with re-anchored disjoint RL (Exp 21) is the flagship composition.
2. **Binary/hard structural edits fail** (Exp 13 D-grade, Exp 16 hard nullspace): continuous directional projection >> masking on narrow models.
3. **Diagnostic atlases pay off, but hub protection alone does not decouple refusal types** — Exp 19 found true/false refusal margin movement coupled at ratio ≈1.00 even with hub neurons masked.
4. **RL alone is cost-inefficient** (Exp 11 E=5) — but it is the only path past the ~82.5% edit ceiling, so its composability (K=10) keeps it A-tier.
5. **Safety-preserving compliance repair is now the central bottleneck** — future S-tier entries need >80% exact without the current 100%→90% unsafe-refusal drop.

## Wave-1 direct-bf16 autograder output — confounded, not ranked

These rows use the same rubric table format, but they are **not canonical technique grades**. They measure the accidental direct-CUDA-bf16 regime documented in `docs/EXPERIMENT_ANALYSIS.md`: the unedited baseline itself falls to 47.5% exact, and DLA rank-1 edits often collapse into always-call behavior (0% unsafe refusal). Do not interpret these as dead ends until fp32-master reruns land.

| Rank | Exp | Technique | Key Result | P | S | C | E | N | K | T | R | **Score** | Grade |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **28** | Head+MLP synergy, direct bf16 | best arm `head_only`: 45.0% exact, 100% unsafe, 60% benign FR; all MLP arms collapse | 0 | 10 | 1 | 4 | 7 | 8 | 7 | 5 | **4.65** | **D** |
| — | **23** | Push-pull, direct bf16 + degenerate directions | best arm `pushpull_1.0`: 42.5% exact, 100% unsafe, 100% benign FR; implementation is net-sign control, not true push-pull | 0 | 10 | 1 | 4 | 7 | 8 | 7 | 5 | **4.65** | **D** |
| — | **29** | Momentum-annealed surgery+RL, direct bf16 | best arm `annealed_edit`: 30.0% exact, 100% unsafe, 90% benign FR; reward collapses when edit reaches zero | 0 | 10 | 1 | 4 | 7 | 8 | 7 | 5 | **4.65** | **D** |
| — | **25** | Gradient-free evolution, direct bf16 | heldout-test half split: 50.0% exact but only 42.9% unsafe; fitness does not hard-constrain safety | 0 | 2 | 10 | 4 | 7 | 8 | 7 | 5 | **4.40** | **D** |
| — | **26** | Entropy-weighted GRPO, direct bf16 | best composite tie shown as `uniform_adv`: 7.5% exact, 0% unsafe; post-edit start already collapsed | 0 | 2 | 10 | 4 | 7 | 8 | 7 | 5 | **4.40** | **D** |
| — | **21** | DLA edit → disjoint RL, direct bf16 | post-edit checkpoint starts collapsed; RL remains at 15.0% exact / 0% unsafe | 0 | 2 | 10 | 4 | 7 | 8 | 7 | 5 | **4.40** | **D** |
| — | **20** | DLA vs random control, direct bf16 | DLA exact 15.0% / unsafe 0%; random mean 47.5±2.5% / unsafe 100% — dtype inversion, not localization refutation | 0 | 2 | 10 | 4 | 7 | 8 | 7 | 5 | **4.40** | **D** |
| — | **30** | Steering-oracle RL, direct bf16 | plain RL and steered rollouts both end at 15.0% exact / 0% unsafe after collapsed initialization | 0 | 2 | 10 | 4 | 7 | 8 | 7 | 5 | **4.40** | **D** |

## Pending grading (running on H100 swarm)
Exps 22, 24, 27, and 31–38 are pending or awaiting valid reruns. Exp 22 completed its bf16 sweep in the log but failed before writing `pareto_sweep_summary.json`; Exp 24 crashed on a device mismatch; Exp 27's first JSON was invalid (identity patch) and has been fixed for rerun. Exp 19 is provisionally graded as diagnostic pending fp32 confirmation.
