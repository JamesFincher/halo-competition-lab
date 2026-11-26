# Experiment Analysis — Living Research Notes

Last updated: 2026-11-26 08:50 UTC. This document is updated continuously as H100 swarm results land.

## Current ingestion status

| Experiment | Folder | Status | Primary question |
|---:|---|---|---|
| 19 | `surgical_crosstalk_audit` | waiting for `crosstalk_summary.json` | Does hub-neuron protection reduce cross-behavior collateral while preserving target-behavior movement? |
| 20 | `random_node_control_proof` | waiting for `random_control_summary.json` | Does matched-budget random editing falsify or validate DLA localization? |
| 21 | `ig5_dla_closed_loop_rl` | waiting for `dla_disjoint_rl_summary.json` | Does DLA edit → re-anchored disjoint RL beat the 82.5% zero-grad edit ceiling? |
| 22 | `multi_seed_stability_sweep` | waiting for `pareto_sweep_summary.json` | Where is the exact-compliance / safety / collateral optimum across K and strength? |
| 23 | `dual_polarity_circuit_surgery` | waiting for `pushpull_summary.json` | Does suppress+amplify beat suppression-only at matched edit locality? |
| 24 | `adaptive_k_proportional_edit` | waiting for `adaptive_k_summary.json` | Does salience-proportional strength reduce collateral versus uniform top-K? |
| 25 | `gradfree_evolution_subspace` | waiting for `evolution_summary.json` | Can gradient-free search find a better surgical hyperparameter point without touching the test split? |
| 26 | `entropy_weighted_fork_rl` | waiting for `entropy_grpo_summary.json` | Does entropy-weighted credit assignment improve post-edit RL over uniform advantage? |
| 27 | `inverse_causal_patching` | waiting for `causal_patching_summary.json` | Do activation patches causally validate layers 29/28 as the minimal edit site? |
| 28 | `crosslayer_head_mlp_synergy` | waiting for `synergy_summary.json` | Does coordinated attention-head + MLP surgery beat either mechanism alone? |
| 29 | `momentum_annealed_rl` | waiting for `annealed_rl_summary.json` | Does annealing the surgical scaffold let RL own the behavior better than a fixed edit? |
| 30 | `steering_oracle_guided_rl` | waiting for `steering_rl_summary.json` | Does steered exploration distill into better unsteered weights than plain post-edit RL? |
| 31 | `request_circuit_mapper` | added in wave 2; waiting for `request_circuit_atlas.json` | Which layers/neurons carry each semantic request type, and which are type-exclusive? |
| 32 | `targeted_circuit_rl` | added in wave 2; waiting for `circuit_rl_summary.json` | Does column-masked circuit-only RL preserve safety better than full-layer RL? |
| 33 | `dtype_edit_sensitivity` | added in wave 2; waiting for `dtype_summary.json` | Does bf16 application corrupt rank-1 surgical edits relative to fp32? |
| 34 | `exp34_activation_vector_only_rl` | added in wave 2; waiting for `activation_vector_rl_summary.json` | Can activation-vector-only RL match weight-space RL while weights stay frozen? |

## Measurement baseline and interpretation rules

The intervention baseline is the merged Halo LoRA SFT checkpoint, not the raw pretrained model. On the 40-task zero-overlap held-out set:

| Model | Exact compliance | Action accuracy | Policy accuracy | Unsafe refusal | Benign false refusal | Mean reward |
|---|---:|---:|---:|---:|---:|---:|
| Raw `SmolLM2-135M-Instruct` | 0.0% (0/40) | 5.0% | 5.0% | 0.0% (0/30) | 0.0% (0/10) | — |
| Halo LoRA SFT `checkpoint-36` | **55.0% (22/40)** | 77.5% | 57.5% | **100.0% (30/30)** | **90.0% (9/10)** | — |

Measurement cautions:
- The eval split has only **10 benign** and **30 unsafe** tasks, so one flipped benign task moves benign false-refusal by 10 pp, while one flipped unsafe task moves unsafe refusal by 3.33 pp.
- Exact compliance moves in 2.5 pp increments on 40 tasks. Most single-run differences below 5 pp are only one or two tasks and should not be over-read without controls.
- `results/eval_summary.json` stores rates as fractions; experiment result JSONs store percentages. All numbers here are normalized to percentages.
- The existing leaderboard's baseline line was inconsistent with the verified SFT result: the SFT checkpoint has **90% benign false refusal**, not 0%. The raw pretrained model has 0% benign false refusal only because it almost never produces valid refusals at all.

## What the pre-swarm experiments already established

| Prior result | Exact | Unsafe refusal | Benign FR | Wall/probe time | Interpretation |
|---|---:|---:|---:|---:|---|
| Exp 17 IG-5 edit | **82.5%** | 90.0% | **0.0%** | 4.72s | Best exact score so far, but sacrifices 3/30 unsafe refusals. |
| Exp 17 DLA top-64 edit | **80.0%** | 90.0% | **0.0%** | 2.24s | Nearly same benefit at half the probe cost; safest high-performing zero-grad mechanism. |
| Exp 11 closed-loop V3 | 77.5% | 90.0% | 0.0% | multi-stage | Re-anchored disjoint RL preserves the edit behavior better than uncontrolled RL. |
| Exp 17 consensus RRF | 72.5% | 90.0% | 0.0% | 11.52s | Consensus localization works but is costlier and weaker than IG/DLA alone. |
| Exp 13 projection arm | 67.5% | 100.0% | 80.0% | — | Continuous projection preserves unsafe refusal but does not solve benign over-refusal enough for high exact compliance. |
| Exp 12 MLP false-node only | 67.5% | 100.0% | 80.0% | — | False-node signal is real but coarse; head+MLP combination did not improve over MLP alone. |
| Exp 16 raw rank-1 subspace | 65.0% | 90.0% | 0.0% | 759.16s total | Directionally useful but much less compute-efficient than DLA/IG. |
| Exp 15 1-pass RelP | 62.5% | 100.0% | 80.0% | 7.07s | Fast causal probe, useful for ranking neurons, but not a stand-alone high-exact edit. |
| Exp 13 zero-mask / sign-flip | 55.0% | 100.0% | 90.0% | — | Binary structural ablations are indistinguishable from baseline on exact compliance. |
| Exp 14 stitched head + short RL | 45.0% | 100.0% | 100.0% | 173.4s | Zero-shock construction is mathematically clean, but short RL does not learn enough and over-refusal worsens. |

### Pre-swarm mechanistic patterns

1. **Continuous directional edits dominate hard structural edits.** DLA/IG/continuous projection all beat zero-mask and sign-flip arms; the binary arms stayed at baseline exact compliance (55.0%).
2. **The best exact-compliance gains currently come with a small safety cost.** IG-5 and DLA reach 82.5%/80.0% exact with 0% benign false refusal, but unsafe refusal falls from 100% to 90% — exactly 3 of 30 unsafe tasks become non-refusals.
3. **Fixing over-refusal is not the same as fixing exact compliance.** Several arms reduce benign false refusal from 90% to 80% or 0% without reaching high exact compliance because policy-code and tool-choice errors remain.
4. **Unrestricted or poorly anchored RL tends to regress toward the SFT failure mode.** The strongest RL result is the closed-loop/re-anchored variant (77.5% exact); other short RL arms revert to 55% or worse.
5. **Localization quality matters more than edit budget alone.** Random/broad interventions are explicitly tested in Exps 20 and 22 because prior wins came from very small neuron subsets in late layers.
6. **Layer 29 is dominant but not sufficient.** The atlas shows late-layer concentration for refusal and JSON behaviors, with hub neurons shared across behaviors; Exps 19 and 27 test whether that overlap is causal collateral or merely correlational.

## Experiment-by-experiment analysis framework

### Experiment 19 — Surgical Cross-Talk & Hub Protection Audit

**Hypothesis:** DLA suppression of one behavior should move low-Jaccard behaviors weakly and high-Jaccard neighbors strongly; masking hub neurons `(L29,#1512)`, `(L28,#1095)`, `(L29,#1022)`, `(L27,#110)` should reduce collateral without eliminating target movement.

**What to inspect when results land:** per-target `target_margin_delta`, `mean_collateral_abs_delta`, protected-vs-unprotected deltas, and specifically sycophancy/verbosity co-movement. The atlas predicts refusal behaviors are nearly identical in direction (top-128 Jaccard 0.778) while JSON/factual/math overlaps are low (mostly 0.016–0.108).

**Decision value:** if protection cuts collateral by a large fraction while preserving most target movement, hub masking becomes a default wrapper for all subsequent edits; if protection also destroys target movement, hub neurons are load-bearing and combinations need disjoint neuron budgets instead.

### Experiment 20 — Random-Node Control Proof

**Hypothesis:** DLA top-64 should beat five matched random 64-neuron controls and anti-bottom-64 by much more than seed noise.

**What to inspect when results land:** `dla_vs_random_exact_gain`, random mean/std, anti-bottom exact, and whether random seeds ever reach DLA performance. Because there are only five random seeds, treat a less-than-2σ gap as inconclusive rather than a refutation.

**Decision value:** this is the main statistical-robustness gate for neuron-level localization. A large gap upgrades all DLA-based compositions; a small gap means the operator/budget/layer choice rather than neuron identity may be doing the work.

### Experiment 21 — DLA Surgical Edit → Re-Anchored Disjoint RL

**Hypothesis:** composing DLA with frozen edited layers and gently trained complementary layers should beat either parent and potentially exceed the 82.5% ceiling.

**What to inspect when results land:** compare `after_edit` and `after_disjoint_rl` at strengths 0.65/0.80, the RL trace reward/KL/grad norm, and whether safety degrades after RL. The composition rule requires higher exact compliance than the best parent without losing more than 0.5 points on S or C.

**Decision value:** this is the flagship composition. If it fails while edit-only succeeds, subsequent RL research should focus on exploration/credit assignment (Exps 26/29/30), not simply more RL steps.

### Experiment 22 — Budget × Strength Pareto Frontier

**Hypothesis:** intermediate K/strength should dominate both under-editing and over-editing; large K or strength should increase collateral and safety damage.

**What to inspect when results land:** all 20 grid cells, not just the best exact cell. The useful output is the Pareto frontier of exact compliance versus unsafe refusal and benign false refusal. Look for plateaus near K=64/128 and nonlinear degradation at strength ≥0.8.

**Decision value:** selects the operating point for all future DLA edits and provides the strongest evidence whether Exp 17's top-64/strength-0.9 choice was optimal or accidental.

### Experiment 23 — Dual-Polarity Push-Pull Circuit Surgery

**Hypothesis:** suppressing refusal while amplifying compliance should beat suppression-only by increasing benign-action margin without needing more neurons.

**What to inspect when results land:** amplification dose response at 0.3/0.5/1.0 relative to suppress-only. Watch whether higher amplification raises exact compliance but lowers unsafe refusal — that would indicate indiscriminate anti-refusal rather than benign-specific repair.

**Decision value:** if low amplification dominates, push-pull is a cheap drop-in modification to every rank-1 edit; if high amplification is required, it is less composable because safety margins become fragile.

### Experiment 24 — Adaptive-K Proportional Editing

**Hypothesis:** scaling strength by salience should preserve high-value neuron edits while reducing collateral from weak tail neurons.

**What to inspect when results land:** uniform vs linear/sqrt/square at K=64/128, with attention to benign false refusal and unsafe refusal. `sqrt` compresses salience differences and should behave closest to uniform; `square` is the sparsest effective edit and should best test proportional editing.

**Decision value:** if proportional editing matches uniform exact compliance at lower collateral, it becomes the default edit operator; if uniform wins outright, salience magnitude is not a reliable strength controller.

### Experiment 25 — Gradient-Free Evolution in 8D Surgical Subspace

**Hypothesis:** dev-split evolutionary search can discover better layer-wise K/strength/mixing hyperparameters without gradient training.

**What to inspect when results land:** dev-test gap, final `test` metrics on only 20 held-out tasks, best parameter values, and whether evolution chooses a third layer (`l3`, `s3`) or stays concentrated on layers 29/28. A large dev-test gap means the search overfit the 20-row dev split.

**Decision value:** if test exact beats manually selected DLA operating points while preserving safety, gradient-free hyperparameter search becomes a cheap outer loop around every surgical method.

### Experiment 26 — Entropy-Weighted Decision-Fork GRPO

**Hypothesis:** entropy-weighted token credit should improve RL efficiency by focusing updates at uncertain decision points.

**What to inspect when results land:** post-edit pre-RL baseline, uniform vs entropy-weighted final evals, and per-step loss traces. Since both RL arms start from the same edited model, improvement over uniform is more meaningful than improvement over the pre-RL baseline alone.

**Decision value:** if entropy weighting wins, it is orthogonal to model localization and can combine with Exps 21/29/30; if it loses, token-level uncertainty is probably not the bottleneck in this tiny strict-JSON task.

### Experiment 27 — Inverse Causal Patching Localization

**Hypothesis:** patching compliant activations into a falsely refusing prompt should produce the largest margin restoration in late layers if layers 29/28 are causal rather than merely correlated.

**What to inspect when results land:** ranked `layer_deltas`, the size separation between top layers and the rest, and whether the top causal set includes 29/28 or points earlier. This experiment uses a single patch pair, so even a clean ranking is hypothesis-generating rather than statistically strong.

**Decision value:** if layers 29/28 are validated, future edits can stay narrow; if earlier layers dominate, the edit layer set must expand and Exp 22's layer-fixed sweep is incomplete.

### Experiment 28 — Cross-Layer Head+MLP Synergy Edit

**Hypothesis:** jointly suppressing L29H1 and top MLP neurons should beat either mechanism alone at matched total budget.

**What to inspect when results land:** `head_only`, `mlp64_only`, half/half synergy, and full synergy. Prior Exp 12 already showed head+false-node no better than MLP-only at 67.5% exact, so a positive result here must exceed that precedent, not merely beat head-only.

**Decision value:** a true synergy would justify mixing mechanisms across attention and MLP; no synergy means attention-head signal is mostly redundant or harmful once DLA-selected MLP neurons are edited.

### Experiment 29 — Momentum-Annealed Surgery+RL

**Hypothesis:** a strong temporary edit should bootstrap high-reward rollouts, then hand off to learned weights as the edit anneals toward zero.

**What to inspect when results land:** fixed-edit vs annealed final evals, per-step edit strength/mean reward, and whether performance persists after the edit reaches zero. A final annealed win with low KL would be strong evidence for scaffolded RL; regression after annealing means the manual edit was doing the work rather than teaching the model.

**Decision value:** this is one of the most important composition mechanisms because it could reduce permanent collateral while retaining RL generalization.

### Experiment 30 — Steering-Oracle Guided RL

**Hypothesis:** anti-refusal steering during rollout generation should expose high-reward benign trajectories, which RL can distill into unsteered weights.

**What to inspect when results land:** plain RL vs steered rollouts, per-step mean reward, and post-training eval without steering. Also watch unsafe refusal: oracle steering toward compliance can easily reward unsafe compliance unless the verifier penalizes it strongly enough.

**Decision value:** if successful, steering is an exploration-only add-on with no inference-time cost; if not, the main RL bottleneck is credit assignment or KL anchoring rather than exploration.

## Cross-experiment combination priorities going into the swarm

1. **Highest-priority composition:** Exp 20-validated DLA neuron selection → Exp 22-selected K/strength → Exp 21 re-anchored disjoint RL.
2. **Best zero-grad stack candidate:** hub protection (Exp 19) + proportional strength (Exp 24) + low-dose push-pull (Exp 23), but only if each preserves unsafe refusal.
3. **Best RL stack candidate:** Exp 21 anchoring/freezing + the winning exploration/credit mechanism from Exps 26/29/30.
4. **Best diagnostic loop:** Exp 27 causal patch validation before editing + Exp 19 cross-talk audit after editing + Exp 20 random-control validation for any new localization claim.

## Open risks and contradictions to resolve

- **Safety trade-off:** the current S-tier edits improve benign compliance by sacrificing exactly 10 pp of unsafe refusal. The competition objective likely cannot tolerate that unless explicitly weighted; future winners need 100% unsafe refusal at >80% exact.
- **Small eval split:** there are only 10 benign tasks, so 0% benign false refusal can coexist with many benign policy/tool errors. Benign FR is necessary but not sufficient.
- **Causal evidence is thin:** activation patching in Exp 27 uses one source/target pair; negative results should not kill the layer hypothesis by themselves.
- **RL traces are short:** all RL arms use 8 micro-steps. A negative RL result may mean insufficient optimization rather than a bad mechanism, but positive results under this constraint are especially compute-efficient.
- **Autograder coverage:** the first version of `scripts/rubric_autograder.py` handles standard `arms`, `grid`, and `runs` schemas, but diagnostic schemas (Exps 19/27) and evolution/test schema (Exp 25) require careful interpretation rather than blind exact-compliance extraction.

## Result log

### Experiment 19 — Surgical Cross-Talk & Hub Protection Audit — preliminary diagnostic landed

**Input:** `experiments/surgical_crosstalk_audit/results/crosstalk_summary.json`, CUDA, wall time **45.35s**. This was a wave-1 bf16-era run, so treat the exact magnitudes as provisional until the fp32 wave-2 validation program confirms them.

**What happened quantitatively:**

| Target behavior | Hub protection | Target margin delta | Mean collateral abs delta | Largest collateral effect |
|---|---|---:|---:|---|
| False over-refusal | no | **-9.2031** | **1.6094** | true safety refusal **-9.2188** |
| False over-refusal | yes | **-8.5156** | **1.4748** | true safety refusal **-8.5157** |
| Sycophancy bias | no | **-5.0416** | **0.3229** | verbosity **-1.0208**, JSON **+0.7500** |
| Sycophancy bias | yes | **-4.9166** | **0.3082** | verbosity **-0.9792**, JSON **+0.7500** |
| Verbosity/hedging | no | **-5.3125** | **0.4800** | JSON **+1.9791**, sycophancy **-0.6458** |
| Verbosity/hedging | yes | **-5.2917** | **0.4574** | JSON **+1.8750**, sycophancy **-0.6666** |

**Hypothesis verdict:** mostly confirmed for cross-talk, but not for hub protection as a useful fix. The atlas predicted refusal-behavior overlap; the measured false-refusal edit moved true safety refusal by essentially the same amount as the target (ratio **1.0017** unprotected, **1.0000** protected). Sycophancy and verbosity also co-moved as predicted, but much more weakly. Hub protection reduced mean collateral only slightly — **8.36%** for false over-refusal, **4.55%** for sycophancy, **4.71%** for verbosity — while attenuating target movement by **7.47%**, **2.48%**, and **0.39%** respectively. For the critical refusal case, protection is almost proportional attenuation rather than selective decoupling.

**Interpretation:** the key negative result is that `false_over_refusal` and `true_safety_refusal` are not separable by simply excluding the four atlas hub neurons. This explains why the best DLA/IG edits trade 0% benign false refusal for a drop from 100% to 90% unsafe refusal: the current refusal-direction edit is acting on a shared refusal circuit, not a false-only circuit.

**Implication for combinations:** suppression-only refusal editing is probably capped unless paired with either (a) a compliance-direction push that does not erase unsafe refusal, (b) request-type-specific neuron targeting from Exp 31, or (c) a safety-restoration arm. Hub protection should not become a default wrapper yet.

**Recommended follow-up:** rerun in fp32 after Exp 33 quantifies dtype sensitivity, then test a two-direction objective: suppress false-refusal neurons while explicitly preserving or amplifying true-safety refusal margins on unsafe prompts. A margin-only audit is not enough; the follow-up must report the standard 40-task exact/unsafe/benign-FR metrics.

### Experiment 27 — Inverse Causal Patching Localization — invalid first result, not gradeable

**Input:** `experiments/inverse_causal_patching/results/causal_patching_summary.json`, CUDA, wall time **5.29s**.

Every layer has `margin_after_patch = 3.125` and `delta = 0.0`, including layers 27–29. This is not evidence that no layer is causal. The result is an identity-patch artifact: `pos_rows[0]` is the minimum-margin prompt, and `neg_rows[-1]` is also the minimum-margin prompt because `neg_rows` is sorted in the opposite direction. The source and target are therefore the same row, so patching activations cannot change the margin.

**Hypothesis verdict:** no verdict. The experiment did not test the hypothesis.

**Interpretation:** the all-zero table is internally consistent with source=target aliasing. The later hook-shape and fp32 fixes were necessary but insufficient; the source/target selection logic is the blocking bug.

**Recommended follow-up:** rerun with the minimum-margin benign prompt as source and the maximum-refusal-margin benign prompt as target, then expand beyond one pair. A strong version should use at least 3 source/target pairs, report mean ± std layer restoration, and include a same-prompt identity control explicitly labeled as such.

### Cross-result update after Exps 19 + 27

1. The largest confirmed mechanistic constraint is now **true/false refusal entanglement**, not merely neuron-budget selection. DLA can find refusal-circuit neurons, but the shared circuit makes safety-preserving suppression hard.
2. The highest-value near-term experiment is therefore not another uniform refusal-suppression sweep; it is a method that separates request type or direction: Exp 23 push-pull, Exp 31 type-specific circuits, or a safety-constrained edit.
3. Exp 27 must be rerun before layer causality evidence can inform the layer choice. Until then, layers 29/28 remain empirically useful but not causally validated.
