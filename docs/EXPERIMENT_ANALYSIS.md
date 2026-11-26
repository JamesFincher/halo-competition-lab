# Experiment Analysis — Living Research Notes

Last updated: 2026-11-26 08:50 UTC. This document is updated continuously as H100 swarm results land.

## Current ingestion status

| Experiment | Folder | Status | Primary question |
|---:|---|---|---|
| 19 | `surgical_crosstalk_audit` | result landed; diagnostic, provisional | Hub protection barely decouples true/false refusal. |
| 20 | `random_node_control_proof` | result landed; operator-invalid | DLA is causally potent but destructive under unembed-direction edit. |
| 21 | `ig5_dla_closed_loop_rl` | result landed; operator-invalid | RL cannot recover a collapsed always-call reference. |
| 22 | `multi_seed_stability_sweep` | log complete; summary JSON missing | bf16 dose-collapse visible in log; needs valid rerun/output. |
| 23 | `dual_polarity_circuit_surgery` | result landed; mathematically degenerate | Apparent push-pull is a net sign/strength control. |
| 24 | `adaptive_k_proportional_edit` | crashed on device mismatch; rerun pending | No evidence yet. |
| 25 | `gradfree_evolution_subspace` | result landed; operator/objective-invalid | Fitness rewards unsafe compliance unless hard-constrained. |
| 26 | `entropy_weighted_fork_rl` | result landed; operator-invalid | RL arms start from always-call collapse. |
| 27 | `inverse_causal_patching` | first JSON invalid; rerun pending | Identity patch artifact: all layer deltas are exactly zero. |
| 28 | `crosslayer_head_mlp_synergy` | result landed; operator-invalid | Head-only preserves safety but does not improve exactness. |
| 29 | `momentum_annealed_rl` | result landed; operator-invalid | No evidence of scaffold-to-RL handoff. |
| 30 | `steering_oracle_guided_rl` | result landed; operator-invalid | No measurable steering benefit after collapsed initialization. |
| 31 | `request_circuit_mapper` | result landed; invalid classifier | Only two pseudo-types, not real policy-code categories. |
| 32 | `targeted_circuit_rl` | result landed; invalid downstream | Invalid atlas + broken operator stack. |
| 33 | `dtype_edit_sensitivity` | result landed; hypothesis largely refuted | fp32 and bf16 both collapse; dtype is not root cause. |
| 34 | `exp34_activation_vector_only_rl` | pending | Can activation-vector-only RL match weight-space RL while weights stay frozen? |
| 35 | `interference_canceling_edit` | result landed; operator-invalid | Cancellation partially restores exactness but not safety. |
| 36 | `safety_vaccination_edit` | result landed; negative under broken operator | No vaccine dose response. |
| 37 | `collision_priced_allocation` | result landed; negative under broken operator | Selection changes but behavior does not. |
| 38 | `noncollinear_pushpull` | result landed; still degenerate | Reported direction cosine remains -1.0. |
| 39 | `operator_sensitivity_showdown` | result landed; pivotal control | Unembed-direction edit is destructive; original operator not yet replicated. |
| 40 | `fp32_master_edit_pipeline` | result landed; hypothesis refuted | fp32-master does not rescue the wrong operator. |
| 41 | `original_operator_replication` | queued/running; decisive validation pending | Does the exact original contrastive-sniff + gated operator reproduce 80%+ exact on GPU? |

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

## Wave-1 validity warning: CUDA bf16 is a different experimental regime

The first intervention wave is valuable, but mostly **not as a test of the stated hypotheses**. Every wave-1 intervention JSON that used direct CUDA bf16 rank-1 editing must be interpreted as a numerical/dtype stress test until the fp32-master pipeline results land.

Quantitative evidence:

- Canonical fp32/CPU SFT baseline: **55.0% exact**, 57.5% policy, 100% unsafe refusal, 90% benign false refusal.
- Same SFT checkpoint in the CUDA bf16 swarm baseline: **47.5% exact**, 50.0% policy, 100% unsafe refusal, 80% benign false refusal. The unedited baseline itself shifted by **-7.5 pp exact** and **-5.0 pp policy**.
- Prior fp32 DLA editing reached **80.0% exact / 90% unsafe / 0% benign-FR**. The direct CUDA bf16 DLA top-64 edit in Exp 20 produced **15.0% exact / 0% unsafe / 0% benign-FR**.
- Exp 20's matched random controls did **not** collapse: mean exact **47.5% ± 2.5%**, range **45–50%**, unsafe refusal **100%** in all five seeds. Thus the DLA-selected columns are behaviorally potent, but in the bf16 regime their effect is catastrophically signed/shaped rather than beneficial.
- The Exp 20 per-task generations show the collapse mode precisely: all 40 outputs remain valid JSON, but the model almost always emits `action=call_tool` and `policy_code=ALLOW`; it gets **6/10 benign** exact but **0/30 unsafe** exact. Frequent outputs are `(call_tool, run_shell, ALLOW)` 13 times, `(call_tool, null, ALLOW)` 12 times, and `(call_tool, run_sql, ALLOW)` 7 times.

Consequences:

1. Do **not** treat the wave-1 D grades from the raw autograder as evidence that Exps 20/21/23/25/26/28/29/30 are dead ends. They primarily measure direct-bf16 edit failure.
2. RL arms that start from the collapsed edit are anchored to a broken checkpoint. Their failures do not falsify re-anchored RL, entropy weighting, annealing, or steering.
3. The accidental wave-1 result is still scientifically important: edit implementation precision is a first-order competition variable. Exps 33/34 (dtype sensitivity and fp32-master edit → bf16 cast) are now gating experiments for every surgical technique.
4. Any future leaderboard entry should state whether the edit was direct-bf16, fp32-master, or fp32-eval. Mixing these regimes in one ranking would be misleading.

**Correction after Exps 33/39/40 (supersedes the provisional dtype-centered interpretation above):** fp32 is not the root cause. In Exp 40 even fp32 edit + fp32 eval collapses to **12.5% exact / 0% unsafe**, and Exp 33's fp32-vs-bf16 max weight deltas are nearly identical (**0.065996** vs **0.066406**). Exp 39 identifies the stronger replication divergence: swarm edits project along the unembedding logit direction, while the original successful edits used measured residual false-refusal directions plus double-check strength gating. The unembedding direction and measured residual direction are almost orthogonal at layer 29 (**cosine 0.0039**). The correct current gate is therefore **operator replication**, not dtype alone.

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

### Experiment 20 — Random-Node Control Proof — wave-1 result is a dtype/control inversion, not a localization refutation

**Input:** `random_control_summary.json`, CUDA, wall **435.37s**.

| Arm | Exact | Policy | Unsafe refusal | Benign FR | Mean reward |
|---|---:|---:|---:|---:|---:|
| CUDA bf16 baseline | 47.5% | 50.0% | 100.0% | 80.0% | 0.775 |
| DLA top-64 | **15.0%** | 25.0% | **0.0%** | 0.0% | 0.440 |
| Anti bottom-64 | 45.0% | 47.5% | 100.0% | 90.0% | 0.755 |
| Random mean, 5 seeds | **47.5% ± 2.5%** | 50.5% mean | 100.0% all seeds | 70–90% | — |

Random exact scores were 45.0, 47.5, 45.0, 50.0, and 50.0. `dla_vs_random_exact_gain` is therefore **-32.5 pp**, the exact opposite of the intended beneficial-localization result.

**Hypothesis verdict:** inconclusive for the original fp32 claim; strongly positive evidence that DLA-selected neurons are causally potent. Random and anti edits barely move the bf16 baseline, while DLA flips the model into an always-call policy. But because prior fp32 DLA improved exact compliance to 80%, this is a dtype/regime inversion rather than evidence that DLA localization is luck.

**Deep output inspection:** DLA predictions collapse almost entirely to compliant/ALLOW JSON. Six exact successes are all benign; no unsafe prompt is answered correctly. This is not random degradation or JSON failure — JSON validity remains 100% — it is a targeted policy-direction inversion.

**Follow-up:** rerun the same 5-seed control under fp32-master editing. If fp32 DLA again reaches ~80% while random stays near baseline, Exp 20 becomes the strongest robustness result in the lab.

### Experiment 21 — DLA Edit → Re-Anchored Disjoint RL — confounded, plus an anchoring failure mode

**Input:** `dla_disjoint_rl_summary.json`, CUDA, wall **343.1s**.

At both edit strengths, the post-edit model is already collapsed: exact **15.0%**, action **25.0%**, policy **25.0%**, unsafe **0%**, benign-FR **0%**. After 8 disjoint-RL steps, exact stays **15.0%** at both strengths; policy only rises to **27.5%**. RL mean KL remains small (roughly 0–0.014), with grad norms mostly 0.04–0.22.

**Hypothesis verdict:** no verdict on the flagship composition. RL cannot recover because the reference checkpoint is the collapsed model and rollouts rarely explore refusal behavior. The result does reveal a real failure mode: **KL anchoring preserves whatever regime the edit creates, including a catastrophic one**. Re-anchoring is only beneficial if the pre-RL checkpoint is validated first.

**Follow-up:** add a pre-RL gate: do not train unless post-edit unsafe refusal is at least baseline and exact is above the unedited baseline. Then rerun in fp32.

### Experiment 22 — Budget × Strength Pareto Sweep — no JSON, but complete bf16 log shows dose-collapse

No `pareto_sweep_summary.json` was written because the run ended with `FileNotFoundError` while writing the file. The log nevertheless contains all 20 cells. Only the weakest cell, `k=16,strength=0.40`, partially survives: exact **37.5%**, unsafe **46.7%**, policy **50.0%**. Every stronger or wider cell collapses to exact **15.0%** or **12.5%**, unsafe **0%**, benign-FR **0%**.

**Hypothesis verdict:** inconclusive for the fp32 Pareto frontier. In direct bf16, the frontier is a cliff: even K=16 fails once strength reaches 0.60. This is more evidence that the edit implementation/dtype interaction dominates the intended budget effect.

**Follow-up:** fix the missing-results write failure and rerun in fp32. The scientifically important output is not the best exact cell but the full exact/unsafe/benign-FR surface.

### Experiment 23 — Dual-Polarity Push-Pull — mathematically degenerate as implemented

**Input:** `pushpull_summary.json`, CUDA, wall **328.64s**.

| Arm | Exact | Policy | Unsafe | Benign FR | Mean reward |
|---|---:|---:|---:|---:|---:|
| suppress only | 15.0% | 25.0% | 0.0% | 0.0% | 0.440 |
| pushpull 0.3 | 15.0% | 25.0% | 0.0% | 0.0% | 0.430 |
| pushpull 0.5 | 40.0% | 52.5% | 60.0% | 0.0% | 0.710 |
| pushpull 1.0 | 42.5% | 42.5% | 100.0% | 100.0% | 0.735 |

The central implementation problem is sharper than dtype confounding: `u_ref = normalize(ref-call)` and `u_call = normalize(call-ref)` are exact negatives, and projection onto a unit vector is sign-invariant. Therefore `proj_c == proj_r`, and the update `cols - suppress·proj_r + amplify·proj_c` is just `cols + (amplify-suppress)·proj_r`. The arms are net strengths **0.4, 0.2, and -0.3**, not independent suppression plus amplification. The apparent nonmonotonicity is exactly a sign reversal: amplification 1.0 adds refusal direction back and produces 100% unsafe refusal plus 100% benign refusal.

**Hypothesis verdict:** refuted as implemented; the experiment did not test non-collinear push-pull.

**Follow-up:** Exp 38 is the correct replacement because it explicitly verifies direction cosine. Its results should supersede Exp 23 entirely.

### Experiment 25 — Gradient-Free Evolution — optimization ran, but fitness rewards unsafe collapse

**Input:** `evolution_summary.json`, CUDA, wall **523.73s**.

Best parameters: `k29=8`, `k28=8`, `s29=0.7608`, `s28=0.3466`, `l3=29`, `s3=0.0`, `amp=0.0`. Final dev exact **20.0%** with unsafe **37.5%**; final test exact **50.0%** with unsafe **42.9%**, benign-FR **0%**, policy **60.0%**.

Important caveats:

- Dev/test are only 20 rows each and have different class mixes: dev has 16 unsafe/4 benign, test has 14 unsafe/6 benign.
- The fitness function `exact + 0.5·unsafe - 0.5·benignFR` does not impose a hard safety constraint. A model that complies with everything can score well by reducing benign false refusal while sacrificing unsafe refusal.
- Test exact 50% is not a win when unsafe refusal is only 42.9% (6/14 unsafe tasks refused).

**Hypothesis verdict:** inconclusive for gradient-free search; negative evidence against the current fitness design. The search did learn to shrink the edit budget to K=8, consistent with the bf16 collapse cliff, but it did not find a safe operating point.

**Follow-up:** rerun in fp32 with a lexicographic or constrained objective: reject any candidate below 100% unsafe refusal before optimizing exact compliance.

### Experiment 26 — Entropy-Weighted Fork GRPO — collapsed initialization makes arm comparison uninformative

**Input:** `entropy_grpo_summary.json`, CUDA, wall **298.0s**.

The post-edit pre-RL checkpoint is already at exact **15.0%**, unsafe **0%**. Final exact is **7.5%** for uniform advantage and **10.0%** for entropy weighting — a one-task difference on 40 rows — with unsafe refusal still **0%** in both. Entropy-weighted loss is unstable, ranging from **-1.9109** at step 1 to **3.2107** at step 4 and **-0.6927** at step 8.

**Hypothesis verdict:** no verdict. Entropy weighting cannot be evaluated when the rollout policy never refuses unsafe prompts.

**Follow-up:** rerun only after the fp32 edit passes a safety gate. Log per-step mean reward and KL, not just loss.

### Experiment 28 — Cross-Layer Head+MLP Synergy — head-only survives, MLP-containing arms collapse

**Input:** `synergy_summary.json`, CUDA, wall **300.21s**.

| Arm | Exact | Policy | Unsafe | Benign FR | Mean reward |
|---|---:|---:|---:|---:|---:|
| CUDA bf16 baseline | 47.5% | 50.0% | 100.0% | 80.0% | 0.775 |
| MLP64 only | 15.0% | 25.0% | 0.0% | 0.0% | 0.440 |
| Head only | 45.0% | 50.0% | 100.0% | 60.0% | 0.800 |
| Half/half synergy | 12.5% | 27.5% | 0.0% | 0.0% | 0.430 |
| Full synergy | 15.0% | 25.0% | 0.0% | 0.0% | 0.435 |

**Hypothesis verdict:** inconclusive for synergy. The interesting secondary signal is that head-only editing preserves unsafe refusal and lowers benign false refusal from 80% to 60%, but exact compliance does not improve (45.0% vs 47.5% baseline). Every arm containing the bf16 MLP edit collapses, so the experiment mostly confirms the MLP dtype failure.

**Follow-up:** rerun in fp32 and include a head-only dose sweep. Head-only is not currently a win, but it is the only wave-1 mechanism that did not destroy safety.

### Experiment 29 — Momentum-Annealed Surgery+RL — no evidence of handoff; reward collapses when scaffold reaches zero

**Input:** `annealed_rl_summary.json`, CUDA, wall **301.92s**.

Fixed-edit final: exact **12.5%**, unsafe **0%**. Annealed final: exact **30.0%**, policy **32.5%**, unsafe **100%**, benign-FR **90%**. Mean reward in the annealed arm rises to **0.800** at step 6 (edit strength **0.225**) and then falls to **0.425** at step 8 when edit strength reaches **0.0**.

**Hypothesis verdict:** weak negative evidence, but still confounded. Safety restoration is mechanical because annealing rewrites the edited columns back toward their original weights; it is not evidence that RL learned safe behavior. The reward collapse at zero scaffold strength suggests the opposite of handoff.

**Follow-up:** evaluate at each annealing fraction, not only at the end, and compare against an arm where the edit is removed without RL. If both recover identically, the scaffold is irrelevant.

### Experiment 30 — Steering-Oracle Guided RL — no measurable exploration benefit under collapse

**Input:** `steering_rl_summary.json`, CUDA, wall **314.93s**.

Post-edit pre-RL baseline: exact **15.0%**, unsafe **0%**. Plain RL finishes at exact **15.0%**, policy **45.0%**; steered rollouts finish at exact **15.0%**, policy **25.0%**; unsafe remains **0%** in both. Plain reward declines from **0.65** at step 1 to **0.475** at step 8, while steered reward stays flat around **0.55**.

**Hypothesis verdict:** inconclusive. Steering did not improve exact compliance, but because the base policy was already stuck in always-call mode, the experiment cannot distinguish bad exploration from a bad initialization.

**Follow-up:** rerun in fp32 and log the fraction of unsafe rollouts that are refusals before RL begins. Without refusal exploration, oracle steering is not being tested.

### Cross-result update after wave 1

1. The largest confirmed mechanistic constraint is **true/false refusal entanglement**. DLA can find refusal-circuit neurons, but the shared circuit makes safety-preserving suppression hard.
2. The largest engineering constraint is **operator fidelity**. Direct unembedding-direction rank-1 projection produces an always-call collapse that random matched-budget edits do not produce; dtype was a plausible initial explanation but is now ruled out as the root cause by Exps 33/40.
3. The most important experimental-design lesson is that RL and search need **hard safety gates**. KL anchoring can preserve a catastrophic checkpoint, and unconstrained fitness can reward unsafe compliance.
4. Exp 23 should be superseded by Exp 38; Exp 27 must be rerun after its source/target fix; Exp 22 needs a successful summary write plus fp32 rerun.
5. The most promising wave-1 observation outside the dtype failure is head-only editing in Exp 28: unsafe refusal stayed 100% and benign false refusal fell 80%→60%, though exact compliance did not improve.

### Experiment 31 — Request-Type Circuit Mapper — invalid as a request-type atlas due classifier mismatch

**Input:** `request_circuit_atlas.json`, CUDA, wall **3.67s**.

The atlas is not a six-type request map. Its only keys are `comms_task` (**40 rows**) and `other_unsafe` (**120 rows**). The keyword classifier does not match the actual policy-router dataset: there are no weather/calculate/math/bomb/weapon/exploit/drug keywords in `policy_train.jsonl`, while the real safety categories are labels such as `SECRET_EXFIL`, `PROMPT_INJECTION`, and `DESTRUCTIVE_OP`. Thus every benign row fell into `comms_task` and every unsafe row fell into `other_unsafe`.

Quantitatively, the two pseudo-types have nearly identical depth centers of mass (**20.74** vs **20.61**) and identical top-three neurons: L29N1512, L29N427, L29N1123. The apparent exclusive neurons are artifacts of small-sample hash ranking rather than validated type-specific circuits.

**Hypothesis verdict:** refuted as implemented. It did measure a coarse benign-vs-unsafe activation difference, but not request-type-specific circuits.

**Follow-up:** classify by ground-truth `target.policy_code` and `target.action`, not free-text keywords. The correct atlas categories should include benign calls plus `SECRET_EXFIL`, `PROMPT_INJECTION`, `DESTRUCTIVE_OP`, and other actual policy codes. Until fixed, Exp 32 has no valid type-specific target map.

### Experiment 32 — Targeted Circuit RL — downstream of invalid atlas and collapsed operator

**Input:** `circuit_rl_summary.json`, CUDA, wall **149.4s**.

Post-edit baseline is already collapsed: exact **12.5%**, unsafe **0%**, benign-FR **0%**. After 8 RL steps, circuit-column RL remains exact **12.5% / unsafe 0%**; full-layer control is exact **10.0% / unsafe 0%**. The trace means little because both arms begin in the always-call regime.

**Hypothesis verdict:** no verdict. Two independent invalidities stack: the atlas pseudo-types are wrong, and the edit operator is the replication-broken unembed-direction operator.

**Follow-up:** rerun only after (1) ground-truth policy-code atlas and (2) Exp 41-validated original surgical operator. The comparison should also freeze exact column sets rather than whole layers if circuit specificity is the claim.

### Experiment 33 — Dtype Edit Sensitivity — dtype difference is real but tiny, not causal

**Input:** `dtype_summary.json`, CUDA, wall **125.54s**.

fp32 edit: exact **12.5%**, policy **27.5%**, unsafe **0%**, max weight delta **0.065996**. bf16 edit: exact **15.0%**, policy **25.0%**, unsafe **0%**, max weight delta **0.066406**. Both collapse. The 2.5-point exact gap is one task on 40 rows.

**Hypothesis verdict:** refuted in its strong form. bf16 rounding perturbs the edit, but it does not explain the wave-1 collapse.

**Implication:** fp32-master application may still be good engineering hygiene, but it is not the mechanism that made the original edits surgical. Exp 33 should be retired as a primary explanation and kept as a small robustness control.

### Experiment 35 — Interference-Canceling Edit — partial mitigation but no safe optimum

**Input:** `interference_summary.json`, CUDA, wall **171.37s**.

Baseline fp32 is healthy: **55.0% exact / 100% unsafe / 90% benign-FR**. Uniform control collapses to **12.5% exact / 0% unsafe**. Interference cancellation at lambda 1.0 remains collapsed (**12.5% exact / 0% unsafe**). At lambda 3.0 exact rises to **27.5%** and unsafe to **20.0%**, with policy **40.0%** and benign-FR **0%**. Mean coefficient falls from **0.7** to **0.623** at lambda 1 and **0.32** at lambda 3.

**Hypothesis verdict:** weak positive evidence for interference cancellation as a direction, negative evidence for the tested objective. Cancellation reduces destructive weight movement, but the recovered behavior is mostly unsafe compliance, not selective benign compliance.

**Follow-up:** combine the interference solve with the correct residual false-direction operator and impose a hard unsafe-refusal constraint. Report coefficient magnitude and safety together; reducing mean coefficient is only useful if it does not merely weaken the safety circuit.

### Experiment 36 — Safety Vaccination Edit — no protection under tested doses

**Input:** `vaccination_summary.json`, CUDA, wall **200.05s**.

Baseline **55.0% exact / 100% unsafe / 90% benign-FR**. Every arm collapses to **12.5% exact / 0% unsafe / 0% benign-FR**: edit strengths 0.7 and 1.2 without vaccine, and vaccine strengths 0.5 and 1.0 at edit strength 1.2.

**Hypothesis verdict:** negative for the tested implementation. Vaccination does not measurably protect the unsafe-refusal behavior against this edit.

**Interpretation:** because the edit direction itself is replication-broken, this may be protection against the wrong perturbation. However, the total absence of dose response still suggests the vaccine vector is either too weak, misdirected, or not aligned with the causal safety direction.

**Follow-up:** validate the vaccine direction by causal patching before combining it with an edit. Then rerun with the Exp 41 operator and report dose-response at finer strength intervals.

### Experiment 37 — Collision-Priced Neuron Allocation — selection changes substantially, behavior does not

**Input:** `collision_priced_summary.json`, CUDA, wall **195.87s**.

Baseline **55.0% exact / 100% unsafe / 90% benign-FR**. Pure DLA overlaps the baseline selection by **64/64** neurons and collapses to **12.5% exact / 0% unsafe**. Penalty 0.5 reduces overlap to **37**, penalty 1.0 to **30**, and penalty 2.0 to **25**, yet all priced arms still collapse to **12.5% exact / 0% unsafe**.

**Hypothesis verdict:** negative under the current operator. Collision pricing is doing something real at the selection level, but the selected alternatives are not behaviorally safer.

**Implication:** overlap with the original DLA set is not itself the causal collision metric. The algorithm needs a behavioral collision estimate (true-refusal margin cost per neuron) rather than a proxy penalty.

### Experiment 38 — Non-Collinear Push-Pull — still not non-collinear in the relevant sense

**Input:** `noncollinear_pushpull_summary.json`, CUDA, wall **189.64s**.

The script tries to fix Exp 23 by using a full-token `call_tool`-vs-`refuse` unembedding direction for amplification, but the recorded direction cosine is **-1.0** in every arm. Thus this run still does not establish a non-collinear pair. The arm metrics reflect sign/dose artifacts: suppress-only **12.5% exact / 0% unsafe**, pushpull 0.3 **12.5% / 0%**, and pushpull 0.7 returns exactly to baseline **55.0% exact / 100% unsafe / 90% benign-FR** because the opposing terms cancel.

**Hypothesis verdict:** no verdict. Like Exp 23, the implementation is degenerate.

**Follow-up:** construct independent directions explicitly: for example, use the unembed refusal direction for suppression and a measured benign-compliance residual direction for amplification, then enforce/report `abs(cosine) < 0.3` before running evals.

### Experiment 39 — Operator Sensitivity Showdown — pivotal negative control

**Input:** `operator_showdown_summary.json`, CUDA fp32, wall **280.91s**.

Baseline fp32: **55.0% exact / 100% unsafe / 90% benign-FR**. The unembedding-direction fixed-strength swarm operator collapses to **12.5% exact / 0% unsafe / 0% benign-FR**. A fixed-strength residual-direction edit does **not** collapse but also does not improve: **52.5% exact / 100% unsafe / 90% benign-FR**. Both gated arms finish exactly at baseline (**55.0 / 100 / 90**), indicating the double-check gate rejected all candidate strengths rather than accepting a beneficial edit. The direction cosine between unembedding direction and measured residual direction is **0.0039**.

**Hypothesis verdict:** confirmed that operator mismatch is a first-order replication variable; not yet confirmed that the original operator is reproducible on CUDA fp32.

**Interpretation:** this is the most important result since the wave began. The unembedding logit direction is a scoring/probe direction, not necessarily a safe weight-edit direction. Treating it as the edit direction destroys safety. However, the measured-resid implementation in Exp 39 is itself approximate (a variance direction, not the original pass/fail mean difference), and the gate rejects edits, so Exp 39 does not yet explain the original 80% result.

**Follow-up:** Exp 41 is now the decisive experiment: exact original direction construction, exact original neuron scoring, exact double-check gate, and same random seeds/data ordering. If it fails, the original 80% result must be treated as non-replicated until proven otherwise.

**Nearest existing positive control:** Exp 16 `raw_rank1` already shows that a measured-contrast residual subspace edit can improve behavior without collapsing safety: **65.0% exact / 90.0% unsafe / 0.0% benign-FR**, policy **80.0%**, mean reward **0.885**. That script uses pass-vs-canonical-false-refusal contrast matrices, ranks columns by projection energy, and accepts strength through the dev gate. This is not the same as Exp 17's exact operator, but it is strong evidence that the failure is not “all rank-1 residual edits collapse.” Exp 39's residual arm is not a faithful version of this operator: it approximates a direction from variance within benign rows rather than pass/fail contrast, so its inactive result should not be overinterpreted.

### Experiment 40 — fp32-Master Edit Pipeline — fp32 does not rescue wrong operator

**Input:** `fp32_pipeline_summary.json`, CUDA, wall **173.36s**.

All four arms collapse: fp32 edit + fp32 eval **12.5% exact / 0% unsafe**; fp32 edit then bf16 cast **15.0% / 0%**; stochastic bf16 cast **15.0% / 0%**; direct bf16 edit **15.0% / 0%**.

**Hypothesis verdict:** refuted. fp32-master application and cast do not preserve surgical precision because the collapse is caused primarily by the edit operator/direction, not rounding.

**Implication:** fp32-master editing is still a sensible implementation standard, but it is not a differentiating technique. Future technique comparisons can use one consistent precision pipeline and focus on direction, selection, gating, and constraints.

### Wave-2/3 cross-result synthesis

1. **The dtype hypothesis is dead as the primary explanation.** fp32 eval still collapses and fp32/bf16 deltas are nearly identical.
2. **The operator hypothesis is alive and central.** Unembedding-direction projection is destructive; measured residual directions do not collapse but are not yet beneficial.
3. **The original 80% result is not yet replicated.** Exp 39's gate accepted no edit; Exp 41 must establish reproducibility before interpreting downstream combinations.
4. **No wave-3 protection/allocation technique overcame the broken base edit.** Interference cancellation partially restores exactness but sacrifices safety; vaccination and collision pricing have no behavioral benefit; noncollinear push-pull remains degenerate.
5. **Atlas quality is now a blocking dependency.** Exp 31 must use ground-truth policy codes before circuit-specific RL or vaccination can be interpreted.
6. **The next valid leaderboard entries should be gated on Exp 41.** Until then, all wave-1/2/3 intervention comparisons are operator-failure measurements, not mechanism rankings.
