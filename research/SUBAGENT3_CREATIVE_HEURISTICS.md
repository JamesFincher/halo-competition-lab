# SUBAGENT 3 — Creative High-Risk Heuristics for Surgical Edit + Targeted RL

**Date:** 2026-11-26  
**Target:** `HuggingFaceTB/SmolLM2-135M-Instruct` (`d_model=576`, 30 layers, `intermediate_size=1536`, 9 query heads / 3 KV heads, tied embeddings)  
**Lab context:** zero-RL surgical editing is already at **80.0% exact compliance with DLA** and **82.5% with IG-5**; disjoint re-anchored GRPO-style RL ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) is at **77.5%** on the 40-task refusal-vs-tool-call eval. The goal is not generic refusal removal: it is **removing benign false-refusal while preserving or strengthening true unsafe refusal**.

## Verification note

Citation anchors below were checked through the arXiv MCP server. Core mechanistic anchors were additionally fetched with `download_paper` and inspected with `read_paper`: `2406.11717`, `2012.14913`, `2202.05262`, `2012.13255`, `2203.05482`, `2310.03693`, and `2207.05221`. IDs from existing lab notes are included only when independently resolvable through arXiv.

## Local evidence that shaped the ranking

1. **Layer-level localization is nearly useless; neuron-level allocation matters.** The behavior atlas found all behaviors concentrated in L27–L29, while unrelated behaviors mostly occupy different neurons.
2. **True safety and false over-refusal are entangled but not identical.** They share the same building (`L27–L29`) and 77.8% of top neurons, including `L27 #110`; hard nullspace projection removed too much signal at `d=576`.
3. **Zero-shock capacity expansion is not enough.** The stitched GQA head was bit-exact at initialization but did not learn useful routing in six RL steps.
4. **Therefore the highest-value direction is conditional or coupled computation**, not more one-direction ablation: use keys, interference-aware solves, function-preserving rotations, and optimizer-state carryover.

## Ranked overview

| Rank | Experiment heuristic | Expected value | Risk | CPU pilot cost | Primary anchors |
|---:|---|---:|---:|---:|---|
| 1 | Benign-keyed negative-image refusal inverter | 9.4/10 | High | Low | `2406.11717`, `2012.14913` |
| 2 | Empirical interference-canceling multi-neuron solve | 9.1/10 | Medium-high | Medium | `2202.05262`, `2305.17333` |
| 3 | Zero-shock GQA policy-lane rotation + salience-priced lane LR | 8.8/10 | High | Low-medium | `2310.01405`, `2308.10248` |
| 4 | Safety vaccination before compliance amplification | 8.6/10 | Medium | Low | `2310.03693`, `2406.11717` |
| 5 | Evolution strategies in an interpretable 32-D surgical subspace | 8.4/10 | Medium-high | Medium-high | `1703.03864`, `2012.13255` |
| 6 | Momentum-carryover half-edits | 8.1/10 | High | Low | `1412.6980`, `2203.05482` |
| 7 | Collision-priced cross-behavior neuron allocation | 7.9/10 | Medium | Low | `2212.04089`, `2309.08600` |
| 8 | Edit-strength scaffold annealing | 7.7/10 | High | Medium | `2203.05482`, `1506.01186` |
| 9 | Self-modeling failure probes as RL advantage routers | 7.4/10 | Medium | Medium | `2207.05221`, `2306.03341` |
| 10 | Oscillating-polarity adversarial edit cycling | 6.9/10 | Very high | Medium | `1506.01186`, `2203.05482` |

---

## 1. Benign-keyed negative-image refusal inverter

**Citation anchors.** [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) shows refusal is mediated by a single direction across chat models; [arXiv:2012.14913](https://arxiv.org/abs/2012.14913) frames feed-forward layers as key-value memories, where keys select contextual patterns and values write output distributions.

**Mechanism.** Do not erase the refusal direction globally. Instead, convert one carefully chosen MLP neuron into a **signed benign/unsafe discriminator** whose `down_proj` column writes the *negative image* of the refusal direction. Use a mostly positive `gate_proj` row as an always-on gate and a discriminative `up_proj` row as the sign-carrying key: positive activation on benign prompts writes negative refusal, while negative activation on unsafe prompts writes positive refusal.

**Novelty rationale.** Existing activation steering and refusal-direction erasure are mostly unconditional or externally gated. This embeds a conditional polarity flip directly into weights by exploiting the key-value structure of the MLP. It is a surgical circuit inverter, not an ablation.

**Implementation sketch for SmolLM2-135M.**
1. Collect last-decision-token MLP inputs at L27–L29 for 20 benign `call_tool` and 20 unsafe `refuse` train prompts; compute mean direction `mu_l` and a centered LDA/Fisher key direction `k_l` separating benign from unsafe contexts.
2. Pick a low-collision neuron `j` outside the known hubs (`L27 #110`, `L29 #1512`, `L29 #1022`); overwrite `gate_proj[j]` with a scaled `mu_l` whose orientation makes the gate input positive across both train classes, overwrite `up_proj[j]` with the signed key `k_l`, then set `down_proj[:, j] ← down_proj[:, j] - alpha * r_l`, where `r_l` is the measured true-refusal direction.
3. Grid `alpha ∈ {0.1, 0.25, 0.5, 0.75}` and layers `{27,28,29}` on a train/dev split; accept only if unsafe refusal remains 100%, benign false-refusal hits 0%, and JSON validity stays 100%.
4. Re-anchor the RL reference to the accepted checkpoint and run 3–6 disjoint GRPO steps on adjacent layers while freezing the inverter neuron and the strongest true-safety neurons.

**Why it might fail.** SmolLM2-style projections are bias-free, so the centered key may not produce clean positive/negative activation signs; negative-image writing may also entangle policy-code logits. If one row cannot separate the distributions, use two adjacent neurons as a crude product/AND circuit.

---

## 2. Empirical interference-canceling multi-neuron solve

**Citation anchors.** [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) introduced localized closed-form editing of transformer weights; [arXiv:2305.17333](https://arxiv.org/abs/2305.17333) shows that forward-only zeroth-order optimization can be practical for language-model fine-tuning.

**Mechanism.** Treat the top-DLA neurons as coupled actuators, not independent volume knobs. Empirically measure how a tiny basis edit at each neuron changes the decision margin on both benign and unsafe prompts, then solve one small ridge-regression system for the coefficients that increase benign compliance while canceling collateral unsafe-refusal loss.

**Novelty rationale.** Current DLA/IG edits choose neurons independently and assume their effects add linearly. This method measures population-level responses at the output margin, so the solve can cancel collateral effects across neurons and layers instead of merely summing independent saliency scores. It is a behavioral analogue of multi-microphone noise cancellation.

**Implementation sketch for SmolLM2-135M.**
1. Choose `K=24` candidate edits: top positive-DLA neurons in L27–L29, each moved a small distance along its local false-refusal write direction.
2. For each candidate `i`, apply only a small probe edit `epsilon_i` to a fresh model copy, record decision-token margin shifts on 12 benign + 12 unsafe dev prompts, revert, and fill sensitivity matrix `A ∈ R^{24×24}`.
3. Solve `(A_bᵀA_b + λA_uᵀA_u + μI)x = A_bᵀb`, where `b` is the desired benign margin shift and the unsafe term penalizes movement of true-refusal margins.
4. Apply the joint edit once at full scale, evaluate the untouched 40-task set, then run disjoint re-anchored RL on complementary layers; compare against independent top-neuron editing at the same parameter budget.
5. If the first-order solve helps but underfits, add 12 random composite probes and refit a low-rank quadratic interaction model before rejecting the approach.

**Why it might fail.** The local Jacobian can be inaccurate once coefficients become large. Use a trust region: cap `||x||∞`, then re-measure `A` after each accepted outer step.

---

## 3. Zero-shock GQA policy-lane rotation + salience-priced lane learning rates

**Citation anchors.** [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) formalizes representation engineering over population-level activations; [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) demonstrates activation-level steering without optimization.

**Mechanism.** Rotate each value-head coordinate system while applying the compensating rotation in the matching `o_proj` input columns, so the network is functionally unchanged at step 0. Choose the rotation so a small number of coordinates align with benign-vs-unsafe, tool-choice, and JSON-format variance. Because SmolLM2 has 3 KV groups feeding 9 query heads, one KV-side rotation creates coherent lanes for query-head triplets.

**Novelty rationale.** Unlike the already-tested stitched head, this adds no random capacity and needs no warmup. It is a function-preserving change of basis that makes behavior-specific directions addressable by the optimizer; per-lane learning rates then price updates by DLA/Fisher salience.

**Implementation sketch for SmolLM2-135M.**
1. For each 64-dimensional value head in the 3 KV groups of layers 27–29, collect activations at the decision token for ALLOW, true-refusal, false-refusal, and malformed-format examples.
2. Compute generalized covariance directions, then QR-orthonormalize them into `R_h`; for each value-head row-block use `W_v,h ← R_hᵀ W_v,h` and compensating output columns `W_o[:, cols_h] ← W_o[:, cols_h] R_h`, so step-0 logits remain unchanged within numerical tolerance.
3. Train only the top behavior-salient rotated lanes, using per-lane learning-rate multipliers `clip(DLA/(Fisher+eps), 0.1, 3.0)`; freeze low-salience and collision-heavy lanes.
4. Require `KL(base || rotated) < 1e-6` before RL, then compare against random orthogonal rotations and unrotated masks at identical trainable-parameter count.

**Why it might fail.** Attention may not be the main policy actuator in this model; the strong empirical signal is late MLP. Treat this as an architecture exploit, not a replacement for MLP surgery.

---

## 4. Safety vaccination before compliance amplification

**Citation anchors.** [arXiv:2310.03693](https://arxiv.org/abs/2310.03693) shows benign fine-tuning can degrade aligned safety behavior; [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) shows refusal behavior has a compact internal direction that can be manipulated.

**Mechanism.** Deliberately overshoot true-safety refusal before attempting aggressive false-refusal suppression. Amplify the safest true-refusal write neurons by a small margin, then remove false-refusal pressure in disjoint neurons while freezing or Fisher-penalizing the vaccinated safety writes during RL.

**Novelty rationale.** Most defenses merely constrain updates so safety does not degrade. Vaccination intentionally creates a safety margin first, on the theory that later RL and compliance amplification will consume part of that margin.

**Implementation sketch for SmolLM2-135M.**
1. Use the atlas and DLA scores to split neurons into true-safety-leaning versus false-refusal-leaning by `unsafe_DLA / benign_DLA`; exclude shared hubs from both sets.
2. Scale selected true-safety `down_proj` columns by `+5%` to `+15%`, verifying that unsafe refusal remains 100% and benign false-refusal does not increase.
3. Apply the compliance edit only in false-refusal-leaning neurons, then mask or strongly Fisher-penalize the vaccinated columns during disjoint GRPO.
4. Add paraphrased unsafe canaries as a hard gate: abort the arm if unsafe refusal falls below 95%, even if exact compliance rises.

**Why it might fail.** At 135M, safety and false-refusal circuits overlap heavily; amplifying safety can increase benign refusal. Use neuron-level ratios, not layer-level direction amplification.

---

## 5. Evolution strategies in an interpretable 32-D surgical subspace

**Citation anchors.** [arXiv:1703.03864](https://arxiv.org/abs/1703.03864) established evolution strategies as a scalable black-box alternative to RL; [arXiv:2012.13255](https://arxiv.org/abs/2012.13255) shows language-model fine-tuning has low intrinsic dimensionality; [arXiv:2305.17333](https://arxiv.org/abs/2305.17333) reinforces the forward-only optimization motivation.

**Mechanism.** Build a 24–48 dimensional coordinate system whose axes are interpretable surgical actions: coefficients on top-DLA neuron directions, per-layer strength scalars, and perhaps one negative-image inverter coefficient. Optimize those coefficients with NES/CMA-ES directly against verifier reward, then use targeted RL only to harden the winning point.

**Novelty rationale.** This searches over semantically meaningful surgical knobs rather than over raw weights or opaque LoRA coefficients. It can discover non-monotone edit combinations that gradient RL cannot reach because reward is sparse and exact compliance is discontinuous.

**Implementation sketch for SmolLM2-135M.**
1. Define `z ∈ R^32`: 24 neuron-direction coefficients in L27–L29, 6 per-layer scale coefficients, and 2 global polarity/strength coefficients.
2. Evaluate NES antithetic pairs on 24 train/dev prompts with fitness `mean_reward - 2*unsafe_misses - 0.5*json_failures`; hold out 16 validation prompts for model selection.
3. Apply the winning `z` once, evaluate the untouched 40-task set, then run 3–6 disjoint re-anchored GRPO steps in complementary layers.
4. Run mandatory controls: random 32-D subspace, coordinate-shuffled DLA subspace, and same-forward-budget zeroth-order optimization on raw weights.

**Why it might fail.** Small evals make black-box selection noisy and easy to overfit. Use prompt paraphrases in fitness and reserve the official 40-task set for one final measurement per arm.

---

## 6. Momentum-carryover half-edits

**Citation anchors.** [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) introduced Adam’s first/second-moment updates; [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) shows averaging multiple weight solutions can improve robustness without inference cost.

**Mechanism.** Apply only half of a validated manual edit in weight space, then encode the remaining half as optimizer momentum in directly trainable sparse weights. The first few RL steps complete the edit smoothly instead of treating it as a discontinuity to undo.

**Novelty rationale.** Surgical editing currently changes weights but leaves optimizer state at zero. This heuristic edits the *training dynamics* themselves: part of the intervention lives in velocity, not position. It tests whether RL rollback is caused by missing momentum compatibility rather than by the edit being wrong.

**Implementation sketch for SmolLM2-135M.**
1. Compute the validated DLA edit delta `ΔW`; apply `0.5ΔW` directly to selected `down_proj` columns.
2. Initialize AdamW `exp_avg` on those elements proportional to `-(1-α)ΔW` with small positive `exp_avg_sq`, so initial descent continues along the surgical direction rather than reversing it.
3. Run hybrid re-anchored targeted GRPO in direct sparse full-fine-tuning mode (not frozen-base LoRA, where base-weight momentum would be inert); leave the edited columns trainable alongside complementary layers so the momentum hypothesis is actually tested, and compare hard-edit/no-momentum, momentum-only ghost edit, and random-sign momentum controls.
4. Track rollback as `cos(ΔW, W_t-W_edit)` after every step, plus unsafe refusal, benign false-refusal, and KL.

**Why it might fail.** Momentum is transient and may either vanish before helping or overshoot into a worse basin. If rollback persists, the correct conclusion is that the edit is inconsistent with the reward geometry, not merely under-powered.

---

## 7. Collision-priced cross-behavior neuron allocation

**Citation anchors.** [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) shows task-vector arithmetic can combine or negate behaviors; [arXiv:2309.08600](https://arxiv.org/abs/2309.08600) shows sparse decomposition can expose more interpretable feature structure in language models.

**Mechanism.** Treat neurons as contested behavioral resources. Allocate each neuron to the behavior where its marginal benefit is highest after subtracting a collision cost for writing into other behaviors. Then enforce ownership during RL so later updates cannot overwrite earlier behavior circuits.

**Novelty rationale.** Hard orthogonalization already failed at 135M because true/false refusal overlap too strongly. Collision pricing is softer and more realistic: it does not require perfect orthogonal subspaces, only explicit accounting of cross-behavior interference.

**Implementation sketch for SmolLM2-135M.**
1. Build benefit matrix `B[b,j]` from DLA salience for ALLOW, true refusal, strict JSON, and correct tool selection; build collision cost `C[j,b']` from atlas neuron overlap and polysemantic-hub membership.
2. Greedily maximize `Σ x_bj B_bj - λ Σ x_bj C[j,b']` subject to one owner per neuron and the same top-64-per-layer budget as existing probes.
3. Apply behavior-specific edits only within allocated neurons; during sequential RL, mask or penalize writes into neurons owned by other behaviors.
4. Compare against random allocation, unrestricted editing, and the failed hard Gram-Schmidt/nullspace baseline at matched parameter count.

**Why it might fail.** Polysemantic neurons may be irreplaceable at 135M. The correct result may sometimes be “do not edit this behavior here,” and the optimizer should be allowed to make that choice.

---

## 8. Edit-strength scaffold annealing

**Citation anchors.** [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) motivates endpoint averaging across related solutions; [arXiv:1506.01186](https://arxiv.org/abs/1506.01186) shows cyclic intensity schedules can improve training behavior.

**Mechanism.** Use a strong surgical edit as temporary scaffolding, then fade it while RL learns to carry the behavior intrinsically. Start above the needed strength, reduce the scaffold in stages, and soup the best adjacent endpoints if the final unaided model is close but unstable.

**Novelty rationale.** Current edit-then-RL treats the manual edit as permanent. This treats it as training wheels: a time-dependent potential that shapes the basin early but disappears before deployment.

**Implementation sketch for SmolLM2-135M.**
1. Keep the pre-edit checkpoint immutable; at each outer stage materialize `W0 - λ_t U Uᵀ W0` with `λ_t ∈ {1.5, 1.0, 0.5, 0.0}` and the same targeted adapter or trainable subset carried across stages.
2. Run a small fixed number of GRPO steps per stage; after each stage, evaluate a probe copy with `λ=0` to measure how much behavior the learned update actually owns.
3. If performance collapses when `λ` drops, repeat the prior stage rather than raising RL learning rate; soup the best two adjacent endpoint adapters.
4. Compare fixed-strength edit, RL-only, and immediate-removal controls at identical gradient-step count.

**Why it might fail.** A strong scaffold can make all rollouts correct, creating zero reward variance and dead GRPO groups. Use partial prompt mixing: some batches unaided, some scaffolded.

---

## 9. Self-modeling failure probes as RL advantage routers

**Citation anchors.** [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) shows language models often encode whether they know an answer; [arXiv:2306.03341](https://arxiv.org/abs/2306.03341) uses internal activations for inference-time intervention; [arXiv:2303.08896](https://arxiv.org/abs/2303.08896) is a useful black-box self-consistency reference point.

**Mechanism.** Train tiny internal probes to predict the model’s own failure mode before the external verifier sees the answer. Use those probes to route RL advantage, edit strength, and curriculum difficulty rather than merely to report interpretability metrics.

**Novelty rationale.** The lab currently seeks failures behaviorally, then sniffs activations. This inverts the loop: use activations to predict behavioral failure in advance and make training allocation adaptive.

**Implementation sketch for SmolLM2-135M.**
1. Record hidden states at the decision token in layers 18, 24, and 29 across train rollouts; label each example with the deterministic verifier outcome: correct, false-refusal, missed-unsafe, wrong-tool, or malformed JSON.
2. Train one logistic probe per layer/class with leave-prompt-family-out validation; choose the layer with the best calibrated AUROC, not merely the best raw accuracy.
3. During RL, upweight advantage for prompts predicted to be high-risk and downweight saturated correct prompts; optionally use probe risk to choose among weak/strong conditional edit coefficients.
4. Report calibration and final behavior separately; never allow a probe to override a hard unsafe-refusal rule at inference.

**Why it might fail.** 160 train rows are too few for stable failure prediction. Generate paraphrase/counterfactual expansions before concluding the probe concept is weak.

---

## 10. Oscillating-polarity adversarial edit cycling

**Citation anchors.** [arXiv:1506.01186](https://arxiv.org/abs/1506.01186) introduced cyclical learning rates; [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) motivates averaging diverse but compatible endpoints.

**Mechanism.** Alternate the sign of a small surgical perturbation during short RL cycles, with decaying amplitude, then average the positive and negative endpoints. The model cannot simply lean on one constant linear write direction; it must remain correct while the perturbation reverses.

**Novelty rationale.** This is deliberately weird: polarity reversal acts like a mechanical stress cycle or adversarial forcing term. If the resulting soup retains gains, the behavior is less likely to depend on a fragile single-direction artifact.

**Implementation sketch for SmolLM2-135M.**
1. Parameterize the surgical edit as `s_t δE` on L28–L29, where `s_t ∈ {+1,-1}` and `δ` decays over cycles such as `0.5 → 0.2 → 0.1`.
2. Flip sign every 2 GRPO steps for 3 cycles while keeping the disjoint re-anchored reference fixed at the last accepted checkpoint.
3. Weight-soup the final positive and negative endpoints; evaluate the soup, both endpoints, and a no-flip constant-edit control.
4. Reject immediately if unsafe refusal falls below 95%, JSON validity falls below 97.5%, or KL grows faster than the constant-edit control.

**Why it might fail.** Sign reversal may simply destroy the basin or teach the model to ignore the edited subspace. This is ranked last because it is scientifically interesting but most likely to be unstable.

---

## Recommended execution order

1. **Pilot first:** idea 1 (negative-image inverter), idea 2 (interference solve), and idea 4 (vaccination). They directly attack the lab’s dominant false-refusal failure mode and cost little.
2. **Then architecture exploitation:** idea 3 if MLP results plateau; it is the most plausible way to use the 9Q/3KV structure rather than treating SmolLM2 as a generic transformer.
3. **Then search/dynamics:** ideas 5–8 once a stable edit basis exists, because they need a meaningful coordinate system to search or anneal.
4. **Diagnostics and moonshots:** idea 9 is a force multiplier for all arms; idea 10 should run only after the main leaderboard is saturated.

## Common controls and gates

- **Mandatory controls where applicable:** random direction, random neuron set, random orthogonal rotation, edit-only, RL-only, and random-sign momentum/sign controls at matched parameter and forward budgets.
- **Selection discipline:** do not tune on the official 40-task eval repeatedly. Tune on train/dev plus generated paraphrases; use the 40-task set for one final measurement per arm.
- **Primary metrics:** exact compliance, policy accuracy, unsafe refusal rate, benign false-refusal rate, benign tool accuracy, JSON validity, KL, and rollback cosine; report raw counts out of 40 because one task is 2.5 pp.
- **Hard aborts:** unsafe refusal `<95%`, JSON validity `<97.5%`, or exact-compliance gain accompanied by policy-code degradation `>5 pp`.
- **Scientific readout:** report whether the method changes behavior by conditional keying, coupled-write cancellation, optimizer dynamics, or ordinary direction erasure. The mechanism is the result, not just the leaderboard score.
