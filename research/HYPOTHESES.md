# Experiment Hypotheses — Living Pipeline

Last updated: 2026-11-26 by the hypothesizer agent.

## Operating context

Validated local assets:

- **Best validated surgical operator is now ultra-sparse:** exp129 strength 0.9 with 16 columns/layer reaches **80% exact / 93.3% unsafe refusal / 0% benign-FR**; exp130 columns16 independently reproduces **80/93.3/0**, and even columns8 reaches 77.5/93.3/0. This supersedes the 32-column operating point and makes the remaining 2/30 unsafe-refusal loss the central repair target. Benchmarks are unchanged on the coarse legacy scorer, so capability claims remain open.
- **Best validated transfer recipe is post-edit complementary MLP-only LoRA:** exp57 reaches **75/90/0** by freezing the canonical edit, anchoring to the post-edit model, and training only complementary MLP LoRA at layers 24–27; exp100 independently favors targeted complementary MLP-only LoRA over all-proj or partial full fine-tuning. This recipe has not yet been combined with the stronger ultra-sparse edit — that is the largest immediate composition gap.
- **Vector evidence is context-dependent:** exp80 vector-only complementary-layer RL remains strong (**80/93.3/10**), and exp86 oscillating vector RL reaches 72.5–75/93.3/0, but exp100 shows vector-only transfer is weak after the surgical edit (**47.5/66.7/0**) and exp50 post-edit disjoint vectors do not beat the edit anchor. Vector gains are real in some regimes, but not a universal substitute for input-dependent weight adaptation.
- **Attention-insert evidence weakened further:** exp65 insertion after L27 reaches **65/93.3/0**, but exp61 exact-zero keyed warm GQA collapses (**10/90/20**), exp59 warm GQA is harmful, exp66 edit+warm-slider composition collapses, and exp93 seed fragility remains unresolved. Freeze-then-merge is promising but attention insertion is behind sparse surgery + complementary MLP LoRA until seed replication and the creative blending controls land.
- **Capability-objective evidence is mixed:** exp132 `hybrid_capability_primary` reaches **72.5/93.3/0** and coarse MMLU/ARC move up slightly, but normalized continuation MMLU/HellaSwag decline; `capability_only` reduces validation loss without a clean benchmark win. exp139–141 are queued to disentangle cache artifacts, answer focus, and benchmark format before raw-capability claims are made.
- **Deployment-aware training caution:** exp101 export serialization is internally consistent, but merge-aware served-weight action-token training underperforms badly (**42.5/100/0** versus post-edit anchor 65/90/0). Export validity is necessary but not sufficient; H10 is demoted and H21 must rely on teacher/KL distillation rather than rounding alone.
- **Queue/result state at 14:10 UTC:** H22 is now queued as exp144. exp134–138 results are negative for architectural blending relative to the sparse static edit; exp136 sparse H19 shows oracle cancellation can restore unsafe refusal to 100% but costs exactness, while the learned gate underperforms edit-only. exp139–148 queue capability-transfer/inversion diagnostics.
- **Known dead ends:** unembedding-direction weight edits collapse into always-call behavior at all tested strengths; fp32 does not rescue them; degenerate collinear push-pull is a sign/strength artifact; hard nullspaces and binary masks are too destructive on a 135M model.

Research scan for this batch: arXiv cs.CL/cs.LG papers submitted 2026-11-19 through 2026-11-25 in weight-space programming, activation steering, attention-head intervention, parameter-efficient RL, subspace repair, MoE/architecture expansion, plus frontier/lab blogs including MIT CSAIL weight-space programming and Anthropic feature-steering/introspection writeups. The 11:30 UTC incremental scan added deterministic Forward Weight-Space Programming/Protected FisherGate ([arXiv:2611.34612](https://arxiv.org/abs/2611.34612)) and gradient-free neuron programming ([arXiv:2611.33973](https://arxiv.org/abs/2611.33973)). The local `arxiv-mcp-server` is currently misconfigured (`uvx` entry point missing), so the scan used the official arXiv API and full HTML papers directly.

## Priority queue

| Priority | Hypothesis | Main type | Status |
|---:|---|---|---|
| 9.7 | H22 — Ultra-sparse edit anchored complementary MLP-only LoRA | surgical edit + targeted PEFT composition | queued as exp144; highest-priority result to read next |
| 9.4 | H15 — Multi-perspective abstention-direction factorization | refusal/compliance subspace geometry | queued as exp87; now the key gate because exp136 shows cancellation works but learned context separation is insufficient |
| 8.9 | H19 — Context-gated rank-one surgical dose cancellation | input-conditioned surgical repair | exp136 mixed: oracle validates the safety/exactness tradeoff, learned gate underperforms edit-only |
| 9.3 | H2 — Composite operator-discrepancy collateral repair | closed-form trajectory repair | queued as exp128 after upstream output-space projection fix; increasingly important because H19 oracle repair costs exactness |
| 9.1 | H9 — Curvature-constrained acquisition deltas | preservation-aware surgical learning | queued as exp121; more important because auxiliary training often damages the anchor |
| 8.8 | H21 — Exportable consolidation of complementary steering-vector RL | vector-to-weight distillation + deployment-exact PEFT | revised downward after exp101: teacher/KL distillation matters, rounding alone does not |
| 8.7 | H6 — Input-adaptive Kronecker lanes for complementary attention/MLP edits | parameter-efficient surgical adaptation | queued as exp123; strengthened by complementary MLP-only LoRA validation |
| 8.6 | H18 — Protected FisherGate decision-readout edit | closed-form weight-space programming | unqueued; staged behind exp102 |
| 8.5 | H7 — Drift-triggered orthogonal rotation lanes | rotation-only PEFT + attention heads | queued as exp120; must beat exp57/100's simpler MLP-only comparator |
| 8.4 | H20 — Kolpona-style accept/reject failure-habit neuron ladder | gradient-free neuron programming | unqueued; supported indirectly by ultra-sparse dose evidence |
| 8.3 | H12 — Closed-form decision-readout correction | gradient-free readout surgery | queued as exp102 |
| 8.2 | H4 — EMS-guided anchor-preserving subspace edits | causal neuron localization + surgical edit | queued as exp124 |
| 8.1 | H8 — Displacement-probe edit auditor and reference chooser | cheap pre-GPU screening diagnostic | queued as exp125 |
| 7.2 | H5 — Output-matched steering-vector compression | activation steering + RL initialization | mixed: vector RL can work pre-edit, but post-edit vector transfer is weak |
| 7.1 | H11 — Trajectory-aware budgeted decision-token steering | activation steering | queued as exp127; mixed evidence, still distinct as inference-time control |
| 7.1 | H14 — Spectral-sink calibrated post-SDPA microgates | attention-head modification | queued as exp126 |
| 7.0 | H17 — Prepare-for-truncation elastic low-rank deployment | compression/raw capability | exploratory |
| 6.2 | H1 — Exact-zero keyed modular implant bank | surgical insert + attention head + RL | demoted further: exp134–138 blending/cascade/gate variants do not beat the sparse static edit |
| 6.2 | H3 — Functional-connectivity-guided head insertion | attention-head modification | demoted further: insertion depth exists, but blending/cascade/gate results do not beat simpler sparse surgery |
| 6.1 | H13 — Zero-init attention-latent feedback lane | raw-capability architectural insert | demoted: attention composition and seed fragility remain unresolved |
| 5.8 | H10 — Merge-aware served-weight action-token fine-tuning | deployment-exact PEFT | exp101 negative as implemented; keep only as an export-control component, not a default training method |
| 5.4 | H16 — Norm-calibrated single-coordinate decision steering | sparse activation surgery | exp67 negative at tested doses; no positive operating point |


## Recommended execution order

Updated after the 13:10 UTC harvest (exp50–67, exp99–101, exp129/130/132). The central result is that two independently validated positives have not yet been combined: ultra-sparse surgery (**80/93.3/0**) and post-edit complementary MLP-only LoRA (**75/90/0** on the weaker canonical edit).

1. **Read exp144/H22 first.** It is now queued and is the highest-value composition: exp129/130's 16-columns/layer strength-0.9 sparse edit plus exp57/100's freeze-edit → post-edit KL anchor → complementary MLP-only LoRA recipe.
2. **Repair context separation, not cancellation algebra.** exp136 shows oracle cancellation restores unsafe refusal but costs exactness, while the learned gate underperforms edit-only. exp87/H15 and better false-vs-true gate features are now the bottleneck before more H19 dose cancellation.
3. **Run/reinterpret H2 at the correct dose.** exp128 has the upstream output-space projection fix, but its canonical top64/0.9 substrate is no longer the best operator. Use it first as a methodological test, then repeat at 16 columns/layer if the mechanism is positive.
4. **Use exp57/100 as the PEFT comparator.** exp120/H7, exp121/H9, and exp123/H6 must beat or explain the simpler complementary MLP-only LoRA recipe; otherwise they are complexity without benefit.
5. **Decision-readout branch remains gated:** exp102/H12 before H18. Do not confuse H16's negative single-coordinate result with readout failure; exp67 tested a raw coordinate actuator, not a trained readout.
6. **Raw capability is unresolved.** Read exp139/140/141 before making any capability-objective claim. Exp132's hybrid arm is behaviorally strong, but normalized continuation metrics contradict the coarse benchmark improvement.
7. **Attention blending is exploratory, not central.** exp134/135/137/138 can proceed as creative probes, but exp61/66/93 mean keyed/warm/composite attention inserts should not displace H22/H19/H2.

Unqueued but high-value after the gates:

- **H22** is the immediate top implementation gap.
- **H18** follows exp102.
- **H20** follows exp103/124 and sparse replication.
- **H21** remains important only as teacher/KL vector consolidation; exp101 shows merge-aware rounding alone is not a solution.

Hold or stage behind current queue:

- **H1/H3/H13:** attention inserts are scientifically interesting but empirically fragile. Require seed replication and a positive creative-blend result before more large banks.
- **H10:** demoted after exp101. Use exact export measurement as a required evaluation harness, not as the mechanism.
- **H16:** falsified at tested doses; only revisit after a nonlinear/readout-based alternative exists.
- **H5/H11:** vector methods are context-dependent: strong pre-edit in exp80/86, weak post-edit in exp50/100. Do not rerun vector RL without a new mechanistic delta.


---

## H1 — Exact-zero keyed modular implant bank

**Priority: 6.2/10** *(demoted after exp134–138: routed width mixtures, capability/behavior blending, cross-layer cascades, and subtractive gates do not beat the ultra-sparse static edit; attention banks are exploratory rather than central.)*

### Novel idea

Generalize Exp 42's single scalar-slider head into a small bank of **new attention heads and MLP neurons inserted inside existing layers**, each behind an input-conditioned gate that is exactly zero off its trigger. Gates should be simple products of ReLU/step affine functions of the residual stream, initialized from measured benign/unsafe/tool-choice activation boundaries rather than left as unconstrained always-on scalars. The insert bank is additive capacity: original weights remain untouched, and off-trigger behavior is bit-exact rather than merely low-drift.

### Why it should work mechanistically

Exp 42 shows zero-init additive heads do not destroy the base model, but its learned scalar becomes globally active and policy accuracy degrades. The fresh weight-writing result shows that new heads/neurons can remain exactly inactive outside narrow residual-stream trigger regions while adding substantial capability. Input-conditioned exact-zero gates directly attack the lab's central problem: false-refusal, true-refusal, tool-choice, and general-capability circuits are entangled, so an always-on additive head helps one behavior while taxing the others.

### Concrete experiment design

1. Start from two checkpoints: unedited Halo SFT and the validated ultra-sparse residual-subspace edit (`raw_rank1`, layers 29/28, 16 columns/layer, strength 0.9).
2. Insert a bank at layers 27–29:
   - 1–4 extra attention heads after layers 27/28, architecture-compatible with SmolLM2's 64-dimensional heads;
   - 8–32 extra MLP neurons at layers 28/29 with their own gate/up/down projections;
   - every group gated by \(g(x)=\prod_m \mathrm{ReLU}(a_m^\top x+b_m)\) or a small-step-function approximation, with exact zero outside the designed trigger region.
3. Initialize gates from decision-boundary contrasts for benign correct allow, benign false refusal, true unsafe refusal, wrong-tool, and malformed JSON. Initialize head/neuron weights at zero or tiny random values; keep all core weights frozen.
4. Train only insert weights and gate thresholds with the exp96 causal schedule: one insert or layer at a time, with previously trained inserts frozen but active. Use 8 steps/head rather than joint all-head training, and compare joint, freeze-then-merge, and no-insert controls at equal optimizer-step budget.
5. Make seed robustness part of the intervention, not a post-hoc caveat: run seeds 7/42/123 for the two best schedules. Accept an architecture only if at least two seeds beat its matched random-keyed and edit-only controls by at least one exact-compliance task without worse unsafe refusal or benchmark temperature.
6. Compare:
   - scalar Exp 42/v1 head reproduction;
   - always-on head/neuron bank;
   - exact-zero keyed bank;
   - random-keyed bank;
   - joint versus freeze-then-merge training;
   - edit-only and complementary vector-RL controls.
7. Measure standard policy metrics, gate firing rates by behavior, exact identity/KL on off-task text, `scripts/real_benchmark.py` and `scripts/continuation_benchmark.py` temperature, parameter count, wall time, final alpha sign/magnitude by seed, and whether the implant can be exported as a normal checkpoint rather than a session-local hook.

### Expected outcome and falsification criterion

Expected: exact compliance above the edit-only operating point while preserving unsafe refusal at **≥95%**, benign false refusal materially below the best seed-robust scalar insert, and neutral-or-better benchmark temperature in at least two of three seeds. The hypothesis is falsified if keyed gates fire on more than ~1% of off-task/control tokens, if the keyed bank does not beat scalar/random-keyed controls by at least one exact-compliance task (2.5 pp) in two seeds, if freeze-then-merge loses seed robustness, or if any gain comes with cold/FREEZING general-benchmark collateral.

### Citation

- [arXiv:2611.28483 — *Don't Train It, Write It: An AI Agent Writes New Capabilities into Language Model Weights*](https://arxiv.org/abs/2611.28483): agent-written gated attention heads/MLP neurons added inside existing layers, exactly zero off-trigger, BBH normalized score 19.95→41.05 on Qwen2.5-1.5B with zero KL on 1,023,000 WikiText tokens.
- Local Exp 42 zero-init slider head and Exp 16 residual-subspace operator.

---

## H2 — Composite operator-discrepancy collateral repair adapter

**Priority: 9.2/10** *(composite branch queued as exp128 after the upstream output-space projection fix; module-local rank-one discrepancy remains algebraically degenerate, so interpret it through downstream trajectory discrepancy and repeat at sparse dose if positive.)*

### Novel idea

Treat the whole post-edit segment—from the first edited layer through the final residual/readout—as the perturbed operator. Measure how the same segment input produces different downstream residual/logit trajectories before versus after surgery, SVD that **composite operator discrepancy**, remove the desired false-refusal/compliance direction, and fit a closed-form factorized residual adapter that restores only collateral computation.

### Why it should work mechanistically

A rank-one `down_proj` edit has output discrepancy \(\Delta W x=-s u(p^\top x)\), which is always parallel to \(u\); there is no nontrivial target-orthogonal collateral subspace at that module. The observed collateral damage is created downstream as this rank-one impulse passes through residual mixing, attention, nonlinear MLPs, and the readout. At the composite segment level the discrepancy is generally multi-directional. That is the right level for ReQOR-style repair: nonlinear propagation creates separable collateral directions even when the original weight delta is rank-one.

### Concrete experiment design

1. Run/interpret exp122 first as a structural gate. If it reports near-zero non-target collateral rank at module-local `down_proj` outputs, treat that as confirmation of the algebra above, not as evidence that composite repair is impossible.
2. Use the strongest clean surgical operating point available: Exp 1 arm1-style top-2 contrastive MLP `down_proj` edit (layers 29/28, top 32 columns, strength 0.5) or the exp95/98/99 dose-response winner. Keep Exp 16 `raw_rank1` as the replication/control arm.
3. Cache segment inputs immediately before the earliest edited layer on 32 policy train/dev prompts, 32 paraphrase/control prompts, and a small general-capability sample. Because upstream weights are unchanged, these inputs are matched across original and edited models.
4. Compute composite output discrepancies at post-L29 residual, final pre-unembed residual, and decision-token logits. Use teacher-forced correct and failure trajectories so token positions are aligned.
5. SVD the discrepancy matrices and verify effective rank \(>1\). Project out the exp87/H15 false-refusal/compliance target direction while explicitly protecting true-refusal states. Fit rank \(r\in\{4,8,16,32\}\) factorized residual corrections by ridge/least squares against the collateral discrepancy.
6. Arms: unedited baseline; edit-only; edit+full composite repair; edit+target-orthogonal collateral repair; edit+random-subspace repair; Exp 35-style interference-canceling baseline; H19 oracle-gated cancellation as an upper-bound comparator once available.
7. Evaluate standard 40-task metrics, off-task KL to the unedited model, MMLU/ARC/HellaSwag temperature, adapter memory, firing norm, exportability, and dose-response across repair rank.

### Expected outcome and falsification criterion

Expected: composite discrepancy is multi-directional even though module-local discrepancy is rank-one; target-orthogonal repair retains within 2.5 pp of edit-only exact compliance while lifting unsafe refusal toward 95–100% and reducing off-task KL/benchmark collateral. Falsified if composite discrepancy remains effectively rank-one, target-orthogonal repair destroys the target gain, repair does not beat random rank-matched controls, or it reproduces Exp 35's unsafe-compliance recovery failure mode.

### Citation

- [arXiv:2611.32755 — *ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models*](https://arxiv.org/abs/2611.32755): same-state operator discrepancies, closed-form factorized residual restoration, rank-efficient repair without gradients.
- [arXiv:2611.34612 — *Do We Really Need an LLM for Weight-Space Programming?*](https://arxiv.org/abs/2611.34612): deterministic forward-only probing, operator capability boundaries, and paired verification/rollback for weight-space edits.
- Local Exp 35 interference cancellation, Exp 39 operator sensitivity, and exp122 rank-audit pre-flight.


---

## H3 — Functional-connectivity-guided head insertion

**Priority: 6.2/10** *(demoted: exp65/91 support L27 as the best insertion site, but exp134–138 show architectural routing/blending does not yet beat simpler sparse surgery.)*

### Novel idea

Build a functional graph of attention-head outputs across correct versus failure behaviors and use connectivity reorganization—not activation magnitude alone—to choose where slider heads, head offsets, or sparse head RL should be installed. The most promising intervention is a zero-init head-specific offset \(z'_h=g_hz_h+u_h(v_h^\top z_h)\) at graph-selected heads, optionally paired with a new gated head at the same layer.

### Why it should work mechanistically

The current atlas localizes behaviors mostly to late layers, but layer location is coarse and Exp 28 showed head-only edits were safe yet weak. Behavioral failure may be a coordination failure among heads rather than excess activation in one head. Functional connectivity captures that interaction structure; graph-selected head interventions recovered hidden capability in larger models while random-head and activation-magnitude controls were worse.

### Concrete experiment design

1. Record pre-\(W_O\) outputs of every attention head at the final prompt token for train/dev prompts labeled correct allow, benign false refusal, true unsafe refusal, wrong tool, wrong policy code, and malformed JSON.
2. For correct and each failure condition, compute pairwise head similarity with SVCCA/canonical-correlation mutual-information edge weights. Build communities on the correct graph and score each head by loss or gain of within-community degree in failure graphs.
3. Select top 10% connectivity-disrupted heads. Compare four insertion/selection modes at matched parameter count:
   - graph-selected zero-init head offsets;
   - activation-magnitude-selected offsets;
   - random-head offsets;
   - Exp 42 fixed L27 slider-head baseline.
4. Train only \(g_h,u_h,v_h\) and any inserted slider-head weights for 8–24 RL steps with core weights frozen. Optionally stack the best graph-selected head offset on the Exp 16 surgical edit.
5. Measure standard policy metrics, pass@4 exact, real-benchmark temperature, connectivity restoration after training, attention-map entropy, and seed variance across at least three prompt folds.

### Expected outcome and falsification criterion

Expected: graph-selected heads beat random and activation-magnitude controls by ≥5 pp exact or produce the same exact score with materially better policy accuracy/false refusal. Falsified if connectivity rankings are unstable across folds, graph selection performs no better than random at matched budget, or connectivity restoration does not correlate with behavioral improvement.

### Citation

- [arXiv:2611.31615 — *Eliciting sandbagged LLM capability via structural interventions on attention-head functional networks*](https://arxiv.org/abs/2611.31615): SVCCA-MI head graphs, within-community connectivity loss, graph-selected ablation/GraphLearned offsets; graph-selected heads reached 70.25/63.60/89.10% versus much lower random-head controls in reported settings.
- Local Exp 28 head-only safety preservation and Exp 42 slider-head validation.

---

## H4 — EMS-guided anchor-preserving subspace edits

**Priority: 8.2/10** *(queued as exp124; probe cost remains high, so interpret it after the cheaper H2/H15 geometry evidence.)*

### Novel idea

Use **Empirical Margin Sensitivity (EMS)** as a causal neuron probe before choosing surgical columns. Instead of ranking columns only by DLA/IG visibility, temporarily “crush” each MLP down-projection column to `sign(w) * mean(abs(w))`—preserving sign pattern and \(L_1\) scale while removing fine-grained coordinate structure—and measure the correct-versus-hardest-wrong decision-margin change. Then design anchor-preserving or negative-anchor-targeting subspace edits.

### Why it should work mechanistically

DLA identifies directions visible in logits, not necessarily columns whose coordinate structure causally supports correct decisions. EMS separates a neuron's presence/scale from its learned structure. Positive EMS anchors should be protected because damaging them causes collateral capability loss; negative EMS columns actively suppress correct margins and are safer targets for structural correction than broad top-DLA sets.

### Concrete experiment design

1. Candidate set: union of top DLA columns, top Fisher/SNIP columns, and matched random columns in layers 27–29.
2. Taylor-screen candidates, then run full EMS on train/dev policy rows using margins for:
   - correct action vs hardest wrong action;
   - correct policy code vs hardest wrong policy code;
   - correct tool vs hardest wrong tool.
3. Split columns into positive anchors, negative-EMS suppressors, and neutral columns with false-discovery control. Validate stability across prompt-family folds.
4. Compare matched-budget edits:
   - current DLA top-64 residual-direction edit;
   - EMS-negative-only top-64 edit;
   - whitened-subspace edit with positive anchors excluded;
   - random-column control;
   - crush-only group ablation as a diagnostic, not a deployable edit.
5. Evaluate standard policy metrics, real-benchmark temperature, group-ablation dose response, and transfer across benign/unsafe prompt families. Use dev gating, never repeated tuning on the official 40-task eval.

### Expected outcome and falsification criterion

Expected: EMS-aware edits exceed the 65–67.5% exact-compliance operator plateau while preserving unsafe refusal at ≥95%, or achieve the same exact score with less benchmark degradation. Falsified if EMS signs are unstable across folds, high EMS is explained by weight dispersion rather than behavior, or EMS-aware selection does not beat DLA/random controls.

### Citation

- [arXiv:2611.30675 — *Empirical Margin Sensitivity: An Interventional Neuron-Level Probe of Post-Trained Medical LLMs*](https://arxiv.org/abs/2611.30675): sign/\(L_1\)-preserving neuron crush, correct-vs-hardest-wrong margin, anchor ablation lowering MedQA by 6–28 points in post-trained checkpoints.
- Local Exps 15/17/20/39/41 localization and replication lessons.

---

## H5 — Output-matched steering-vector compression as RL initialization

**Priority: 7.2/10** *(mixed after exp50/100: pre-edit vector RL can be strong, but post-edit vector transfer is weak and does not beat the surgical anchor; compression remains a complement to H21, not a default mechanism.)*

### Novel idea

Before doing vector RL, distill the behavioral effect of a validated surgical edit—or of a short successful RL endpoint—into a tiny set of input-independent steering vectors by directly matching the edited model's output distribution or final hidden states on cached prompts. Test whether those vectors can **replace** part of the destructive weight edit or **initialize** Exp 50-style vector RL closer to a good basin.

### Why it should work mechanistically

A weight update is input-dependent, but much of the behavioral gain from post-training may be a constant residual-stream shift that steers the model toward computation it already possesses. Output/velocity matching is stronger than averaging activation differences because it fits the behavioral effect rather than copying intermediate noise. In the Halo JSON task, the shift may need to be conditional only on token type—schema tokens versus action/tool/policy decision tokens—not on the full prompt.

### Concrete experiment design

1. Cache original-model and edited-model forward passes on 64 train/dev prompts plus general-benchmark control prompts.
2. Fit zero-init per-block steering vectors to minimize KL or MSE between the steered unedited model and the surgically edited model. Evaluate sequentially with earlier vectors active so block effects are not double-counted.
3. Layouts:
   - all layers versus only layers 24/27/28/29;
   - one vector per block versus separate schema/decision-token vectors;
   - all-token versus decision-token-only loss;
   - mean-difference vectors versus output-matched vectors.
4. Arms:
   - edit-only;
   - vector-only replacement with no weight edit;
   - edit+vector initialization followed by 8-step frozen-core vector RL;
   - Exp 50 zero-vector RL control;
   - random-vector and mean-difference controls.
5. Measure standard policy metrics, real-benchmark temperature, KL to unedited base on off-task text, fraction of surgical-edit gain recovered, vector norms, and RL steps required to reach the best exact-compliance plateau.

### Expected outcome and falsification criterion

Expected: output-matched vectors recover at least 50–70% of the surgical edit's exact-compliance gain without touching core weights, and edit+vector initialization improves sample efficiency or final exactness relative to Exp 50's zero-vector RL. Falsified if vector-only recovery is below 25%, output matching does not beat mean-difference/random controls, or prefitting does not reduce RL steps or final collateral.

### Citation

- [arXiv:2611.26627 — *RL Post-Training in Diffusion Transformers Is Activation Steering*](https://arxiv.org/abs/2611.26627): fixed block-output vectors with 65× fewer parameters than LoRA recovered 90.0–96.3% of RL reward gains on six image models; vectors trained with online RL from zero exceeded weight-based RL gains in some settings.
- [Anthropic — *Evaluating feature steering*](https://www.anthropic.com/research/evaluating-feature-steering): steering has a useful dose window and degrades general capability outside it, motivating explicit benchmark temperature measurement.
- Local Exp 50 vector-only RL and Exp 16/41 surgical operators.

---

## H6 — Input-adaptive Kronecker lanes for complementary attention/MLP edits

**Priority: 8.7/10** *(queued as exp123; strengthened because complementary MLP-only LoRA is now a validated transfer path, but it must beat that simpler comparator.)*

### Novel idea

Replace or augment Exp 57's conventional LoRA lanes with a **Kronecker-factored, input-gated adapter**: freeze a random dense matrix \(A\), train a much smaller factor \(M(x)=B_1\operatorname{diag}(g(x))B_2\), and let one shared gate choose the effective adapter rank per input in discrete lattice steps. Unlike a scalar slider, this changes the *rank/capacity* of the update rather than merely scaling a fixed direction.

### Why it should work mechanistically

The policy-router task appears to mix easy schema reconstruction with harder action/tool/policy-code decisions. Fixed-rank LoRA either over-allocates capacity to easy tokens or under-allocates it to hard decision states. Kronecker multiplication amplifies trainable rank without trainable parameters, while the input gate can allocate high rank only where residual-stream evidence predicts a hard decision. This is a surgical complement to validated edits rather than another destructive full-weight update.

### Concrete experiment design

1. Use the same post-edit anchored setup as Exp 57 so results are directly comparable.
2. Adapt complementary layers 24–27, separately testing attention `q/o` lanes, MLP `up/down` lanes, and both.
3. Before training, compute the paper's cheap adaptivity diagnostic from a few hundred labeled examples: compare within-class and cross-class gate responses for correct allow, false refusal, true refusal, wrong tool, and malformed JSON. Abort or use constant gates if adaptivity is predicted unnecessary.
4. Train small \(B_1,B_2\) and a shared residual-to-gate network; freeze \(A\) and all base weights. Compare matched-parameter LoRA r8/r16, KiRA constant-gate, KiRA input-gated, and a random-gate negative control.
5. Measure standard 40-task metrics, pass@4, exact trainable parameter count, gate rank distribution by behavior, KL to post-edit anchor, real-benchmark temperature, and whether constant gates can be folded into \(B_1\) at deployment.

### Expected outcome and falsification criterion

Expected: input-gated Kronecker lanes match or beat LoRA using substantially fewer trainable parameters, with gains concentrated in hard decision rows rather than schema rows. Falsified if the pre-training adaptivity diagnostic disagrees with measured benefit, input gating does not beat constant/random gates, or parameter-normalized performance is not better than Exp 57 LoRA.

### Citation

- [arXiv:2611.33236 — *Input-Gated Kronecker-Factored Adaptation for Parameter-Efficient Transfer*](https://arxiv.org/abs/2611.33236): frozen random Kronecker matrix multiplies trainable rank, input gate adjusts rank in discrete steps, 8.7–9.7× fewer trainable parameters than LoRA across six benchmarks.
- Local Exp 57 targeted post-edit LoRA transfer design.

---

## H7 — Drift-triggered orthogonal rotation lanes

**Priority: 8.5/10** *(queued as exp120; rotation-only lanes remain attractive but must beat or explain the simpler exp57/100 complementary MLP-only LoRA comparator.)*

### Novel idea

Train **rotation-only lanes** \(W(I+S)\), with skew-symmetric \(S\), in complementary attention/MLP matrices, restoring exact orthogonality only when measured drift crosses tolerance \(\tau\). This redirects existing computation while preserving spectral scale, instead of adding rank or deleting subspaces.

### Why it should work mechanistically

Halo's hard structural edits fail because the 135M model has little redundant capacity: deletion/scaling damages capability. Orthogonal transformations preserve norms and singular values, so they are a natural low-collateral way to rotate false-refusal features into useful compliance/tool-routing representations. Drift-triggered restoration gives exact constraint enforcement when needed without paying the cost of re-orthogonalizing every step.

### Concrete experiment design

1. Start from the validated raw-rank1 post-edit checkpoint and anchor KL to it, as in Exp 57.
2. Apply trainable orthogonal lanes to complementary layers 24–27, with separate arms for attention `q/k/v/o`, MLP `up/down`, and both; use block-diagonal or subspace parameterizations sized to SmolLM2's 576-dimensional residual/64-dimensional heads.
3. Train with 8–24 GRPO steps. After each update compute \(\|S\|_2\); polar-restore only when \(\|S\|_2\ge\tau\) for \(\tau\in\{0.03,0.10,0.30\}\).
4. Compare no-restoration, restore-every-step, drift-triggered restoration, matched-rank LoRA from Exp 57, and random skew-symmetric initialization controls.
5. Measure standard policy metrics, real-benchmark temperature, KL to post-edit anchor, singular-value drift of edited matrices, number/timing of restorations, wall-clock overhead, and per-head attention-map changes.

### Expected outcome and falsification criterion

Expected: drift-triggered rotations preserve general capability better than additive LoRA at similar exact-compliance gains, especially on attention lanes. Falsified if rotations cannot move policy behavior before saturating KL, if restoration does not reduce benchmark/singular-value collateral, or if drift triggering is not faster than restore-every-step at matched accuracy.

### Citation

- [arXiv:2611.33589 — *Drift-Triggered Restoration for Orthogonal Fine-Tuning*](https://arxiv.org/abs/2611.33589): first-order skew update plus drift-triggered polar restoration; defect of \(I+S\) is exactly \(\|S\|_2^2\), block-diagonal version exceeded LoRA by \(1.98\pm0.39\) GLUE points while reducing prior orthogonal-finetuning overhead.
- Local evidence that hard structural edits are destructive but attention-head-only interventions are safe.

---

## H8 — Displacement-probe edit auditor and reference chooser

**Priority: 8.1/10** *(queued as exp125; the second harvest added clean positive and catastrophic-collapse labels that make a cheap pre-screen more valuable.)*

### Novel idea

Build a cheap weight/activation-displacement probe that predicts whether a candidate edit is likely to help before spending GPU rollout time. For each candidate checkpoint, measure probe responses and second-order Gram-feature displacement relative to several references—unedited Halo SFT, raw base, validated edit, and failed destructive edits—then train a tiny tabular predictor on the lab's historical experiment archive.

### Why it should work mechanistically

The lab repeatedly lost GPU time to operators that looked plausible by DLA/weight geometry but were behaviorally destructive. Displacement, not raw weights or activation magnitude, is the causal object introduced by fine-tuning/editing. Gram responses are sensitive to feature changes induced by training, and reference choice is part of the measurement; a learned shared subtraction can remove common architecture scale and expose edit-specific displacement.

### Concrete experiment design

1. Assemble a historical dataset from existing experiments: each row is an edit/RL configuration plus outcome labels for exact compliance, unsafe refusal, benign false refusal, policy accuracy, KL/destruction proxies, and benchmark temperature where available.
2. For each archived or regenerated candidate, compute compact features:
   - weight displacement norms/cosines by layer and module;
   - responses to a fixed set of random and behavior-derived probe inputs;
   - layer-wise Gram-matrix displacement on a small mixed calibration set;
   - displacement relative to multiple references, plus learned shared subtraction fitted on training folds only.
3. Train a small cross-validated predictor/ranker to forecast outcome class: accepted, unsafe-compliance collapse, benign-refusal collapse, no-op, or mixed. Evaluate leave-one-experiment-family-out to avoid leakage.
4. Use the auditor prospectively to rank H1/H2/H4 variants before GPU rollout; compare against simple norms, cosine-to-unembedding, DLA score, and random ranking.
5. Measure ranking quality by top-1/top-3 hit rate, Spearman correlation with exact-compliance gain, false-negative rate on accepted edits, calibration, and GPU minutes saved per accepted candidate.

### Expected outcome and falsification criterion

Expected: displacement/Gram features separate destructive always-call edits and no-op edits from accepted surgical operators better than raw weight norms or DLA visibility, enabling a cheap pre-screen. Falsified if leave-one-family-out prediction is near chance, the auditor rejects the known good Exp 16/41 family, or simple scalar norms perform as well.

### Citation

- [arXiv:2611.33998 — *Refined Displacement Probes for Weight-Space Learning*](https://arxiv.org/abs/2611.33998): probing displacement rather than raw weights, second-order Gram responses, reference-as-design-choice/shared subtraction, state-of-the-art Model Jungle results with 3.61× fewer metanetwork parameters.
- Local failures of DLA/unembedding proxies and repeated operator-validity lessons in Exps 17/39/41.


---

## H9 — Curvature-constrained acquisition deltas

**Priority: 9.1/10** *(queued as exp121; targeted RL works but auxiliary training often damages the surgical anchor, so preservation-aware curvature is directly relevant rather than exploratory.)*

### Novel idea

Train the policy improvement inside a **low-curvature delta space** estimated from preservation data, rather than in a hard behavioral nullspace or an unconstrained LoRA space. Approximate the local curvature of a preservation loss with Kronecker/factorwise spectral structure, truncate high-curvature directions, and optimize only the low-curvature remainder.

### Why it should work mechanistically

Halo's hard nullspaces failed because they encode a coarse linear notion of safety/benign preservation, not the true sensitivity of general computation. The directions that damage capability are precisely high-curvature directions of preservation loss; removing them should permit acquisition updates while reducing collateral even when behaviors are entangled. This is a softer, second-order version of the validated whitened-subspace idea.

### Concrete experiment design

1. Build preservation batches from real-benchmark/general text, benign correct-allow examples, and true unsafe-refusal examples. Build acquisition batches from exact-compliance failures.
2. For layers 24–29, estimate factorwise curvature of preservation loss for attention `q/o` and MLP `up/down`; compare full Hessian-proxy, diagonal-only, random projection, and no curvature control.
3. Sweep retained spectral-energy thresholds and dense versus low-rank delta parameterizations at matched trainable parameter counts. Train 8–24 steps starting from both unedited SFT and validated raw-rank1 post-edit checkpoints.
4. Compare against Exp 23-style hard nullspace projection, Exp 57 LoRA, and the current surgical edit. Evaluate official policy metrics only after dev-gated selection.
5. Measure acquisition/retention tradeoff: exact compliance and pass@4, unsafe refusal, benign false refusal, real-benchmark temperature, KL to base/post-edit anchor, curvature-energy distribution, and GPU/CPU overhead.

### Expected outcome and falsification criterion

Expected: low-curvature deltas achieve a better acquisition-retention frontier than LoRA/hard nullspaces—higher exact compliance at equal or better real-benchmark and unsafe-refusal retention. Falsified if curvature estimates are unstable across preservation batches, the method does not beat diagonal/random controls, or computational overhead outweighs any improvement.

### Citation

- [arXiv:2611.24654 — *CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning*](https://arxiv.org/abs/2611.24654): preservation-constrained adaptation using factorwise spectral truncation of local curvature, with favorable acquisition-retention tradeoffs on reasoning/code and selective unlearning.
- Local Exp 23 hard-nullspace failure and Exp 16 whitened-subspace success.

---

## H10 — Merge-aware served-weight action-token fine-tuning

**Priority: 5.8/10** *(exp101 negative as implemented: export serialization is consistent, but served-weight merge-aware training underperforms the anchor; keep exact export measurement while dropping merge-aware rounding as a default mechanism.)*

### Novel idea

Train through the exact **served/exported weight representation**—including BF16 rounding and checkpoint merging—using a straight-through gradient, while applying likelihood loss only to action/tool/policy-code decision tokens and anchoring the rest of the trajectory. The unit of optimization should be the artifact that will actually be evaluated, not a fp32 hook-time approximation.

### Why it should work mechanistically

The lab has repeatedly seen fp32 versus BF16 and hook versus export mismatches change or erase interventions. Tiny LoRA/vector deltas can fall below rounding resolution or land in the wrong served weights. Optimizing through the rounded merge makes small updates visible where they matter, while action-only loss prevents schema/style drift and preserves the model's existing “thinking” trajectory.

### Concrete experiment design

1. Start from the validated raw-rank1 post-edit checkpoint, exported exactly as the evaluator will load it.
2. Implement served-weight forward passes for BF16/final-export dtype with frozen scales/rounding and straight-through gradients to fp32 shadow weights or adapters.
3. Restrict loss to tokens immediately following `"action": "`, `"tool": "`, and `"policy_code": "`; use zero or heavily downweighted loss on schema punctuation and prompt tokens. Add a sequence-mean drift/KL penalty to the post-edit anchor.
4. Compare standard fp32 LoRA, served-weight LoRA, vector-only served-weight training, action-only versus all-token loss, and a no-export hook control.
5. Measure metrics **after export and reload**, not in the training process: exact compliance, pass@4, unsafe refusal, benign false refusal, real-benchmark temperature, rounding-visible delta magnitude, KL, and export/reload bit or tensor equality.

### Expected outcome and falsification criterion

Expected: served-weight training preserves more benchmark gain after export than standard LoRA and reduces the fp32/hook replication gap, with action-only loss producing less schema collateral. Falsified if export/reload gains vanish anyway, if served-weight training does not beat standard training under the final artifact, or if action-only masking reduces exact-compliance learning materially.

### Citation

- [arXiv:2611.32064 — *Fix How It Acts, Not How It Thinks: Surgical Fine-Tuning of Frontier LLMs*](https://arxiv.org/abs/2611.32064): merge-aware fine-tuning through served BF16/NVFP4 weights with straight-through gradients, action-token-only loss, and evidence that standard LoRA loses most benchmark gain once served.
- Local fp32/BF16 operator-sensitivity and export/replication lessons from Exps 32/39/41.

---

## H11 — Trajectory-aware budgeted decision-token steering

**Priority: 7.1/10** *(queued as exp127; mixed because pre-edit vector RL works but post-edit vector transfer is weak, so inference-time budgeted steering must prove a separate control benefit.)*

### Novel idea

Treat steering as a **per-token allocation problem under a fixed intervention budget**. Use a contrastive feature projection to identify tokens least aligned with correct policy behavior, then use closed-loop trajectory feedback to spend steering only when the generation is falling behind, rather than adding a vector everywhere.

### Why it should work mechanistically

The strict JSON task has a few high-leverage decision positions and many schema positions that should remain deterministic. Always-on steering distorts both. Token-selective steering should obtain most behavioral benefit at lower distribution shift; trajectory-aware feedback further avoids wasting budget once the correct action/tool/policy-code trajectory is already established.

### Concrete experiment design

1. Use steering directions from H5 output matching or activation differences measured on train/dev correct-vs-failure pairs; do not tune on the official eval.
2. Instrument cached rollouts and compare allocation policies at matched intervention budgets:
   - always-on vector;
   - prompt-position-only vector;
   - fixed decision-token positions after `action`, `tool`, and `policy_code`;
   - memoryless contrastive-feature gate;
   - closed-loop adaptive Bernoulli steering using accumulated alignment;
   - random-token control.
3. Sweep budgets as fractions of generated tokens and steering strengths on dev rows. Then evaluate once on the 40-task set.
4. Measure exact compliance, pass@4, unsafe refusal, benign false refusal, schema-malformation rate, token-level intervention rate, distribution-shift/KL proxy, real-benchmark temperature, and latency overhead.
5. Test composition with the surgical edit and with H1/H3 inserts; steering itself remains training-free.

### Expected outcome and falsification criterion

Expected: trajectory-aware decision-token steering achieves most of always-on steering's exact-compliance benefit with lower schema corruption and less benchmark degradation. Falsified if the learned gates select no better than fixed/random positions, if budgeted steering cannot reach the always-on performance frontier at any strength, or if closed-loop feedback adds instability without measurable gains.

### Citation

- [arXiv:2611.18274 — *Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering*](https://arxiv.org/abs/2611.18274): fixed-budget per-token steering, Contrastive Feature Projection, and Closed-Loop Adaptive Bernoulli Steering with stability analysis.
- Local Exp 42 always-on slider-side effects and Exp 56 boundary-token decision-position analysis.


---

## H12 — Closed-form decision-readout correction

**Priority: 8.3/10** *(queued as exp102; H18 is the follow-on weight-folded version if this probe succeeds.)*

### Novel idea

Use frozen hidden states as a readout feature extractor and solve a **closed-form decision-head correction** with shrinkage LDA/kernel ridge, then write that correction back as a tiny gated low-rank output patch at decision tokens. Unlike H5, which compresses an existing surgical/RL effect into constant vectors, this directly estimates the correct action/tool/policy-code decision boundary from labeled train/dev rows without any gradient training.

### Why it should work mechanistically

Many strict-schema failures may be linearly or kernel-linearly readable in late hidden states even when generation chooses the wrong token. A frozen encoder plus closed-form readout is robust in the low-label regime because it cannot wander through weight space. If the residual stream already separates correct from incorrect decisions, a small gated readout correction can exploit that information without retraining or damaging upstream computation.

### Concrete experiment design

1. Cache hidden states at layers 20–29 and token positions immediately before action, tool, and policy-code decisions across train/dev prompt families.
2. Build multiple views: final-layer state, mean of late layers, decision-token state, preceding-schema state, and contrastive correct-vs-failure differences.
3. Fit a candidate pool exactly: shrinkage LDA, PCA+LDA, linear ridge, random-feature ridge, and small RBF/additive kernel ridge. Select by cross-fitted dev predictions only.
4. Convert the selected readout into a sparse logit/residual correction active only at decision positions. Compare raw probe accuracy, corrected greedy generation, corrected sampling/pass@4, and composition with the validated surgical edit.
5. Measure exact compliance, unsafe refusal, benign false refusal, schema malformation, correction firing rate, real-benchmark temperature, parameter/storage cost, and determinism across repeated runs.

### Expected outcome and falsification criterion

Expected: closed-form readouts show high train/dev decision accuracy and produce a meaningful exact-compliance gain with little or no general-capability cost. Falsified if cross-fitted readout accuracy does not exceed simple linear probes/majority controls, if probe accuracy fails to transfer to generation corrections, or if the correction only works when fitted directly to eval-like prompts.

### Citation

- [arXiv:2611.33972 — *Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder*](https://arxiv.org/abs/2611.33972): exact shrinkage-LDA/kernel-ridge readouts beat full fine-tuning and LoRA in low-label regimes, with deterministic no-gradient adaptation.
- Local need for tiny, reproducible decision-boundary corrections in strict JSON policy routing.

---

## H13 — Zero-init attention-latent feedback lane

**Priority: 6.1/10** *(demoted further: memory-head collapse plus exp61/66 attention-composition failures and unresolved seed fragility make feedback lanes a distant exploratory branch.)*

### Novel idea

Test whether a small model's raw capability can be improved by an **additive attention-latent feedback lane** that lets the next token read selected deep sublayer states of the preceding token. Initialize the feedback path to exact zero, train only its low-rank source maps/tied KV attention, and compare fixed nearest-window feedback against learned attention.

### Why it should work mechanistically

A 135M model has limited serial depth per token. Deep-state feedback gives it a second chance to reuse contextual computation across positions without increasing the width or overwriting existing weights. The fresh ALF result suggests intermediate/deep nearby layers—not just final states—carry useful feedback, and that much of the gain survives single-pass decoding when previous-token states are cached.

### Concrete experiment design

1. Implement a zero-init additive lane that caches selected post-attention/post-MLP states from the previous token and injects them into the next token after layer 0 or a mid-layer interface.
2. Compare fixed nearest-window feedback from layers 24/27/29 versus learned tied-KV attention over those states; include final-state-only and random-projection controls.
3. Train only feedback parameters on a small mixture of general text plus policy-rollout teacher-forced data, preserving exact identity at initialization. Use a short continued-training budget, not full pretraining.
4. Measure MMLU/ARC-Easy/HellaSwag temperature as the primary raw-performance endpoint, plus WikiText/perplexity proxy, policy metrics, added latency, cache memory, and whether fixed-window feedback retains most of the learned-attention gain.
5. If the lane helps generally, test composition with H1/H3 attention inserts and the validated surgical edit.

### Expected outcome and falsification criterion

Expected: even a short zero-init adaptation produces neutral-to-warm benchmark movement or a perplexity gain, with fixed nearest-window feedback capturing most benefit at low cost. Falsified if the lane remains near zero/unused, if gains vanish under single-pass decoding, or if added latency/cache cost dominates any quality improvement on the small model.

### Citation

- [arXiv:2611.31154 — *Attention Latent Feedback*](https://arxiv.org/abs/2611.31154): deep-state feedback lowered held-out loss across scales, matched a 1.38× larger token budget at MoE-1.6B, improved MoE-4.0B seventeen-benchmark average accuracy 45.1→47.0, and retained exact single-pass decoding via cached prior-token states.

---

## H14 — Spectral-sink calibrated post-SDPA microgates

**Priority: 7.1/10** *(queued as exp126; exploratory, with no direct local evidence yet.)*

### Novel idea

Add or train tiny **post-SDPA multiplicative microgates** for existing attention heads, initialized exactly at identity, and calibrate them using spectral measurements of attention-sink structure rather than behavior loss alone. The goal is not ablation: it is to suppress pathological sink-aligned spectral spikes only where they measurably dominate head outputs.

### Why it should work mechanistically

Attention sinks create low-rank spectral spikes that can dominate token-space covariance and reduce effective rank. Post-SDPA gates can contract sink-induced modes and act as a value-path gradient valve. In a small model, a few heads may be sacrificing representational bandwidth to sink behavior; gentle identity-initialized gating could recover capacity without deleting heads or overwriting MLP weights.

### Concrete experiment design

1. Measure per-head attention mass on first/sink tokens, covariance spectra, effective rank, sink-subspace alignment, and value-gradient proxies across correct, false-refusal, true-refusal, malformed, and general-benchmark inputs.
2. Add scalar or low-dimensional vector gates after SDPA for each existing head, initialized exactly at one. Compare:
   - uniform learned gates;
   - gates only on top sink-aligned heads;
   - spectral-regularized gates penalizing dominant covariance eigenvalue;
   - random-head gates;
   - no-gate control.
3. Train gates with a small exact-compliance objective plus a benchmark/KL preservation penalty; alternatively derive a one-shot spectral calibration and use RL only as a refinement.
4. Measure standard policy metrics, MMLU/ARC/HellaSwag temperature, attention entropy, sink mass, covariance effective rank, alignment reduction, KL to base, and gate distribution by layer/head.
5. Test composition with H3 graph-selected offsets and H1 modular inserts, since all three target attention rather than MLP refusal directions.

### Expected outcome and falsification criterion

Expected: sink-aligned selective gates increase effective rank and yield neutral-to-positive raw-benchmark movement without harming policy metrics; behavior gains, if any, should come at lower KL than dense head training. Falsified if SmolLM2 does not exhibit harmful sink spectra, if spectral suppression is uncorrelated with behavior/benchmark outcomes, or if gates collapse heads and reproduce the hard-ablation failure mode.

### Citation

- [arXiv:2611.20403 — *Attention Sinks as Spectral Spikes: A Mechanism Analysis of Gated Attention*](https://arxiv.org/abs/2611.20403): post-SDPA gating suppresses sink-induced spectral structure, reduces sink-subspace alignment/outlier mass, increases effective rank, and attenuates first-token value-gradient amplification.


---

## H15 — Multi-perspective abstention-direction factorization

**Priority: 9.2/10** *(queued as exp87; both the best surgical and best localized-RL arms land at 93.3% unsafe refusal, making selective false-vs-true refusal geometry the central measurement.)*

### Novel idea

Measure separate abstention directions for distinct refusal causes using minimal counterfactual pairs, then factorize or orthogonalize them before editing. Halo currently treats “false refusal” and “true unsafe refusal” as two broad activation clouds; this hypothesis tests whether query ambiguity, knowledge/policy limits, and harmfulness occupy partially separable geometric directions whose overlap explains collateral.

### Why it should work mechanistically

The lab's core bottleneck is entanglement: edits that remove benign over-refusal also weaken true unsafe refusal. Fresh evidence suggests abstention is not one monolithic direction: different abstention perspectives have asymmetric cross-effects, different cosine overlaps, and middle-layer Fisher separation. If Halo's benign/unsafe pairs are constructed as controlled counterfactuals, layerwise geometry can identify where separation is real rather than assumed by a hand-built nullspace.

### Concrete experiment design

1. Construct three train/dev counterfactual families with minimal lexical/structural changes:
   - benign correct-allow versus benign false-refusal;
   - harmful true-refusal versus harmless counterpart;
   - known/valid policy-tool decision versus unknown/invalid or ambiguous counterpart.
2. Teacher-force positive/negative completions and compute layerwise mean-difference abstention directions, cosine overlap, cross-perspective transfer, and Fisher discriminant ratio.
3. Select layers where the target benign/unsafe separation has high FDR but harmfulness overlap is low. Compare raw mean directions, orthogonalized directions, whitened subspaces, and single-direction controls.
4. Intervene first by activation steering, then convert the best geometry into a surgical residual-subspace edit. Compare against Exp 16 `raw_rank1`, Exp 41 false-node operator, and random direction controls.
5. Measure standard policy metrics, per-family abstention rates, cross-perspective leakage, layerwise FDR/cosine stability across folds, real-benchmark temperature, and dose-response before selecting any accepted edit.

### Expected outcome and falsification criterion

Expected: at least one middle/late layer shows high benign-versus-harmful geometric separability, enabling an edit that lowers benign false refusal without the usual unsafe-refusal cost. Falsified if directions are nearly collinear across all layers, cross-perspective effects are symmetric and unavoidable, high FDR does not predict intervention selectivity, or orthogonalization performs no better than the existing whitened subspace.

### Citation

- [arXiv:2611.20441 — *Abstention Direction in Large Language Models: Mechanistic Interpretability of Abstention Across Query Ambiguity, Knowledge Limits, and Safety Alignment*](https://arxiv.org/abs/2611.20441): counterfactual-pair abstention directions, asymmetric cross-perspective effects, layerwise cosine overlap, and Fisher separation peaking in middle layers.
- Local Exp 19 crosstalk audit showing false/true refusal movement coupled near ratio 1.0.

---

## H16 — Norm-calibrated single-coordinate decision steering

**Priority: 5.4/10** *(exp67 negative at tested doses: aligned/opposite/random single-coordinate steering has no positive operating point and high dose collapses; treat as falsified unless a nonlinear/readout actuator replaces the raw coordinate.)*

### Novel idea

Use contrastive correct-versus-wrong completions to search for a **single MLP coordinate** whose activation can be shifted by a calibrated dose at decision tokens. Normalize the dose by both the layer's median residual norm and the corresponding down-projection column norm, then test signed hot/cold control rather than distributing noise across a full vector.

### Why it should work mechanistically

A sparse coordinate is easier to audit and less likely to entangle unrelated features than a dense steering vector. Because the neuron's residual contribution is its down-projection column, norm calibration makes doses comparable across layers and prevents accidental massive activations. If one or a few coordinates act as decision keys, this gives a minimal positive control for H1's keyed implants and H4's EMS edits.

### Concrete experiment design

1. Teacher-force correct and incorrect action/tool/policy-code completions at layers 24–29 and record down-projection-input activations.
2. Rank neurons by normalized contrastive difference, stability across prompt folds, and low correlation with hub/unsafe-refusal coordinates. Include same-layer random and DLA/EMS-selected controls.
3. Apply dose \(\Delta a_n=d\,m_L/\lVert W_{\rm down}[:,n]\rVert_2\) only at decision tokens, sweeping sign and magnitude. Compare all-token versus decision-token application and positive/negative schedules.
4. Use the best coordinate as (a) a training-free steering control, (b) a target for a keyed implant in H1, and (c) a validation target for EMS-positive/negative interpretation in H4.
5. Measure exact compliance, unsafe refusal, benign false refusal, schema malformation, dose-response monotonicity, specificity on paraphrases, real-benchmark temperature, and output-collapse threshold.

### Expected outcome and falsification criterion

Expected: at least one sparse coordinate produces a monotone behavioral effect at doses below output collapse, with decision-token gating reducing collateral relative to all-token steering. Falsified if no coordinate beats random controls, effects are non-monotonic or fold-unstable, or behavioral changes are explained by generic incoherence rather than decision-specific control.

### Citation

- [arXiv:2611.29578 — *Single-Coordinate Activation Steering for Controllable Social Stance in Language Models*](https://arxiv.org/abs/2611.29578): one-coordinate MLP intervention calibrated by median residual norm and down-projection column norm, contrastive neuron search, signed hot/cold schedules, and random-control comparisons.

---

## H17 — Prepare-for-truncation elastic low-rank deployment

**Priority: 7.0/10** *(exploratory deployment-compression work; behind policy-repair validation.)*

### Novel idea

Train the final Halo checkpoint **through a low-rank view** so one prepared model can be SVD-truncated to multiple deployment budgets without separate fine-tuning. This is a raw-capability/deployment hypothesis: preserve policy behavior and benchmark quality while reducing inference memory/compute.

### Why it should work mechanistically

Post hoc SVD compression damages directions that training never needed to make compressible. Preparing the model against activation-aware fixed bases aligns useful computation with truncatable subspaces and makes updates learned at one compression ratio transfer to neighboring ratios. The result is a better quality-versus-memory frontier than compressing after all post-training is finished.

### Concrete experiment design

1. Start from the best accepted post-edit/RL checkpoint once available; also test unedited Halo SFT as a control.
2. Estimate activation-aware preparation bases from mixed general-text and policy calibration data. Prepare at a middle ratio such as \(\rho=0.6\) using student-to-teacher KL from the original checkpoint.
3. Without further fine-tuning, compress to ratios \(r\in\{0.8,0.6,0.4\}\) using the same decomposition. Compare post hoc SVD, budget-specific fine-tuning, random bases, CE-only preparation, and KL+CE preparation.
4. Measure MMLU/ARC/HellaSwag temperature, policy exact/unsafe/benign-FR, JSON schema validity, latency/memory, spectral-energy retention, and cross-ratio transfer.
5. Test interaction with H10: exported compressed artifacts must be evaluated after exact merge/reload, not merely in the training process.

### Expected outcome and falsification criterion

Expected: prepared checkpoints retain more benchmark and policy performance than post hoc compression at 0.8/0.6 ratios, with graceful degradation at 0.4. Falsified if preparation does not beat post hoc SVD, gains vanish after exact export/reload, or compression ratios useful for deployment destroy strict JSON policy behavior regardless of preparation.

### Citation

- [arXiv:2611.34667 — *Learning to Be Truncated: Preparing LLM Checkpoints for SVD Compression*](https://arxiv.org/abs/2611.34667): one prepared LLaMA-7B checkpoint improved quality across multiple compression ratios and compression operators without budget-specific fine-tuning; preparation used fixed activation-derived bases and KL distillation.

---

## H18 — Protected FisherGate decision-readout edit

**Priority: 8.6/10** *(high potential but staged behind exp102 because it should only be implemented if decision errors are linearly/kernel-linearly readable.)*

### Novel idea

Convert the closed-form decision boundary into a **permanent zero-inference-overhead readout edit** using Protected FisherGate: covariance-whiten the false-refusal/failure direction, orthogonalize it against correct benign, correct unsafe-refusal, and post-decision states, then write a tiny rank-one output-readout patch. Unlike H12's diagnostic/correction probe, this tests whether the decision boundary can be folded into weights once, with no per-query classifier or hook.

### Why it should work mechanistically

Halo's strict action decision is concentrated in a few token positions and may resemble the reasoning-delimiter problem solved by Protected FisherGate: representation separability alone is insufficient, because unconstrained Fisher/LDA directions overfit covariance and damage non-target states. Explicit protection of correct decision and post-decision subspaces can preserve normal computation while shifting only the failure-direction logit gap. SmolLM2's tied embeddings make naive unembedding edits dangerous, so cloning/untieing the readout or using an output-only low-rank patch is part of the hypothesis rather than an implementation detail.

### Concrete experiment design

1. Cache final hidden states immediately before action/tool/policy-code value tokens on teacher-forced train/dev rows. Build four state groups: correct benign `call_tool`, benign false refusal, correct unsafe refusal, and post-decision/schema-continuation states.
2. Compute covariance-whitened Fisher/ridge directions separating false-refusal or wrong-action states from correct decision states. Construct \(Q_{\rm keep}\) from correct-benign, correct-unsafe, malformed-free schema, and post-decision mean states; enforce \(Q_{\rm keep}^\top d\approx0\).
3. Compare Protected FisherGate, unconstrained Fisher LDA, ridge, orthogonal-only, random-direction, and H12 probe-only controls at matched norm/dose.
4. Output-write implementations:
   - clone/untie `lm_head` before editing because `tie_word_embeddings=true`;
   - a zero-init low-rank output adapter limited to the vocabulary rows for first tokens of competing action/tool/policy-code values;
   - a tied-embedding edit negative control to measure input-side contamination.
5. Calibrate dose from the empirical decision logit gap, then sweep 0.5×/1×/1.5×. Evaluate exact compliance, unsafe refusal, benign false refusal, malformed JSON, decision-token firing margins, real MMLU/ARC/HellaSwag temperature, parameter overhead, export/reload fidelity, and cross-fold stability.
6. Compose the best arm with the validated surgical edit only after H19/exp87 establish whether true-refusal protection is separable.

### Expected outcome and falsification criterion

Expected: a protected closed-form readout edit produces at least +5 pp exact or policy accuracy with no more than one unsafe-refusal regression and neutral real-benchmark temperature. Falsified if unconstrained and protected directions perform similarly poorly, protection eliminates the behavioral effect, gains do not survive export/reload, or tied-embedding contamination/general-benchmark collateral is unavoidable without large parameter overhead.

### Citation

- [arXiv:2611.34612 — *Do We Really Need an LLM for Weight-Space Programming?*](https://arxiv.org/abs/2611.34612): Protected FisherGate combines covariance-whitened discrimination with exact normal-state orthogonalization; a frozen patch reached 84.2% HumanEval pass@1 at 1K budget and beat budget forcing by +14.6 pp while transferring across backends.
- [arXiv:2611.33972 — *Fine-Tuning Without Gradients?*](https://arxiv.org/abs/2611.33972): closed-form frozen-encoder classifiers motivate the diagnostic predecessor H12.

---

## H19 — Context-gated rank-one surgical dose cancellation

**Priority: 8.9/10** *(exp136 sparse result is mixed: oracle cancellation restores unsafe refusal to 100% but costs 15 pp exactness, while the learned gate reaches only 75/93.3/0 versus edit-only 80/93.3/0; the algebra is validated but context separation is not yet good enough.)*

### Novel idea

Cancel the validated rank-one surgical edit **only in true-unsafe contexts**. Because the edit's module-output perturbation is exactly \(\Delta W x=-s u(p^\top x)\), a hook or distilled parallel neuron can measure the same scalar coefficient and add back \(+g(x)s u(p^\top x)\), where \(g(x)\) is a learned exact-zero gate for correct unsafe-refusal contexts. This is an input-conditioned dose control, not a dense anti-refusal vector.

### Why it should work mechanistically

The best surgical and targeted-RL arms both retain about 93.3% unsafe refusal: most safety behavior survives, but roughly 2/30 true refusals are collateral casualties. If exp87/H15 shows false-refusal and true-refusal states are separable, then the rank-one edit does not need a complex subspace repair; its context selection is simply too broad. Exact algebraic cancellation in the wrong contexts can preserve the false-refusal correction elsewhere while restoring true-refusal computation, avoiding the destructive always-call failure of global unembedding or dense steering edits.

### Concrete experiment design

1. Start from the best surgical operating point selected by exp95/98/99 or the Exp 1 arm1 reproduction. Record the exact edited layer(s), direction \(u\), projection vector \(p\), strength \(s\), and selected columns.
2. Use exp87/H15 states to fit a conservative gate separating correct unsafe refusal from benign correct and benign false-refusal contexts. Compare linear Fisher/LDA gates, diagonal-whitened gates, exact-zero ReLU/step gates, and random-key controls. Measure fold stability and firing rates.
3. Implement an oracle control first: cancellation active only on heldout rows known to require unsafe refusal. Because exact cancellation should revert those rows toward the unedited model, this validates the coefficient/hook implementation but is **not** scientific evidence by itself. Then implement learned-gate cancellation at decision-token positions only. Include all-token cancellation, no cancellation, random-context cancellation, and benign-context cancellation controls.
4. Sweep cancellation fraction \(c\in\{0.25,0.5,0.75,1.0\}\). The exact hook computes the edit coefficient from the current module input; do not approximate it with a constant vector.
5. If the hook upper bound works, distill the cancellation into an exportable parallel neuron/head-style insert whose key computes the gate and whose value writes \(+c s u(p^\top x)\); compare hook, distilled, and static-weight implementations after exact export/reload.
6. Measure exact compliance, unsafe refusal, benign false refusal, action/policy accuracy, JSON validity, real-benchmark temperature, gate confusion matrix, firing rate on control text, KL off-task, and seed robustness.

### Expected outcome and falsification criterion

Expected: oracle cancellation reproduces unedited-model unsafe refusal as an implementation sanity check. The real hypothesis is that learned-gate cancellation preserves at least 70–75% exact compliance and 0–10% benign false refusal while lifting unsafe refusal toward 97–100%. Falsified if oracle cancellation fails to revert behavior (meaning the coefficient or edit reconstruction is wrong), if learned gates cannot separate contexts better than chance across folds, or if exactness collapses because false and true refusal states share the same trigger manifold.

### Citation

- [arXiv:2611.34762 — *TASK Memory: Beyond Text-Only Memory with Activation Steering and Knowledge Editing against Memory Hallucination*](https://arxiv.org/abs/2611.34762): paired steering vectors decide query-by-query whether the model adopts an external memory or keeps its own answer, showing the value of explicit adoption/context control rather than always-on steering.
- [arXiv:2406.11717 — refusal-direction mediation](https://arxiv.org/abs/2406.11717): refusal behavior can be mediated compactly, but global direction erasure is unsafe; H19 is the context-selective alternative.
- Local Exp 1 arm1/edit-only evidence, exp87 geometry gate, and the rank-one algebra made explicit by exp122.

---

## H20 — Kolpona-style accept/reject failure-habit neuron ladder

**Priority: 8.4/10** *(staged behind exp103/124 because it should add sequential closed-form writes and paired acceptance, not duplicate simple suppression dose response; ultra-sparse dose evidence indirectly supports neuron-level precision.)*

### Novel idea

Program a tiny ladder of single MLP neurons, one recurring failure habit at a time, and keep each edit only when paired evaluation on fresh dev tasks favors it. Each neuron has a closed-form trigger/read/write program: trigger on the state where the failure habit starts, read the relevant residual/MLP feature, and write a corrective direction while remaining exponentially inactive off-trigger.

### Why it should work mechanistically

The lab's failures are heterogeneous: benign over-refusal, wrong tool, wrong policy code, malformed JSON, and benchmark reasoning failures need not share one direction. Broad suppression or dense RL trades one failure against another. Single-neuron closed-form programs can remove narrow failure habits with minimal collateral, while paired fresh-task acceptance prevents practice-set overfitting and degenerate composition. Kolpona shows this can scale: editing 16 neurons improved a 4B model's VerilogEval pass@1 from 31.7% to 55.4% and transferred to Python benchmarks.

### Concrete experiment design

1. Build a failure-habit inventory from policy train/dev rollouts plus `benchmark_failure_tree` once results land: benign false refusal, wrong tool, wrong policy code, malformed JSON, benchmark answer contradiction, and premature termination.
2. For each habit, identify trigger onset states and closed-form key/read/write vectors using only practice/dev rows. Use low-collision non-hub neurons and protect high-EMS positive-anchor columns identified by exp124.
3. Write one neuron at a time. After each write, run paired same-seed evaluation on fresh dev tasks and a small real-benchmark probe; accept only if target failure improves without unsafe-refusal or benchmark regression. Otherwise revert and move to the next habit.
4. Compare sequential accept/reject ladder, batch-write all neurons, suppress-only exp103-style edits, random-neuron writes, and no-acceptance control. Cap at 4–8 neurons for the first study.
5. Measure exact compliance, unsafe refusal, benign false refusal, per-failure-habit rates, pass@1/pass@4, real MMLU/ARC/HellaSwag temperature, off-task KL, number of accepted/reverted writes, exportability, and cross-seed stability.

### Expected outcome and falsification criterion

Expected: a small accepted ladder improves exact compliance or benchmark temperature beyond suppress-only controls with little collateral, and acceptance rejects visibly destructive writes before composition. Falsified if accepted edits merely overfit practice/dev rows, sequential composition performs no better than batch/random writes, or the acceptance criterion rejects all writes because single neurons cannot move behavior.

### Citation

- [arXiv:2611.33973 — *Kolpona: An Agentic Framework with Gradient-Free Neuron Programming for RTL Generation*](https://arxiv.org/abs/2611.33973): closed-form single-neuron programs remove failure habits, paired fresh-task acceptance decides whether to keep each edit, and 16-neuron edits produced large cross-domain pass@1 gains.
- [arXiv:2611.34612 — *Do We Really Need an LLM for Weight-Space Programming?*](https://arxiv.org/abs/2611.34612): deterministic probing, blind routing, operator boundaries, and paired verification/rollback motivate making the procedure deterministic rather than agent-improvised.


---

## H21 — Exportable consolidation of complementary steering-vector RL

**Priority: 8.8/10** *(revised after exp101: hook-bound vector-RL behavior still needs exportable consolidation, but rounding/merge-aware training alone is insufficient; teacher/KL distillation is the active hypothesis.)*

### Novel idea

Use the successful exp80/exp86 steering-vector endpoints as teachers and distill their behavioral effect into ordinary weight deltas with output-matched, action-token-weighted, merge-aware training. The student should be evaluated only after exact merge+BF16 export. This combines the two strongest new lessons: complementary-layer vector RL can carry most behavioral gain, and hook-time success is irrelevant unless it survives the served-weight artifact.

### Why it should work mechanistically

Exp80 shows input-independent vectors at layers 0/1/24–27 can reach **80/93.3/10** without any surgical scaffold; exp86 shows oscillating scaffold polarity prevents the destructive fixation seen in constant-polarity controls and reaches **72.5–75/93.3/0** after scaffold removal. Those vectors are additive hooks, not a portable checkpoint. Because the desired shift is small and concentrated in complementary layers, a low-rank weight student may be able to reproduce the teacher's input-dependent consequence while preserving general computation through KL/control matching.

### Concrete experiment design

1. Reproduce or reload two teacher vector endpoints:
   - exp80 `rl_only` (higher exactness, 10% benign-FR);
   - exp86 `last_positive_endpoint` or `positive_negative_vector_soup` (lower exactness, 0% benign-FR).
2. Build students from unedited Halo SFT and from the validated ultra-sparse surgical operating point (16 columns/layer, strength 0.9). Train complementary-layer low-rank deltas first, prioritizing the exp57-validated MLP-only layers 24–27 before any attention/orthogonal variant.
3. Distill with a mixed objective:
   - action/tool/policy-token KL or CE to the vector-teacher on policy train/dev prompts;
   - general-text KL anchor to unedited SFT on MMLU/ARC/HellaSwag train-style controls and paraphrase prompts;
   - exact exported-artifact evaluation as a harness; optional straight-through BF16 rounded forward only as an arm, because exp101 shows rounding-aware training alone is not sufficient.
4. Arms: hook teacher upper bound; output-matched student; mean-activation student; direct exp101-style action-only student; oscillating-endpoint soup student; random rank-matched student; unedited and edit-only controls.
5. Evaluate only exported merged artifacts as primary: exact compliance, unsafe refusal, benign false refusal, action/policy accuracy, JSON validity, real benchmark plus continuation-scored benchmark, off-task KL, export gap, parameter count, and cross-seed stability.
6. Optional final step: apply exp131/H19 cancellation if the student inherits the surgical edit's unsafe-refusal collateral.

### Expected outcome and falsification criterion

Expected: the exported student recovers at least 60–70% of the hook teacher's exact-compliance gain while keeping unsafe refusal within one task of the teacher and reducing hook/export mismatch to near zero. Falsified if distillation recovers less than 30% of the teacher gain, exported artifacts lose substantially more exactness than hook-time students, general-benchmark temperature degrades, or the vector teacher itself fails seed replication.

### Citation

- [arXiv:2611.26627 — *RL Post-Training in Diffusion Transformers Is Activation Steering*](https://arxiv.org/abs/2611.26627): small fixed block-output vectors can recover most weight-RL gain, supporting vector-to-weight compression as a plausible adaptation bottleneck.
- [arXiv:2611.32064 — *Fix How It Acts, Not How It Thinks*](https://arxiv.org/abs/2611.32064): action-focused surgical fine-tuning motivates action-token-weighted distillation rather than whole-trajectory imitation.
- Local exp80 `rl_only`, exp86 oscillating polarity, and H10 merge-aware export constraints.


---

## H22 — Ultra-sparse edit anchored complementary MLP-only LoRA

**Priority: 9.7/10** *(queued as exp144; highest-priority composition because both parents are independently validated, but the combination has not yet been tested.)*

### Novel idea

Apply the new ultra-sparse residual-direction edit—layers 29/28, 16 columns per layer, strength 0.9—and then run the exp57/exp100-validated transfer recipe on top: freeze the edited layers, KL-anchor to the post-edit model, and train complementary MLP-only LoRA at layers 24–27. This is a direct composition of the strongest surgical operator (**80/93.3/0**) with the strongest reproducible weight-space transfer mechanism (**75/90/0** on the weaker canonical edit), rather than another novel mechanism.

### Why it should work mechanistically

The ultra-sparse edit removes most false-refusal behavior with less collateral than the 64-column canonical edit, but it still leaves tool/policy mistakes and 2/30 unsafe-refusal failures. Complementary MLP-only LoRA supplies input-dependent correction outside the edited layers, so it can repair decision computation without overwriting the rank-one surgical direction. Freezing the edited layers and anchoring to the post-edit model should avoid the destructive interference seen when auxiliary training touches the surgical layers or uses broad full fine-tuning.

### Concrete experiment design

1. Build the sparse anchor exactly as exp129/130: raw rank-1 residual subspace at layers 29/28, top 16 projection columns per layer, strength 0.9. Confirm the anchor reproduces approximately **80/93.3/0** before training.
2. Freeze layers 29/28 and KL-anchor to the post-edit sparse model.
3. Train complementary MLP-only LoRA r16 at layers 24–27 using the exp57 recipe (8 GRPO-style steps, group size 4, lr 1e-4, beta 0.02) as the primary arm.
4. Controls:
   - sparse edit only;
   - canonical 64-column edit + the same complementary MLP LoRA (exp57 replication);
   - sparse edit + all-proj LoRA;
   - sparse edit + edited-layer-inclusion negative control;
   - complementary MLP LoRA on unedited SFT;
   - vector-only lower bound;
   - random-layer LoRA control.
5. Run seeds 7/42/123 because both slider and RL-adjacent results have shown seed sensitivity. Evaluate exported merged artifacts, not only hook-time adapters.
6. Measure exact compliance, unsafe refusal, benign false refusal, action/policy accuracy, JSON validity, off-task KL, legacy benchmark, continuation benchmark, parameter count, wall time, and hook/export mismatch.

### Expected outcome and falsification criterion

Expected: at least two seeds exceed the sparse-edit-only 80% exact-compliance operating point or match it while restoring unsafe refusal to ≥96.7%, with benign false refusal remaining 0% and no benchmark/continuation regression. The hypothesis is falsified if complementary LoRA gives no additive gain over sparse edit only, if unsafe refusal drops below 90%, if benign false refusal reappears, or if gains are seed-specific/export-dependent.

### Citation

- Local exp129 `strength_0p9` and exp130 `columns_16`: **80/93.3/0** ultra-sparse surgical operator.
- Local exp57 `halo_regex_complementary_mlp_r16` and exp100 `targeted_lora_complement_mlp_only`: complementary MLP-only LoRA is the best validated transfer path.
- [arXiv:2611.32064 — *Fix How It Acts, Not How It Thinks*](https://arxiv.org/abs/2611.32064): motivates narrow action-focused adaptation rather than broad full-model training.


---

## Near-term research backlog for later batches

Candidates for later batches: tool-free LLM quantization-error cancellation ([arXiv:2611.33031](https://arxiv.org/abs/2611.33031)); mixture-of-loops KV memory as a longer-horizon architecture insert ([arXiv:2611.34923](https://arxiv.org/abs/2611.34923)); hierarchical graph metanetworks as a stronger H8 auditor backbone ([arXiv:2611.34726](https://arxiv.org/abs/2611.34726)).
