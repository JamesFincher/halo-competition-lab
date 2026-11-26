# Experiment Hypotheses — Living Pipeline

Last updated: 2026-11-26 by the hypothesizer agent.

## Operating context

Validated local assets:

- **Best surgical operator:** measured residual/pass-vs-failure subspace edits, not unembedding-logit directions. Exp 16 `raw_rank1` reached **65% exact / 90% unsafe refusal / 0% benign false refusal**; Exp 41's faithful contrastive+dev-gated operator reached **67.5% exact / 100% unsafe / 80% benign-FR** but did not reproduce the older 80% CPU-era result.
- **Best architectural insert:** Exp 42's zero-init slider-gated attention head was exactly identical at initialization and reduced benign false refusal **90%→30%** at \(\alpha\approx0.031\), but exact compliance fell **55%→42.5%** and policy accuracy fell **57.5%→42.5%**.
- **Best composition target:** combine subtractive residual-subspace edits, additive zero-damage inserts, and tiny vector/head-only RL. Exps 50–56 already cover vector-only RL, decision-anchor alignment, PGS directions, internal advantage sculpting, activation-cap scaffolding, policy-hint internalization, and boundary-token temperature, so those are not duplicated here.
- **Known dead ends:** unembedding-direction weight edits collapse into always-call behavior at all tested strengths; fp32 does not rescue them; degenerate collinear push-pull is a sign/strength artifact; hard nullspaces and binary masks are too destructive on a 135M model.

Research scan for this batch: arXiv cs.CL/cs.LG papers submitted 2026-11-19 through 2026-11-25 in weight-space programming, activation steering, attention-head intervention, parameter-efficient RL, subspace repair, MoE/architecture expansion, plus frontier/lab blogs including MIT CSAIL weight-space programming and Anthropic feature-steering/introspection writeups. The local `arxiv-mcp-server` is currently misconfigured (`uvx` entry point missing), so the scan used the official arXiv API and full HTML papers directly.

## Priority queue

| Priority | Hypothesis | Main type | Status |
|---:|---|---|---|
| 9.6 | H1 — Exact-zero keyed modular implant bank | surgical insert + attention head + RL | proposed |
| 9.3 | H2 — Operator-discrepancy collateral repair adapter | closed-form subspace repair | proposed |
| 8.9 | H3 — Functional-connectivity-guided head insertion | attention-head modification | proposed |
| 8.7 | H4 — EMS-guided anchor-preserving subspace edits | causal neuron localization + surgical edit | proposed |
| 8.5 | H5 — Output-matched steering-vector compression | activation steering + RL initialization | proposed |

---

## H1 — Exact-zero keyed modular implant bank

**Priority: 9.6/10**

### Novel idea

Generalize Exp 42's single scalar-slider head into a small bank of **new attention heads and MLP neurons inserted inside existing layers**, each behind an input-conditioned gate that is exactly zero off its trigger. Gates should be simple products of ReLU/step affine functions of the residual stream, initialized from measured benign/unsafe/tool-choice activation boundaries rather than left as unconstrained always-on scalars. The insert bank is additive capacity: original weights remain untouched, and off-trigger behavior is bit-exact rather than merely low-drift.

### Why it should work mechanistically

Exp 42 shows zero-init additive heads do not destroy the base model, but its learned scalar becomes globally active and policy accuracy degrades. The fresh weight-writing result shows that new heads/neurons can remain exactly inactive outside narrow residual-stream trigger regions while adding substantial capability. Input-conditioned exact-zero gates directly attack the lab's central problem: false-refusal, true-refusal, tool-choice, and general-capability circuits are entangled, so an always-on additive head helps one behavior while taxing the others.

### Concrete experiment design

1. Start from two checkpoints: unedited Halo SFT and the validated Exp 16-style residual-subspace edit (`raw_rank1`, layers 29/28, dev-gated strength).
2. Insert a bank at layers 27–29:
   - 1–4 extra attention heads after layers 27/28, architecture-compatible with SmolLM2's 64-dimensional heads;
   - 8–32 extra MLP neurons at layers 28/29 with their own gate/up/down projections;
   - every group gated by \(g(x)=\prod_m \mathrm{ReLU}(a_m^\top x+b_m)\) or a small-step-function approximation, with exact zero outside the designed trigger region.
3. Initialize gates from decision-boundary contrasts for benign correct allow, benign false refusal, true unsafe refusal, wrong-tool, and malformed JSON. Initialize head/neuron weights at zero or tiny random values; keep all core weights frozen.
4. Train only insert weights and gate thresholds for 8–24 GRPO-style steps after the surgical edit. Compare:
   - scalar Exp 42 head reproduction;
   - always-on head/neuron bank;
   - exact-zero keyed bank;
   - random-keyed bank;
   - edit-only and RL-only controls.
5. Measure standard policy metrics, gate firing rates by behavior, exact identity/KL on off-task text, `scripts/real_benchmark.py` MMLU/ARC-Easy/HellaSwag temperature, parameter count, wall time, and whether the implant can be exported as a normal checkpoint rather than a session-local hook.

### Expected outcome and falsification criterion

Expected: exact compliance above the current edit-only operating point while preserving unsafe refusal at **≥95%**, benign false refusal materially below the scalar head's 30%, and neutral-or-better real-benchmark temperature. The hypothesis is falsified if keyed gates fire on more than ~1% of off-task/control tokens, if the keyed bank does not beat the scalar and random-keyed controls by at least one exact-compliance task (2.5 pp), or if any gain comes with cold/FREEZING general-benchmark collateral.

### Citation

- [arXiv:2611.28483 — *Don't Train It, Write It: An AI Agent Writes New Capabilities into Language Model Weights*](https://arxiv.org/abs/2611.28483): agent-written gated attention heads/MLP neurons added inside existing layers, exactly zero off-trigger, BBH normalized score 19.95→41.05 on Qwen2.5-1.5B with zero KL on 1,023,000 WikiText tokens.
- Local Exp 42 zero-init slider head and Exp 16 residual-subspace operator.

---

## H2 — Operator-discrepancy collateral repair adapter

**Priority: 9.3/10**

### Novel idea

Treat the surgical edit as a module-level operator perturbation and repair its collateral damage with a **closed-form factorized residual adapter**. Measure same-state output discrepancies between the original and edited modules along the edited model's own trajectory, SVD the discrepancy matrix, remove the target behavior direction, and restore pre-edit computation only in the collateral subspace.

### Why it should work mechanistically

Exp 35's interference cancellation was directionally promising but crude because it treated neurons as independent coefficients and recovered mostly unsafe compliance. Same-state operator discrepancy separates local module damage from upstream activation drift. Its dominant directions are trajectory-conditioned behavioral directions, not raw weight-error noise, so a low-rank closed-form repair can preserve the desired refusal/compliance shift while restoring off-target computation more precisely.

### Concrete experiment design

1. Use the Exp 16 `raw_rank1` edit as the base intervention because it is the strongest non-collapsing operator: **65% exact / 90% unsafe / 0% benign-FR**.
2. For layers 28/29 and modules `mlp.down_proj`, `mlp.up_proj`, and `self_attn.o_proj`, cache inputs \(x_m\) along the edited trajectory on:
   - 32 policy training/dev prompts;
   - 32 paraphrase/control prompts;
   - a small general-capability sample from the real benchmark loader.
3. Compute \(c_m(x)=f_{\rm orig,m}(x)-f_{\rm edit,m}(x)\) at identical states, SVD the matrix of discrepancies, and construct rank \(r\in\{4,8,16,32\}\) repair bases.
4. Explicitly project out the measured target false-refusal/compliance direction before selecting collateral directions; compare against no projection and random rank-matched directions.
5. Arms:
   - unedited baseline;
   - edit-only;
   - edit+full operator repair;
   - edit+target-orthogonal collateral repair;
   - edit+random-subspace repair;
   - Exp 35-style interference-canceling baseline.
6. Evaluate standard 40-task metrics, off-task KL to the unedited model, MMLU/ARC/HellaSwag temperature, adapter memory, and dose-response across repair rank.

### Expected outcome and falsification criterion

Expected: retain within 2.5 pp of the edit-only exact-compliance gain while lifting unsafe refusal from 90% toward 95–100% and reducing off-task KL/benchmark collateral. Falsified if target-orthogonal repair either destroys the target gain, fails to beat random rank-matched repair, or merely reproduces the unsafe-compliance recovery failure mode of Exp 35.

### Citation

- [arXiv:2611.32755 — *ReQOR: Operator-Discrepancy Subspace Repair for Quantization-Conditioned Backdoors in Large Language Models*](https://arxiv.org/abs/2611.32755): same-state operator discrepancies, closed-form factorized residual restoration, rank-efficient repair without gradients.
- Local Exp 35 interference cancellation and Exp 39 operator-sensitivity result.

---

## H3 — Functional-connectivity-guided head insertion

**Priority: 8.9/10**

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

**Priority: 8.7/10**

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

**Priority: 8.5/10**

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

## Near-term research backlog for later batches

Candidates not yet written as full hypotheses: KiRA-style input-gated high-rank adapters ([arXiv:2611.33236](https://arxiv.org/abs/2611.33236)); drift-triggered exact orthogonal lane training ([arXiv:2611.33589](https://arxiv.org/abs/2611.33589)); agent-written closed-form readouts on frozen encoders ([arXiv:2611.33972](https://arxiv.org/abs/2611.33972)); displacement probing as a edit/reference-selection diagnostic ([arXiv:2611.33998](https://arxiv.org/abs/2611.33998)); tool-free LLM quantization-error cancellation ([arXiv:2611.33031](https://arxiv.org/abs/2611.33031)); mixture-of-loops KV memory as a longer-horizon architecture insert ([arXiv:2611.34923](https://arxiv.org/abs/2611.34923)).
