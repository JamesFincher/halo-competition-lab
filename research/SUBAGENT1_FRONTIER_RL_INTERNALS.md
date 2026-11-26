# Subagent 1: Frontier internals × RL research synthesis

**Date:** 2026-11-26  
**Target lab:** SmolLM2-135M refusal-vs-tool-call routing, with DLA/IG surgical edits and restricted GRPO-style RL.

## Verification scope

I used the local `arxiv-mcp-server` extensively: 22 `search_papers` queries, 99 unique candidate records, 26 papers downloaded, and 22 papers explicitly opened through `read_paper`; the highest-value papers were then mined line-by-line from the server cache for equations, hyperparameters, tables, and limitations. Every arXiv identifier cited below was verified through the arXiv MCP server. I did **not** rely on the unverified citation placeholders in the competition README.

The direct literature is surprisingly concentrated outside the branded frontier labs. There is relevant frontier-adjacent evidence — Google’s parameter-efficient RLHF study ([2403.10704](https://arxiv.org/abs/2403.10704)) and Meta-affiliated work on RLVR LoRA geometry ([2606.31813](https://arxiv.org/abs/2606.31813)) — but most 2025–2026 internals×RL hybrids are academic/open-source papers. That is itself an opportunity: the surgical-edit + restricted-RL regime in this lab is not crowded.

## Current lab constraints that matter

| Lab result | Exact compliance | Policy accuracy | Unsafe refusal | Benign false refusal | Mean reward |
|---|---:|---:|---:|---:|---:|
| Starting Halo SFT | 55.0% | 57.5% | 100.0% | 90.0% | 0.775 |
| DLA surgery, zero RL | 80.0% | 92.5% | 90.0% | 0.0% | 0.915 |
| IG-5 surgery, zero RL | **82.5%** | 92.5% | 90.0% | 0.0% | 0.920 |
| Fork-V3 seek/sniff/edit + disjoint RL | 77.5% | 90.0% | 90.0% | 0.0% | 0.910 |
| Random-column control | 60.0% | — | — | — | — |
| Full-model LoRA GRPO | 27.5% | — | — | — | — |

Important implementation facts:

- SmolLM2-135M-Instruct has hidden size 576, intermediate size 1536, 30 layers, 9 query heads, and 3 key/value heads. Thus all-layer per-layer steering vectors cost only \(30\times576=17{,}280\) parameters, about **0.0128%** of a 134.5M-parameter model.
- The best surgical edit touches **32 MLP `down_proj` columns** in layers 28–29, **36,864 weights = 0.02741%** of the model.
- The best disjoint RL arm trains **976,896 parameters = 0.72%** in layers `[0,1,11,19,20,27]`, reaches exact compliance **72.5%**, unsafe refusal **93.3%**, false refusal **0%**, and final KL **0.0023**.
- The failed naive edit-then-RL arm is diagnostic: RL trained the same edited layers while KL-anchored to the pre-edit model, causing destructive interference. Any new method should preserve the edited mechanism, train disjoint parameters, and anchor KL to the edited checkpoint.

## Literature evidence by requested hybrid category

### 1. Steering-vector-guided RL exploration

- **[2505.18706](https://arxiv.org/abs/2505.18706), “Steering LLM Reasoning Through Bias-Only Adaptation”**: RL trains one vector per layer while the base model is frozen. On an 8B model this is roughly **0.0016%** additional parameters; on Qwen2.5-1.5B the steering model reaches average **23.2 vs 22.7** for full RL tuning. The learned vectors amplify interpretable token clusters such as causality, validation, and programming tokens.
- **[2509.06608](https://arxiv.org/abs/2509.06608), “Small Vectors, Big Effects”**: RL-induced steering is mechanistically real but layer-specific. Last-layer steering behaves partly like first-token substitution; replacing the first token with tokens such as “To”/“Step” recovers about **10–11 points**, roughly three quarters of the explicit last-layer gain. Penultimate-layer steering acts mainly through MLP/unembedding rather than attention. Early-layer effects diffuse and become nearly orthogonal to later vectors.
- **[2611.23124](https://arxiv.org/abs/2611.23124), “Get Hint-Augmented GRPO Done Right Through Activation Steering” (ASH)**: hints create useful training groups but a train/inference mismatch. ASH learns the behavioral shift of a hint as a steering vector and injects it at inference. Versus GRPO it improves average pass@8 by **3.84/5.29 points** in-domain and **5.79/3.61** out-of-domain for Qwen3-1.7B/4B; activation steering accounts for **68.9–83.1%** of the Qwen gains and **68.6–80.2%** of the Llama gains.
- **[2602.13035](https://arxiv.org/abs/2602.13035), IntroLLM**: a hierarchical RL temperature controller reads hidden states and decides whether/how much to change token-level temperature. It reaches Avg@8 **30.47/40.91** for Qwen3-1.7B/4B versus **29.41/39.66** for EAD; selective token-level control outperforms prompt-level control (**38.72/50.80 vs 35.91/49.42** average Avg@8). High temperatures concentrate on reasoning-critical tokens.

### 2. Reward shaping from internal activations or probes

- **[2610.40998](https://arxiv.org/abs/2610.40998), “Steering Vectors as a Training Signal in LLM Post-Training”**: constructs a correctness direction \(v^l_{\rm corr}=E_{\rm correct}[\bar h^l]-E_{\rm wrong}[\bar h^l]\), projects hidden states onto it, and adds a small shaped advantage to verifier-wrong responses: \(A=A_{\rm GRPO}+\lambda z_{\rm sv}\mathbf 1[r=0]\). With one direction from 500 samples, middle layer \(L/2+1\), and \(\lambda=0.1\), SV-Explore improves Qwen2.5-1.5B GSM8K from **77.72 to 79.18** and DeepScaleR from **40.22 to 42.30**; Llama-3.2-3B improves GSM8K **81.83→86.71** and MATH-500 transfer **26.60→32.00**.
- **[2606.03234](https://arxiv.org/abs/2606.03234), “Right Makes Might” (Hidden-Align)**: correct rollouts’ last-layer hidden states at the token immediately before `\boxed` have cosine similarity about **0.84**, versus about 0.4 during the reasoning chain. Aligning only correct rollouts at only that anchor token improves average pass@1 over DAPO by **3.83, 6.19, and 5.42 points** on Qwen3-1.7B/4B/14B with zero extra rollout cost.
- **[2611.30627](https://arxiv.org/abs/2611.30627), Attention Process Reward Model (APRM)**: a roughly **46M-parameter** internal process reward model uses hidden states and attention maps; adding group-centered internal advantage with \(\beta\approx0.3\) gives average pass@8 gain **2.88%** and improves GRPO pass@8 **52.62→55.97** on Qwen3-1.7B and **61.68→65.06** on Qwen3-4B. It is too heavy for this lab as written, but the design principle is useful.
- **[2604.01476](https://arxiv.org/abs/2604.01476), “From Rebound to Remedy”**: representation-derived advantage modification is much stronger than generation-time suppression. With multiplicative advantage modification, Qwen3-4B hacking falls **79.0→18.1** while LeetCode rises **11.5→20.1**; Qwen3-8B hacking falls **75.2→16.0** while LeetCode rises **14.0→24.5**. This is the strongest evidence that internal directions should modify training credit, not merely filter outputs.
- **[2603.04069](https://arxiv.org/abs/2603.04069), “Monitoring Emergent Reward Hacking During Generation via Internal Activations”**: SAE/linear internal monitors expose reward-hacking probabilities during generation, often before the externally visible behavior. Useful as diagnostics and stop/go gates even if not used as rewards.

### 3. Surgery-then-RL and sparse post-training

- **[2611.10866](https://arxiv.org/abs/2611.10866), “The Surprising Effectiveness of Deleting Weights in LLM Post-Training”**: zeroing fewer than **0.05%** of weights, with no new values, matches full fine-tuning at the largest tested scales across RLVR, SFT, and on-policy distillation. Under RL, deletion keeps prior-task accuracy near base while FFT/LoRA/random sparse tuning lose up to **5.4 points**. This supports the lab’s core premise that post-training can be selection rather than dense rewriting.
- **[2605.21468](https://arxiv.org/abs/2605.21468), RELEX**: RLVR weight trajectories are effectively rank-1; coefficients evolve nearly linearly with \(R^2>0.98\) for most tensors. Observing only **15–20%** of training matches full RLVR and can extrapolate **10–20×** beyond the observed prefix. Rank-5/rank-10 and nonlinear predictors do not help.
- **[2610.21336](https://arxiv.org/abs/2610.21336), attention-head sparse RL**: initial gradient-norm ranking identifies heads whose sparse training can match or exceed dense attention training. Examples: Qwen3-1.7B math top-50% **36.7±1.0** versus bottom-50% **30.9±1.0**; Mistral-7B instruction following top-50% **69.9** versus bottom **53.5** and all-attention **56.9**. Random-head controls sometimes perform surprisingly well, so matched controls are mandatory.

### 4. Inference-time intervention plus RL

- **[2610.29129](https://arxiv.org/abs/2610.29129), activation capping for realistic reward hacking**: optimized steering vectors reduced reward hacking from **27% to 0%** in one setting while solve rate rose **75%→82%**; activation capping kept hacking at **3% or lower** without significant capability loss. Crucially, this paper uses intervention at inference, not as a training scaffold.
- **[2607.27574](https://arxiv.org/abs/2607.27574), Policy Gradient Steering (PGS)**: computes a removable activation vector from return-weighted action-score gradients. In gridworlds, PGS shifts preferred-route probability from **0.43±0.06 to 0.82±0.07**, while CAA stays at **0.41±0.07**. In chess-style composition, **1,152** steering scalars give composed gain **0.804±0.014**, close to fine-tuning **0.807±0.020**. PGS is direct evidence that RL credit assignment can construct better activation interventions than contrastive means.
- **[2609.34344](https://arxiv.org/abs/2609.34344), “Learning to Steer, Steering to See”**: a single input-invariant vector per controlled layer recovers **over 85%** of full-RL gains on most tasks. Effective steering directions live mainly in the low-variance complement of the activation principal subspace; principal-subspace intrusion predicts collapse, and Alpha-Stabler stabilizes training for **2,000 steps** with little overhead.

### 5. RL constrained to small parameter subspaces

- **[2609.18587](https://arxiv.org/abs/2609.18587), label-free bias-only TTRL**: about **100K** trainable bias/steering parameters, roughly **76,000× fewer** than full TTRL, reaches **76.67%** on MATH-500 for Qwen2.5-7B and transfers to 4,500 disjoint MATH problems (**46.3→70.9**). Compute is **4.45–4.95 GH200-hours** versus **79.24–82.12** for full fine-tuning. Gradient energy predicts trainability with Spearman \(\rho=0.917, p=0.0013\).
- **[2608.11674](https://arxiv.org/abs/2608.11674), GCPO**: constrains RL updates to the bilateral orthogonal complement of dominant pretrained singular subspaces. It improves over base models by up to **27.69 points** and over the strongest baseline by up to **2.37**, while reducing cross-task degradation, response-length inflation, and policy-entropy instability.
- **[2606.31813](https://arxiv.org/abs/2606.31813), geometry-preserving orthonormal LoRA initialization** (Meta-affiliated coauthor): SVD-aligned PiSSA/MiLoRA initializations can be unstable under RLVR even when strong in SFT; orthonormal initialization minimizes the theoretical LoRA/full-RL gap and empirically stabilizes RLVR.
- **[2403.10704](https://arxiv.org/abs/2403.10704), PE-RLHF** (Google ecosystem): LoRA-based reward modeling and RL match standard RLHF across six datasets while reducing reward-model training time up to **90%**, RL training time up to **30%**, and memory up to **50%/27%** respectively. This is efficiency evidence, not internals evidence, but it validates restricted RL as a serious training regime.
- **[2512.23260](https://arxiv.org/abs/2512.23260), SAILS**: SAE decoder directions initialize a safety-relevant low-rank subspace, reaching up to **99.6% safety** on Gemma-2-9B and exceeding full fine-tuning by **7.4 points**. It is SFT-style rather than RL, but the interpretable-subspace initialization idea transfers naturally to restricted RL.

## Ranked experiment ideas

Scores are ordinal design judgments, not measured results: **N** novelty, **G** expected performance gain, **C** implementation cost (higher = more expensive). The ranking weighs novelty × expected gain ÷ implementation cost, then adjusts for fit to the lab’s exact bottleneck.

| Rank | Experiment | N | G | C | Primary reason |
|---:|---|---:|---:|---:|---|
| 1 | Surgically anchored bias-only steering-vector RL | 8 | 9 | 2 | Cheapest way to harden a zero-RL edit without overwriting it |
| 2 | Decision-anchor Hidden-Align for correct routings | 8 | 8 | 2 | Directly stabilizes the internal decision state with almost no cost |
| 3 | PGS-credit direction discovery for the next surgical edit | 9 | 7 | 2 | Uses RL credit assignment to find better directions than contrastive means |
| 4 | Dual-sided internal advantage sculpting | 9 | 8 | 3 | Attacks zero-gradient groups and over-refusal simultaneously |
| 5 | Activation-capped rollout scaffolding, annealed into RL | 10 | 8 | 4 | Converts inference-time control into trainable behavior |
| 6 | ASH-style policy-hint internalization | 9 | 7 | 4 | Uses privileged hints during training without needing them at deployment |
| 7 | Hidden-state-conditioned boundary-token temperature | 8 | 6 | 4 | Makes exploration selective instead of uniformly entropic |
| 8 | Attention-head sparse RL complementary to MLP surgery | 7 | 6 | 3 | Clean mechanistic complement to existing `down_proj` edits |
| 9 | Principal-subspace guardrails for edit-safe RL | 8 | 6 | 4 | Prevents destructive drift rather than directly adding capability |
| 10 | Rank-1 extrapolation of restricted-RL trajectories | 10 | 7 | 5 | Potentially large compute win and a novel measurement of surgery/RL alignment |

### 1. Surgically anchored bias-only steering-vector RL

**Citations:** [2505.18706](https://arxiv.org/abs/2505.18706), [2609.18587](https://arxiv.org/abs/2609.18587), [2509.06608](https://arxiv.org/abs/2509.06608).

**Core mechanism.** Freeze the entire edited model, including the surgical `down_proj` columns, and use GRPO/RLVR to train only additive per-layer steering vectors in complementary layers. This converts the already-successful zero-RL edit into a hardened policy without allowing RL to rewrite the mechanism that created the behavior.

**Why untapped.** Bias-only/vector RL has been demonstrated for math reasoning and test-time majority-vote pseudo-labels, but not as the second stage of an explicit mechanistic surgery pipeline, and not for refusal-vs-tool-call policy routing. The closest papers also do not exploit the lab’s disjoint-layer anchoring trick.

**Implementation sketch (4 lines).**
1. Start from the best IG-5/DLA edited checkpoint; freeze all weights and set the KL reference to this edited checkpoint.
2. Insert trainable zero vectors at either all 30 layers (**17,280 params, 0.0128%**) or only the six disjoint RL layers (**3,456 params, 0.0026%**).
3. Train with the existing exact-compliance reward, group size 4–8, 10–30 steps, KL to the edited reference, and no weight decay on vectors.
4. Compare all-layer vectors, disjoint-layer vectors, random-layer controls, and the existing 976,896-parameter LoRA arm; require cap-off exact compliance, unsafe refusal, false refusal, and KL.

**Expected result / cost / risk.** This is the best first experiment because the surgical edit already places the model near the correct behavior; vectors only need to reduce residual boundary errors. A realistic target is exact compliance **85%+** while preserving unsafe refusal **≥90%** and false refusal near zero. Cost is very low. Main risk: input-invariant vectors may be too weak for token-conditional routing, but [2609.34344](https://arxiv.org/abs/2609.34344) suggests low-to-middle layers tolerate this approximation better than output-adjacent layers.

### 2. Decision-anchor Hidden-Align for correct routings

**Citations:** [2606.03234](https://arxiv.org/abs/2606.03234); supporting internal-process evidence from [2611.30627](https://arxiv.org/abs/2611.30627).

**Core mechanism.** Identify the token position immediately before the decision-critical JSON field (for example before the action/tool token) as an anchor. During RL, add an auxiliary loss that increases pairwise cosine similarity among hidden states of **correct compliant rollouts only** at that anchor.

**Why untapped.** Hidden-Align was demonstrated for mathematical answers immediately before `\boxed`, not for agentic tool routing or safety boundaries. The lab’s strict schema is an unusually good fit because the anchor token can be identified exactly, unlike long free-form reasoning.

**Implementation sketch (4 lines).**
1. Log hidden states at layers 20/25/29 for the token immediately before `"action"` or its tokenizer-equivalent boundary.
2. Within each prompt group, select only exact-compliant rollouts; add pairwise cosine alignment loss at that anchor.
3. Sweep \(\lambda\in\{10^{-4},10^{-3},10^{-2}\}\) and compare last-layer versus middle-layer alignment.
4. Keep the surgical edit frozen and train either bias vectors (idea 1) or the existing disjoint layers; evaluate exact compliance, unsafe refusal, false refusal, and representation collapse.

**Expected result / cost / risk.** In math, this exact class of objective adds **3.83–6.19 points**; a reasonable lab target is **+3–6 exact-compliance points** and reduced seed variance. Cost is very low because it uses states already computed in the forward pass. The key risk is aligning the wrong behavior: never include false-refusal or unsafe-pass rollouts in the positive set.

### 3. PGS-credit direction discovery for the next surgical edit

**Citations:** [2607.27574](https://arxiv.org/abs/2607.27574), with mechanistic context from [2509.06608](https://arxiv.org/abs/2509.06608).

**Core mechanism.** Replace or augment contrastive mean-direction sniffing with a return-weighted policy-gradient direction: \(v_l=E[A_t\nabla_{h_l}\log\pi(a_t\mid o_t)]\). This assigns credit to activation components that actually control the action token, rather than merely separating prompt classes.

**Why untapped.** PGS has been demonstrated in gridworld, chess-like compositionality, and control settings, not in LLM safety routing or as a discovery method for rank-1 weight surgery. The lab is positioned to test whether PGS directions outperform DLA/IG directions causally.

**Implementation sketch (4 lines).**
1. Sample 4–8 rollouts per hard benign/unsafe boundary prompt from the edited checkpoint and compute exact-compliance advantages.
2. At the action-decision token, compute \(v_l\) for layers 20, 28, and 29; normalize and optionally Fisher-calibrate by activation variance.
3. Compare cosine alignment and causal effect of PGS, DLA, IG, canonical-contrast, and random directions using the existing probe/edit harness.
4. Use the best accepted direction either as a new surgical projection or as the initialization for bias-only RL.

**Expected result / cost / risk.** This may not directly add RL capability, but it can raise the zero-RL ceiling above **82.5%** or reduce the edit strength needed to reach it. Cost is low: mostly forward/backward instrumentation on a small prompt set. Risk: return-weighted gradients may be noisy at 135M scale, so use mean over groups and compare against random directions.

### 4. Dual-sided internal advantage sculpting

**Citations:** [2610.40998](https://arxiv.org/abs/2610.40998), [2604.01476](https://arxiv.org/abs/2604.01476), [2611.30627](https://arxiv.org/abs/2611.30627).

**Core mechanism.** Use two internal directions during RL: a positive compliance/correct-routing direction that rescues verifier-wrong but internally promising rollouts, and a negative over-refusal/shortcut direction that discounts internally over-defensive behavior before group normalization.

**Why untapped.** [2610.40998](https://arxiv.org/abs/2610.40998) uses correctness directions to rescue incorrect math rollouts; [2604.01476](https://arxiv.org/abs/2604.01476) uses representation directions to penalize reward hacking. Nobody has combined the two into a safety-utility tradeoff controller for refusal-vs-tool-call routing, especially with surgical edits defining the initial directions.

**Implementation sketch (4 lines).**
1. Cache hidden states at the decision token for correct, false-refusal, unsafe-pass, and true-refusal examples.
2. Build \(v_{\rm compliance}\) from correct-vs-incorrect means and \(v_{\rm overrefusal}\) from false-refusal-vs-benign-allow means; orthogonalize the latter against a true-unsafe direction.
3. Before group normalization, add \(\lambda_+ z_{\rm compliance}\) only to verifier-wrong rollouts and subtract \(\lambda_- z_{\rm overrefusal}\) only when its z-score is positive.
4. Sweep \((\lambda_+,\lambda_-)\) around `(0.1,0.1)`, `(0.1,0.3)`, `(0.3,0.1)`; train only disjoint/vector parameters and track dead-group rate.

**Expected result / cost / risk.** This directly attacks the two RL failure modes visible in the lab: zero-signal groups and over-refusal collapse. Expected gain is **+3–8 exact-compliance points** if direction quality is adequate. Cost is moderate because it requires activation caching and advantage surgery. The main risk is rewarding internal confidence rather than correctness; keep the external verifier dominant.

### 5. Activation-capped rollout scaffolding, annealed into RL

**Citations:** [2610.29129](https://arxiv.org/abs/2610.29129), [2604.01476](https://arxiv.org/abs/2604.01476).

**Core mechanism.** Use activation capping or flooring along a false-refusal/compliance direction only during early RL rollouts. Treat it as a temporary scaffold that keeps the policy in the desirable region while weight updates internalize the behavior, then anneal the intervention to zero.

**Why untapped.** The strongest activation-capping results are inference-time controls. The literature has not answered whether temporary inference-time interventions can bootstrap permanent RL behavior without becoming a permanent crutch. This lab can answer that cleanly because evaluation is cheap and exact.

**Implementation sketch (4 lines).**
1. Derive a false-refusal direction from the surgical edit or validation contrasts; cap its positive component at the 90th percentile of benign-allow activations in layers 28–29.
2. Run RL with the cap active on 100% of rollouts for 2–5 steps, then linearly reduce cap strength or application probability to zero over the next 5–10 steps.
3. Train disjoint vectors/adapters only; evaluate with the cap off.
4. Compare permanent cap, cap-only inference, RL-only, cap-annealed RL, and no intervention.

**Expected result / cost / risk.** This is the highest-novelty experiment. If scaffolding works, it could combine the zero-RL edit’s **82.5%** ceiling with RL’s unsafe-refusal improvement and exceed **85% exact compliance**. Cost is moderate because generation hooks are invasive. Risk: the policy may learn to rely on the scaffold; cap-off evaluation must be the primary metric.

### 6. ASH-style policy-hint internalization

**Citation:** [2611.23124](https://arxiv.org/abs/2611.23124).

**Core mechanism.** Construct privileged “hinted” training prompts containing a concise policy reminder or canonical exemplar, and unhinted deployment prompts. Train a steering vector that reproduces the hinted behavior shift, then use it during RL rollouts and deployment.

**Why untapped.** ASH was built for competition-math hints, not explicit agent-policy boundaries. The lab already has canonical token-aligned contrast pairs, making it unusually easy to construct hint/no-hint pairs without ambiguous natural language.

**Implementation sketch (4 lines).**
1. For each training prompt, create a hinted variant containing the exact policy rule and one canonical correct JSON example.
2. Measure the activation difference between hinted and unhinted forward passes at layers 20/25/28/29, excluding the hint tokens themselves from deployment-position alignment.
3. Train a vector or low-rank vector set to mimic the hint effect on unhinted prompts, jointly with RL on mixed hinted/plain prompts.
4. Evaluate unhinted deployment performance with the vector active; ablate vector-only, RL-only, ASH-style joint, and hinted-prompt upper bound.

**Expected result / cost / risk.** Expected gain is mainly generalization to unseen surface forms rather than memorizing training prompts: target **+3–6 points** over the current RL loop. Cost is medium due prompt construction and activation alignment. Risk: a 135M model may encode the hint as topic rather than rule content, so canonical and random-hint controls are necessary.

### 7. Hidden-state-conditioned boundary-token temperature

**Citation:** [2602.13035](https://arxiv.org/abs/2602.13035).

**Core mechanism.** Train a small controller that reads the hidden state at boundary tokens and decides whether to increase temperature for the next token. Exploration becomes selective: high entropy only at action/tool/refusal decision points, low entropy in schema tokens.

**Why untapped.** IntroLLM targets long math reasoning. It has not been applied to ultra-short structured action outputs, where most tokens are deterministic schema and only a handful carry behavioral uncertainty. Nor has it been combined with a surgically induced decision direction.

**Implementation sketch (4 lines).**
1. Identify decision positions for policy class, action, and tool name; freeze schema punctuation at low temperature.
2. Add a tiny head (hidden size 576 → 64 → 2 or → Beta parameters) that predicts whether/how much to adjust temperature at each decision token.
3. Coordinate-ascent train the controller and the token policy/vectors with exact-compliance reward; compare always-update, prompt-level, and selective-token control.
4. Measure group entropy, zero-variance-group fraction, pass@4, exact compliance, and unsafe-refusal retention.

**Expected result / cost / risk.** Expected gain is indirect but important: fewer zero-gradient groups and better exploration, perhaps **+2–5 points** when paired with ideas 1 or 4. Cost is medium-high because generation instrumentation is more complex. Risk: at 135M scale the controller may exploit temperature noise rather than learn meaningful uncertainty.

### 8. Attention-head sparse RL complementary to MLP surgery

**Citations:** [2610.21336](https://arxiv.org/abs/2610.21336), [2611.10866](https://arxiv.org/abs/2611.10866).

**Core mechanism.** Keep the MLP surgical edit frozen and perform RL only in a small set of gradient-ranked attention heads. This separates mechanism discovery: MLPs store the surgical refusal/tool direction, while attention learns context-dependent routing.

**Why untapped.** Head-ranked sparse RL has not been combined with explicit MLP surgery. The lab’s existing restriction is layer-disjoint but mechanism-coarse; attention-head partitioning would make the complement mechanistically specific.

**Implementation sketch (4 lines).**
1. Run one small GRPO batch on validation failures and rank heads by initial gradient norm on action-decision tokens.
2. Select top 10–25% of heads; update only their q/output slices while freezing all MLPs, embeddings, and edited layers.
3. Compare top-ranked, bottom-ranked, random-ranked, and all-attention controls at the same parameter budget.
4. Evaluate task metrics plus cross-task retention and KL; inspect whether selected heads attend to policy-relevant entities or tool names.

**Expected result / cost / risk.** This is a medium-gain experiment: perhaps **+2–5 points** over the current disjoint RL arm if attention is the missing contextual component. Cost is moderate because PyTorch parameter grouping by head is fiddly. Random sparse controls are essential because [2610.21336](https://arxiv.org/abs/2610.21336) found random partitions can occasionally beat top-ranked ones.

### 9. Principal-subspace guardrails for edit-safe RL

**Citations:** [2608.11674](https://arxiv.org/abs/2608.11674), [2609.34344](https://arxiv.org/abs/2609.34344), [2606.31813](https://arxiv.org/abs/2606.31813).

**Core mechanism.** Measure whether RL updates intrude into dominant pretrained weight/activation subspaces, and project updates into their orthogonal complement. This is a geometric version of the lab’s current “do not overwrite the surgical edit” constraint.

**Why untapped.** Current lab restrictions specify *which layers/parameters* may move, not *which directions within those parameters* may move. GCPO and Alpha-Stabler suggest the latter predicts collapse, forgetting, and length/reward hacking better than magnitude alone.

**Implementation sketch (4 lines).**
1. Precompute top-\(k\) left/right singular vectors for each trainable matrix in the disjoint layers or vector-adapter parameterization.
2. During each update, measure principal-subspace overlap and plot it against exact compliance/unsafe-refusal degradation.
3. For the constrained arm, project gradients or effective updates into the bilateral orthogonal complement.
4. Compare KL-only, layer-disjoint-only, orthonormal-LoRA, and geometrically constrained RL; track prior-task retention and false refusal.

**Expected result / cost / risk.** This may not raise the ceiling immediately, but it should reduce destructive interference and make other methods safer. In GCPO, geometric constraints improve stability and cross-task retention while slightly outperforming strong baselines. Cost is medium-high due SVD/projection machinery. Risk: a 135M model may need some principal-direction movement because its capacity is small.

### 10. Rank-1 extrapolation of restricted-RL trajectories

**Citations:** [2605.21468](https://arxiv.org/abs/2605.21468), [2611.10866](https://arxiv.org/abs/2611.10866).

**Core mechanism.** Treat the restricted-RL update trajectory as a low-rank time series. Observe the first 10–20 steps, fit rank-1 structure and nearly linear coefficients, extrapolate to a later step, and use the surgical/DLA probe suite as a safety gate.

**Why untapped.** RELEX compresses ordinary RLVR trajectories, but nobody has tested whether surgically anchored, layer-disjoint RL is even lower-rank or whether extrapolated updates preserve a manually inserted causal mechanism. A positive result would be scientifically interesting even if not immediately deployed.

**Implementation sketch (4 lines).**
1. Train the best restricted arm for 20 short steps, checkpointing every step.
2. For each trainable tensor, compute \(\Delta W_t\), extract its leading rank-1 component, and fit coefficient magnitude versus step.
3. Extrapolate to steps 40/60/100 and compare against actually continued RL at matched compute.
4. Accept extrapolation only if exact compliance, unsafe refusal, false refusal, KL, and DLA probe direction all pass gates.

**Expected result / cost / risk.** Potential compute savings are large — RELEX matches full RL after observing only **15–20%** of training — but the lab’s runs are already short, so the primary value may be diagnostic rather than practical. Cost is medium because checkpoint/SVD infrastructure is straightforward but validation is extensive. Risk: the current task may have abrupt policy transitions that violate linear coefficient evolution.

## Recommended execution order

1. **Run idea 1 first.** It is cheap, directly aligned with the lab’s surgical advantage, and creates the vector-training harness needed by several later ideas.
2. **Add idea 2 as an auxiliary objective** once idea 1 is stable. It should require only a small loss addition.
3. **Use idea 3 as a diagnostic parallel branch** to determine whether PGS directions beat DLA/IG directions before investing in new surgery.
4. **Then choose one high-novelty branch:** idea 5 if the goal is maximum behavioral gain, or idea 4 if RL currently suffers from dead groups/over-refusal.
5. Treat ideas 7–10 as second-wave mechanisms after the first branch establishes whether steering-vector RL helps at all.

## Quantitative success gates for the lab

A hybrid method should not be called successful merely because mean reward rises. Recommended acceptance gates on the same 40-task held-out evaluation:

- **Primary:** exact compliance **≥85%** cap-off, exceeding the zero-RL IG-5 result of **82.5%**.
- **Safety floor:** unsafe refusal **≥95%** without benign false refusal rising above **5%**.
- **Stability:** final KL to the edited reference **≤0.005** and no monotonic degradation across the last three checkpoints.
- **Mechanistic preservation:** the surgical direction/probe remains detectable and materially unchanged after RL; random-direction and random-layer controls must be significantly worse.
- **Compute/parameter discipline:** report trainable parameter count, wall time, number of rollouts, and whether the method uses inference-time intervention at evaluation.

## Bottom line

The most promising untapped synthesis is not “more RL.” It is **surgical anchoring plus extremely small steering-vector updates, with internal-state objectives controlling credit assignment and exploration**. The lab already has the two prerequisites most papers lack: a causal zero-RL intervention that works, and a restricted-RL harness that can isolate where updates go. The highest-value novel contribution would be showing that a mechanistic edit can serve as a permanent anchor while RL optimizes only a tiny activation-level residual manifold — potentially pushing exact compliance beyond the current **82.5% zero-RL ceiling** without sacrificing the hard-won refusal boundary.
