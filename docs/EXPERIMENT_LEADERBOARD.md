# Experiment Leaderboard — Quality-Controlled Rubric Grades

Last updated: **2026-11-26 15:49 UTC** by the tracker/post-processor.

Composite = `0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R` (see `research/RUBRIC.md`). Machine-generated scores and the integrity scan are in `research/AUTOGRADED_SCORES.json` and `research/RESULT_INTEGRITY_REPORT.json`.

**Canonical baseline:** Halo LoRA SFT checkpoint-36 on the 40-task held-out eval = **55.0% exact / 100.0% unsafe refusal / 90.0% benign false refusal** (also 77.5% action accuracy, 57.5% policy accuracy). Positive Δ benign-FR is worse; negative Δ benign-FR is better only when unsafe refusal remains high.

**Comparator policy:** exact random/uniform/reference/control arms are excluded from technique scoring. Scientifically important comparators remain in raw result JSONs and analysis, but they do not win leaderboard rows. This poll added exclusions for Exp52's DLA reference, Exp55's zero-vector/plain RL comparator, Exp66's validated-edit reference, and Exp101's standard-unrounded comparators.

## Interpretable technique ranking

The top CPU-era probe row is still not exactly GPU-replicated. The 12:59 sparse-edit harvest now establishes a much stronger validated-edit operating point: strength 0.9 at 16 columns/layer reaches **80.0 / 93.3 / 0** in exp129 and is exactly reproduced as columns16 in exp130. Seed replication is still required, but this is no longer a single isolated edit measurement. However, Exp52 now provides adjacent positive GPU evidence for the DLA/unembed direction on the validated top-neuron edit mask: its reference arm is **72.5 / 93.3 / 0**, while PGS-estimated directions are nearly anti-aligned with DLA and fail. This is not yet the missing exact Exp17 replication, but it substantially weakens the earlier hypothesis that the CPU-era DLA direction was purely an artifact.

| Rank | Experiment / best arm | Regime & validity | Exact 55→after (Δ) | Unsafe 100→after (Δ) | Benign-FR 90→after (Δ) | Score | Grade | Interpretation |
|---:|---|---|---:|---:|---:|---:|:---:|---|
| 1 | **17 / `probe4_integrated_grads_5`** | legacy CPU-era; exact GPU replication still outstanding | 55→82.5 (**+27.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **9.05** | **S** | Historical best zero-grad probe (IG-5). |
| 2 | **102 / `validated_edit_manual_probe`** | **GPU-valid inference-time forced-decoding scaffold; not a weight-level model improvement** | 55→95 (**+40**) | 100→100 (**0**) | 90→0 (**-90**) | **8.30** | **A** | Train-only probes perfectly read decision variables; proves representation/readout bottleneck, not internal capability. |
| 3 | **1 / `arm2_localized_grpo_only`** | GPU-valid but baseline-shifted/overwrite-restored | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **8.10** | **A** | Targeted localized GRPO; directionally supported by exp80. |
| 4 | **144 / `h22_seed7/42/123`** | **GPU-valid 3-seed exported-artifact replication; layer specificity falsified by random control** | 55→82.5 (**+27.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.90** | **A** | Strongest replicated raw stack: ultra-sparse edit + post-edit MLP LoRA, but only +1 task over edit alone and random layers match. |
| 5 | **85 / `es_final_center`** | **GPU-valid n=40; selection-noise caution** | 55→82.5 (**+27.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.90** | **A** | Strongest final metric, but replication is mandatory. |
| 6 | **129 / `strength_0p9` + 130 / `columns_16`** | **GPU-valid deterministic configuration replication; seed replication still needed** | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.90** | **A** | Ultra-sparse 16-column/layer edit is the new validated surgical operating point. |
| 7 | **11 / `v3_closed_loop_sniff_edit_rl`** | legacy CPU-era | 55→77.5 (**+22.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.85** | **A** | Strongest historical closed-loop composition. |
| 8 | **80 / `rl_only`** | **GPU-valid canonical-baseline replication** | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→10 (**-80**) | **7.60** | **A** | Major replication of targeted RL alone. |
| 9 | **57 / `halo_regex_complementary_mlp_r16`** | **GPU-valid positive transfer path** | 55→75 (**+20**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.60** | **A** | Post-edit-anchored complementary MLP-only LoRA. |
| 10 | **63 / `strength_0p90_cols32`** | **GPU-valid dose/width replica** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.35** | **B** | Confirms narrow 32-column canonical-strength edit. |
| 11 | **86 / `oscillating_decay`** | **GPU-valid positive** | 55→72.5 (**+17.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.35** | **B** | Polarity cycling prevents destructive fixation. |
| 12 | **98 / `columns_32`** | **GPU-valid dose-response** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.15** | **B** | Narrower edit beats 64/128 columns. |
| 13 | **99 / `l29_l28_split_32_each`** | **GPU-valid allocation result** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.15** | **B** | Split allocation reproduces the narrow-edit operating point. |
| 14 | **145/146 / cache-corrected L20 slider** | **cache-corrected GPU diagnostic; same-run baseline shifted to 52.5/93.3/90** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.15** | **B** | Important positive inserted-head result; supersedes legacy-cache L20 collapse but needs dedicated replication. |
| 15 | **96 / `freeze_then_merge_8x4`** | **legacy-cache confounded** | 55→70 (**+15**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.90** | **B** | Promising schedule measurement, not clean causal evidence until rerun cache-corrected. |
| 16 | **64 / `l29_l28_plus_l27`** | **GPU-valid layer ablation** | 55→67.5 (**+12.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.85** | **B** | L29 dominates; L27 adds a small interaction. |
| 17 | **53 / `complianceheavy_0.3_0.1`** | GPU-valid but safety-costly | 55→72.5 (**+17.5**) | 100→83.3 (**-16.7**) | 90→0 (**-90**) | **6.75** | **B** | Exactness gain has a large unsafe-refusal cost. |
| 18 | **16 / `raw_rank1`** | GPU-validated positive operator | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.70** | **B** | Prior validated surgical operator, now superseded by sparse/RL variants. |
| 19 | **100 / `targeted_lora_complement_mlp_only`** | GPU-valid transfer comparison | 55→67.5 (**+12.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.65** | **B** | Targeted LoRA beats partial full-FT from the same anchor. |
| 20 | **60 / `inverter_then_complementary_lora`** | GPU-valid but no gain over validated-edit anchor | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.60** | **B** | Reaches the standard post-edit operating point; does not exceed it. |
| 21 | **65 / `insert_after_l27`** | **legacy-cache confounded** | 55→65 (**+10**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **6.60** | **B** | Do not interpret depth causally until cache-corrected replication. |
| 22 | **95 / `strength_0p9`** | GPU-valid dose-response | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.40** | **C** | Canonical strength validated; lower strength trades precision for exactness. |
## 12:59 ultra-sparse edit and capability-objective results

| Experiment / arm | Exact 55→ | Unsafe 100→ | Benign-FR 90→ | Benchmark/probe evidence | Interpretation |
|---|---:|---:|---:|---|---|
| **129 / `strength_0p9` (16 cols/layer)** | **80.0 (+25)** | **93.3 (-6.7)** | **0 (-90)** | MMLU/ARC/HellaSwag exactly unchanged at 26/38/32; probe changes small | New best validated sparse-edit point; strength response is coherent. |
| **130 / `columns_16`** | **80.0 (+25)** | **93.3 (-6.7)** | **0 (-90)** | Same unchanged benchmark triple; exact same metrics/probe as exp129 strength0.9 | Independent deterministic replication of the same canonical sparse configuration; still needs seed replication. |
| 129 `strength_0p7` | 77.5 | 93.3 | 20 | unchanged benchmarks | Slightly lower dose preserves exactness but leaves benign-FR. |
| 130 `columns_8` | 77.5 | 93.3 | 0 | unchanged benchmarks | Even sparser edit nearly matches columns16. |
| 132 / `hybrid_capability_primary` (**superseded: legacy-cache confounded**) | **72.5 (+17.5)** | **93.3 (-6.7)** | **0 (-90)** | coarse MMLU 26→30, ARC 38→40; normalized continuation MMLU/HellaSwag decline; capability val loss 3.0886→3.0336 | **Do not use as a positive result.** Runtime provenance implies launch before the cache fix; exp139 is the corrected replication and does not reproduce the strong behavioral point. |
| 132 `capability_only` | 67.5 | 100.0 | 80 | capability val loss improves to 2.8203; coarse ARC +2 but normalized MMLU falls | Capability validation loss moves, but exp139 shows cache-corrected behavior is not the claimed positive operating point. |
| 132 `behavior_only` | 12.5 | 3.3 | 0 | coarse MMLU rises to 40 despite behavioral collapse | Legacy-cache-confounded destructive artifact; do not interpret MMLU increase as capability gain. |
| 91 `layer_27` | 60.0 | 100.0 | 50 | unchanged benchmarks | **Superseded/confounded:** run used legacy cached generation; do not interpret the L27/L20 ordering causally after exp146. |


## 13:33 blending/cancellation family results

| Experiment / mechanism | Exact 55→ | Unsafe 100→ | Benign-FR 90→ | Interpretation |
|---|---:|---:|---:|---|
| **136 / learned sparse-context cancellation** | **75.0 (+20)** | **93.3 (-6.7)** | **0 (-90)** | Strong raw score but **worse than its own edit-only reference (80/93.3/0)**; not a performance win. Oracle cancellation reaches 100% unsafe refusal but exactness falls to 65. |
| 138 / negative subtractive gate | 57.5 | 100.0 | 100.0 | Safe but over-refuses; sign inversion works mechanistically, not beneficially. |
| 135 / learned capability-behavior blend | 40.0 | 100.0 | 90.0 | Exactness-negative despite capability-loss reduction; no blend synergy. |
| 137 / cross-layer cascade | 37.5 | 100.0 | 80.0 | Safer than parallel control but still exactness-negative. |
| 134 / routed width mixture | 25.0 | 96.7 | 100.0 | Routing fails; single-width reference head was better but excluded as comparator. |

**Key control finding:** exp136 oracle cancellation demonstrates a controllable safety/exactness frontier, but the learned gate fires too rarely (~1.1–1.6%) to recover unsafe refusal. The dev sweep also gives identical metrics for fractions 0.25/0.50/0.75, a dose-insensitivity/implementation warning requiring replication before causal interpretation.


## 14:19 cache-mode correction and diagnostic harvest

**Critical correction:** exp146 decisively shows that the pre-12:41 slider generation path was cache-confounded. With legacy cached decoding, the inserted head sees token-local rather than full-context attention during generation; the L20 head collapses to **0 / 0 / 0** while coarse benchmarks misleadingly rise. With `use_cache=False`, the same L20 insertion reaches **72.5 / 90 / 0** from a same-run cache-corrected baseline of **52.5 / 93.3 / 90**. Therefore all pre-fix slider depth/width/count/seed/composition conclusions are legacy-cache measurements, not clean causal inserted-head evidence.

| Experiment / arm | Result | Interpretation |
|---|---:|---|
| **146 `layer20_cache_corrected`** | **72.5 / 90 / 0** | Positive L20 inserted-head result; distribution diagnostics remain near baseline. |
| 146 `layer20_legacy_cache` | 0 / 0 / 0; coarse benchmark 34/40/34 | Degenerate behavior plus benchmark inversion; flattened logits/length bias explain the false capability signal. |
| 145 `layer_20` | **72.5 / 90 / 0** | Same cache-corrected positive result in the smaller diagnostic run. |
| 82 rank-1 trajectory extrapolation | t8 **15 / 0 / 0**, worse farther | Negative, but validation itself is weak: predicted step-8 norms are 1.21–1.33× actual and relative vector errors are ~0.50–0.56 despite cosine 0.90–0.93. |
| 87 false-margin steering | all arms **55 / 100 / 90** | Stable factorization directions exist, but steering is a no-op at tested dose. |
| 103 failure-neuron suppression | 30→15 exact as dose rises | Destructive; no repair evidence. |
| 105 slider LR bank | low LR no-op, high LR collapse | Pre-cache-fix confounded; not causal slider-LR evidence. |
| 143 contrastive-choice objective | **45 / 93.3 / 90**, validation accuracy unchanged | Actual contrastive objective does not improve contrastive validation; correct-answer reference improves validation but damages behavior/safety. |


## 14:34 H22 and capability-ladder results

| Experiment / arm | Result | Interpretation |
|---|---:|---|
| **144 / H22 seeds 7/42/123** | **82.5 / 93.3 / 0** each | Best replicated raw exported-artifact stack. Adapter and merged-export evals agree; benchmarks unchanged. However this is only one task (+2.5pp) above the canonical sparse edit at n=40. |
| 144 / canonical sparse edit replication | **80.0 / 93.3 / 0** | Third deterministic confirmation of the ultra-sparse 16-column operating point. |
| 144 / random-layer MLP control | **82.5 / 93.3 / 0** | Matches H22 exactly; layer-specific complementary transfer is falsified. LoRA without edit is 52.5/100/90, vector-only post-edit is 57.5/100/90, so the sparse edit remains the necessary driver. |
| **139 / cache-corrected hybrid capability-primary** | **55.0 / 100.0 / 40.0** | Corrected replication of exp132. It preserves unsafe refusal and halves benign false refusal, but does not increase exactness and does not yield reliable benchmark/continuation transfer. |
| 140 / answer-only weighting | validation loss improves, behavior **50.0 / 63.3 / 0** | Answer alignment improves answer-only validation loss but sacrifices unsafe refusal; no benchmark transfer. |
| 141 / benchmark-direct format | **37.5 / 63.3 / 0** | Exact continuation-format matching does not rescue transfer and is behaviorally worse than the lettered reference. |
| 142 / capability budget 1x/4x/16x | **37.5 / 40.0 / 20.0 exact** | More optimization does not solve transfer; validation loss is nonmonotonic and behavior degrades. |
| 147 / additive or learned capability-behavior union | best component **50.0 / 93.3 / 90**; blends 40–45 exact | Capability/behavior heads interfere rather than compose additively. |

**Critical comparability warning:** exp140–142/147 evaluate behavior with `max_new_tokens=48` and have a shifted same-run baseline **52.5 / 93.3 / 90**. Exp139's metadata says 48 but its behavioral eval omits the argument and therefore uses the library default 80; exp144 uses 64 and retains canonical-baseline behavior. Use same-run baselines for mechanism comparisons until the eval harness is standardized.


## 15:14 closed-form readout and scorer-floor diagnostics

| Experiment / measurement | Result | Interpretation |
|---|---:|---|
| **102 / train-only probe-guided forced decoding** | **95.0 / 100.0 / 0** for both SFT and validated edit | Decision variables are perfectly linearly represented; constrained decoding removes nearly all behavioral error. This is a readout/scaffold result, not a weight-level capability intervention. |
| 102 / no forcing | SFT 55 / 100 / 90; validated edit 65 / 90 / 0 | Matches the generation bottleneck: the same hidden states support far better behavior when decision fields are externally read out. |
| 102 / majority and random controls | majority over-refuses; random collapses | Probe performance is not explained by trivial forcing or fixed labels. |
| **148 / legacy scorer flattening audit** | uniform-logit legacy accuracy: MMLU 25.0, ARC 31.25, HellaSwag 21.88 | Legacy scoring is length-biased and empirically has a nonzero floor, but uniform logits do **not** manufacture large gains by themselves. Benchmark inflation still requires model-dependent logit structure; continuation/token-normalized metrics remain necessary. |


## Critical replication correction

**Exp41 did not test the original Exp17 DLA/IG operator.** Its script imports `sniff_heads_and_false_nodes` and `apply_false_node_mlp_edit` from the Exp12 false-node pipeline; it never uses Exp17's `probe_dla_and_variance` node selection or IG-5 node selection. The resulting 67.5/100/80 triple exactly matches Exp17 probe2 false-node. Therefore:

1. Exp41 is a valid GPU replication of the **false-node residual-direction** branch.
2. The historical **80.0% DLA** and **82.5% IG-5** results remain untested as exact GPU replications.
3. Exp52 is adjacent evidence, not closure: the DLA/unembed direction is positive on the validated neuron mask, but Exp17's original probe-selection pipeline still needs exact replication.
4. Any queue item claiming to validate “the original operator” must report node-selection provenance and selected-neuron overlap with Exp17 probe1/probe4.

## Full before/after delta table — all metric-bearing experiment summaries

Rows below are sorted by raw composite score, **not by scientific trustworthiness**. Comparator/reference arms are excluded where identified. `warnings` marks collapse, no-op, missing-artifact, cache confounding, comparator/eval-format, or interpretation caveats; inspect `research/AUTOGRADED_SCORES.json` before using the row mechanistically.

| Rank | Experiment / best arm | Validity | Exact Δ (pp) | Unsafe Δ (pp) | Benign-FR Δ (pp) | Score | Grade |
|---:|---|---|---:|---:|---:|---:|:---:|
| 1 | 17 / `probe4_integrated_grads_5` | valid data | **+27.5** | -10.0 | -90.0 | **9.05** | S |
| 2 | 102 / `sft_manual_probe` | warnings | **+40.0** | +0.0 | -90.0 | **8.30** | A |
| 3 | 1 / `arm2_localized_grpo_only` | warnings | **+25.0** | -6.7 | -90.0 | **8.10** | A |
| 4 | 144 / `h22_seed7` | warnings | **+27.5** | -6.7 | -90.0 | **7.90** | A |
| 5 | 85 / `es_final_center` | warnings | **+27.5** | -6.7 | -90.0 | **7.90** | A |
| 6 | 129 / `strength_0p9` | warnings | **+25.0** | -6.7 | -90.0 | **7.90** | A |
| 7 | 130 / `columns_16` | warnings | **+25.0** | -6.7 | -90.0 | **7.90** | A |
| 8 | 11 / `v3_closed_loop_sniff_edit_rl` | valid data | **+22.5** | -10.0 | -90.0 | **7.85** | A |
| 9 | 80 / `rl_only` | valid data | **+25.0** | -6.7 | -80.0 | **7.60** | A |
| 10 | 57 / `halo_regex_complementary_mlp_r16` | warnings | **+20.0** | -10.0 | -90.0 | **7.60** | A |
| 11 | 136 / `learned_gate_best_decision` | warnings | **+20.0** | -6.7 | -90.0 | **7.40** | B |
| 12 | 86 / `oscillating_decay` | valid data | **+17.5** | -6.7 | -90.0 | **7.35** | B |
| 13 | 63 / `strength_0p90_cols32` | valid data | **+17.5** | -10.0 | -90.0 | **7.35** | B |
| 14 | 132 / `hybrid_capability_primary` | warnings | **+17.5** | -6.7 | -90.0 | **7.15** | B |
| 15 | 145 / `layer_20` | warnings | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 16 | 146 / `layer20_cache_corrected` | warnings | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 17 | 98 / `columns_32` | valid data | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 18 | 99 / `l29_l28_split_32_each` | valid data | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 19 | 96 / `freeze_then_merge_8x4` | warnings | **+15.0** | -10.0 | -90.0 | **6.90** | B |
| 20 | 64 / `l29_l28_plus_l27` | warnings | **+12.5** | -10.0 | -90.0 | **6.85** | B |
| 21 | 53 / `complianceheavy_0.3_0.1` | valid data | **+17.5** | -16.7 | -90.0 | **6.75** | B |
| 22 | 16 / `raw_rank1` | warnings | **+10.0** | -10.0 | -90.0 | **6.70** | B |
| 23 | 100 / `targeted_lora_complement_mlp_only` | warnings | **+12.5** | -10.0 | -90.0 | **6.65** | B |
| 24 | 65 / `insert_after_l27` | warnings | **+10.0** | -6.7 | -90.0 | **6.60** | B |
| 25 | 60 / `inverter_then_complementary_lora` | valid data | **+10.0** | -10.0 | -90.0 | **6.60** | B |
| 26 | 95 / `strength_0p9` | valid data | **+10.0** | -10.0 | -90.0 | **6.40** | C |
| 27 | 58 / `bottom_ranked_heads` | warnings | **+12.5** | -16.7 | -90.0 | **6.25** | C |
| 28 | 13 / `arm3_projection` | valid data | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 29 | 18 / `armB_mlp_false_node_only` | valid data | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 30 | 101 / `served_weight_action_only` | warnings | **-12.5** | +0.0 | -90.0 | **5.80** | C |
| 31 | slider head fork/v1 single / `final` | warnings | **+2.5** | +0.0 | -30.0 | **5.70** | C |
| 32 | 42 / `after` | warnings | **-12.5** | -10.0 | -60.0 | **5.65** | C |
| 33 | 41 / `top2_gated` | warnings | **+12.5** | +0.0 | -10.0 | **5.60** | C |
| 34 | 50 / `disjoint_layer_vectors` | valid data | **+5.0** | -13.3 | -90.0 | **5.50** | D |
| 35 | 51 / `hidden_align_late_l29_lambda1em3` | valid data | **+5.0** | -13.3 | -90.0 | **5.50** | D |
| 36 | 54 / `cap_annealed_rl` | valid data | **+5.0** | -13.3 | -90.0 | **5.50** | D |
| 37 | relp causal sniff rl / `eval` | valid data | **+7.5** | +0.0 | -10.0 | **5.40** | D |
| 38 | 90 / `width_16` | warnings | **-5.0** | -6.7 | -90.0 | **5.40** | D |
| 39 | 62 / `bc_lambda_2` | warnings | **+5.0** | +0.0 | -10.0 | **5.15** | D |
| 40 | 61 / `keyed_zero_warm_gqa` | warnings | **-45.0** | -10.0 | -70.0 | **5.05** | D |
| 41 | 91 / `layer_27` | warnings | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 42 | 92 / `count_1` | warnings | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 43 | 93 / `seed_42` | warnings | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 44 | 52 / `pgs_l29` | warnings | **+2.5** | +0.0 | +0.0 | **4.90** | D |
| 45 | noncollinear pushpull / `pushpull_0.7` | warnings | **+0.0** | +0.0 | +0.0 | **4.85** | D |
| 46 | 34 / `vector_only_rl` | valid data | **-15.0** | +0.0 | +0.0 | **4.85** | D |
| 47 | combined slider subspace / `stage2_combined` | warnings | **-5.0** | -3.3 | -20.0 | **4.80** | D |
| 48 | 142 / `budget_4x_128_steps` | warnings | **-15.0** | -16.7 | -90.0 | **4.80** | D |
| 49 | 140 / `answer_upweighted_4x` | warnings | **-17.5** | -13.3 | -90.0 | **4.80** | D |
| 50 | 138 / `negative_subtractive_gate` | warnings | **+2.5** | +0.0 | +10.0 | **4.70** | D |
| 51 | zero shock stitched head rl / `eval` | valid data | **-10.0** | +0.0 | +10.0 | **4.70** | D |
| 52 | 67 / `opposite_decision_dose` | valid data | **+0.0** | +0.0 | +10.0 | **4.65** | D |
| 53 | operator sensitivity showdown / `resid_dir_gated` | warnings | **+0.0** | +0.0 | +0.0 | **4.65** | D |
| 54 | crosslayer head mlp synergy / `head_only` | warnings | **-10.0** | +0.0 | -30.0 | **4.65** | D |
| 55 | dual polarity circuit surgery / `pushpull_1.0` | warnings | **-12.5** | +0.0 | +10.0 | **4.65** | D |
| 56 | momentum annealed rl / `annealed_edit` | valid data | **-25.0** | +0.0 | +0.0 | **4.65** | D |
| 57 | 59 / `warm_l29kv0_gqa_lane_l27` | warnings | **-27.5** | -40.0 | -90.0 | **4.60** | D |
| 58 | 105 / `lr_0p0003` | warnings | **+0.0** | +0.0 | +0.0 | **4.45** | D |
| 59 | 139 / `hybrid_capability_primary` | warnings | **+0.0** | +0.0 | -50.0 | **4.45** | D |
| 60 | 87 / `negative_false_margin` | warnings | **+0.0** | +0.0 | +0.0 | **4.45** | D |
| 61 | 135 / `behavior_alone` | warnings | **-2.5** | +0.0 | +0.0 | **4.45** | D |
| 62 | 137 / `cross_layer_cascade` | warnings | **-17.5** | +0.0 | -10.0 | **4.45** | D |
| 63 | 103 / `suppression_0p25` | warnings | **-25.0** | +0.0 | +0.0 | **4.45** | D |
| 64 | 83 / `probe_routed` | valid data | **-7.5** | -30.0 | -90.0 | **4.40** | D |
| 65 | 141 / `benchmark_direct_answer_only` | warnings | **-17.5** | -36.7 | -90.0 | **4.20** | D |
| 66 | 147 / `behavior_alone` | warnings | **-5.0** | -6.7 | +0.0 | **4.05** | D |
| 67 | 143 / `contrastive_choice_objective` | warnings | **-10.0** | -6.7 | +0.0 | **4.05** | D |
| 68 | 56 / `schema_frozen_selective_fixed` | valid data | **-22.5** | -3.3 | +10.0 | **4.05** | D |
| 69 | 134 / `routed_width_mixture` | warnings | **-30.0** | -3.3 | +10.0 | **4.05** | D |
| 70 | slider head fork/v5 memory / `final` | warnings | **-40.0** | -100.0 | -90.0 | **3.85** | D |
| 71 | slider head fork/v2 bank4 / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 72 | slider head fork/v3 deep spread / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 73 | slider head fork/v4 wide / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 74 | slider head fork/v6 fullbank wide / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 75 | interference canceling edit / `cancel_lambda3.0` | valid data | **-27.5** | -80.0 | -90.0 | **3.25** | D |
| 76 | 66 / `composite` | warnings | **-37.5** | -100.0 | -90.0 | **3.25** | D |
| 77 | dtype edit sensitivity / `bf16` | warnings | **-40.0** | -100.0 | -90.0 | **3.25** | D |
| 78 | fp32 master edit pipeline / `b_fp32edit_bf16cast` | warnings | **-40.0** | -100.0 | -90.0 | **3.25** | D |
| 79 | collision priced allocation / `pure_dla` | warnings | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 80 | targeted circuit rl / `circuit_columns` | warnings | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 81 | gradfree evolution subspace / `heldout_test` | warnings | **-5.0** | -57.1 | -90.0 | **3.05** | D |
| 82 | multi seed stability sweep / `best_reported` | valid data | **-17.5** | -53.3 | -90.0 | **3.05** | D |
| 83 | 81 / `kl_layer_disjoint_unconstrained` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 84 | ig5 dla closed loop rl / `st0.65` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 85 | random node control proof / `dla_top64` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 86 | steering oracle guided rl / `plain_rl` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 87 | safety vaccination edit / `no_vaccine_edit0.7` | warnings | **-42.5** | -100.0 | -90.0 | **3.05** | D |
| 88 | entropy weighted fork rl / `entropy_weighted` | warnings | **-45.0** | -100.0 | -90.0 | **3.05** | D |
| 89 | 55 / `ash_joint_hint_init_mixed` | warnings | **-55.0** | -100.0 | -90.0 | **3.05** | D |
| 90 | 82 / `rank1_extrapolation_t8` | warnings | **-40.0** | -100.0 | -90.0 | **2.85** | D |
| 91 | 94 / `clamp_none` | warnings | **-55.0** | -100.0 | -90.0 | **2.85** | D |

## Integrity and completeness notes

- **Misplaced duplicate artifacts removed:** the 12:34 harvest accidentally committed byte-identical top-level copies of eight result summaries and one `step_metrics.jsonl` directly under `experiments/`. Because each file was byte-identical to its canonical nested `experiments/*/results/...` artifact, the tracker removed only the duplicate top-level copies; no experiment data was lost.
- **No active duplicate experiment-number directories** at this scan. The earlier exp129 context-gated collision remains resolved as exp131.
- **No NaN/Inf values and no invalid JSON objects** were detected in committed result scans.
- **Exp67 telemetry gap:** `exp67_single_coordinate_decision_steering/results/step_metrics.jsonl` is empty even though the final summary is complete; treat optimizer-trace conclusions as unavailable for that experiment.
- **Diagnostic-only result:** `benchmark_failure_tree` is complete as a benchmark diagnostic (38.3% accuracy, suppression delta 0.0 pp) but has no behavioral intervention metrics and is not leaderboard-graded.
- **Scaled-corpus QC:** `policy_train_scaled`/`policy_eval_scaled` schema and counterfactual pairing audit clean (1600 train/120 heldout, 800 complete train pairs, 0 schema errors). However, 218 shared user-prompt 8-grams mean the split is compositional/template-disjoint rather than entity-level; scaled n=120 metrics require fresh baselines and must not be compared absolutely with legacy n=40. Capability scaled corpus has target-complete 5280 rows and exact+13-gram context decontamination.
- Known missing/queued experiments remain tracked in `research/RESULT_INTEGRITY_REPORT.json`; exp111/120–128/131/133/149/150/152/153 are queues; exp151 is intentionally staged outside autoqueue; exp152/153 declare exclusive_gpu and are serialized behind the newest exclusive job, not completed negative results.
