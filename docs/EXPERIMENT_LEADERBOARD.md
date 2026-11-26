# Experiment Leaderboard — Quality-Controlled Rubric Grades

Last updated: **2026-11-26 12:54 UTC** by the tracker/post-processor.

Composite = `0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R` (see `research/RUBRIC.md`). Machine-generated scores and the integrity scan are in `research/AUTOGRADED_SCORES.json` and `research/RESULT_INTEGRITY_REPORT.json`.

**Canonical baseline:** Halo LoRA SFT checkpoint-36 on the 40-task held-out eval = **55.0% exact / 100.0% unsafe refusal / 90.0% benign false refusal** (also 77.5% action accuracy, 57.5% policy accuracy). Positive Δ benign-FR is worse; negative Δ benign-FR is better only when unsafe refusal remains high.

**Comparator policy:** exact random/uniform/reference/control arms are excluded from technique scoring. Scientifically important comparators remain in raw result JSONs and analysis, but they do not win leaderboard rows. This poll added exclusions for Exp52's DLA reference, Exp55's zero-vector/plain RL comparator, Exp66's validated-edit reference, and Exp101's standard-unrounded comparators.

## Interpretable technique ranking

The top CPU-era probe row is still not exactly GPU-replicated. However, Exp52 now provides adjacent positive GPU evidence for the DLA/unembed direction on the validated top-neuron edit mask: its reference arm is **72.5 / 93.3 / 0**, while PGS-estimated directions are nearly anti-aligned with DLA and fail. This is not yet the missing exact Exp17 replication, but it substantially weakens the earlier hypothesis that the CPU-era DLA direction was purely an artifact.

| Rank | Experiment / best arm | Regime & validity | Exact 55→after (Δ) | Unsafe 100→after (Δ) | Benign-FR 90→after (Δ) | Score | Grade | Interpretation |
|---:|---|---|---:|---:|---:|---:|:---:|---|
| 1 | **17 / `probe4_integrated_grads_5`** | legacy CPU-era; exact GPU replication still outstanding | 55→82.5 (**+27.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **9.05** | **S** | Historical best zero-grad probe (IG-5). |
| 2 | **1 / `arm2_localized_grpo_only`** | GPU-valid but baseline-shifted/overwrite-restored | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **8.10** | **A** | Targeted localized GRPO; directionally supported by exp80. |
| 3 | **85 / `es_final_center`** | **GPU-valid n=40; selection-noise caution** | 55→82.5 (**+27.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.90** | **A** | Strongest new final metric, but replication is mandatory. |
| 4 | **11 / `v3_closed_loop_sniff_edit_rl`** | legacy CPU-era | 55→77.5 (**+22.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.85** | **A** | Strongest historical closed-loop composition. |
| 5 | **80 / `rl_only`** | **GPU-valid canonical-baseline replication** | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→10 (**-80**) | **7.60** | **A** | Major replication of targeted RL alone. |
| 6 | **57 / `halo_regex_complementary_mlp_r16`** | **GPU-valid positive transfer path** | 55→75 (**+20**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.60** | **A** | Post-edit-anchored complementary MLP-only LoRA is the strongest transfer mechanism this wave. |
| 7 | **63 / `strength_0p90_cols32`** | **GPU-valid dose/width replica** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.35** | **B** | Confirms narrow 32-column canonical-strength edit. |
| 8 | **86 / `oscillating_decay`** | **GPU-valid positive** | 55→72.5 (**+17.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.35** | **B** | Polarity cycling prevents destructive fixation. |
| 9 | **98 / `columns_32`** | **GPU-valid dose-response** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.15** | **B** | Narrower edit beats 64/128 columns. |
| 10 | **99 / `l29_l28_split_32_each`** | **GPU-valid allocation result** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.15** | **B** | Split allocation reproduces the narrow-edit operating point. |
| 11 | **96 / `freeze_then_merge_8x4`** | **GPU-valid schedule result** | 55→70 (**+15**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.90** | **B** | Freeze-then-merge avoids joint-training collapse. |
| 12 | **64 / `l29_l28_plus_l27`** | **GPU-valid layer ablation** | 55→67.5 (**+12.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.85** | **B** | L29 is necessary/sufficient for core effect; L27 adds a small increment. |
| 13 | **53 / `complianceheavy_0.3_0.1`** | GPU-valid but safety-costly | 55→72.5 (**+17.5**) | 100→83.3 (**-16.7**) | 90→0 (**-90**) | **6.75** | **B** | Exactness gain has a large unsafe-refusal cost. |
| 14 | **16 / `raw_rank1`** | GPU-validated positive operator | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.70** | **B** | Prior validated surgical operator, now superseded by parameterized/RL variants. |
| 15 | **100 / `targeted_lora_complement_mlp_only`** | GPU-valid transfer comparison | 55→67.5 (**+12.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.65** | **B** | Targeted LoRA beats partial full-FT from the same anchor. |
| 16 | **60 / `inverter_then_complementary_lora`** | GPU-valid but no gain over validated-edit anchor | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.60** | **B** | Reaches the standard post-edit operating point; does not exceed it. |
| 17 | **65 / `insert_after_l27`** | GPU-valid insertion-depth point | 55→65 (**+10**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **6.60** | **B** | L27 insertion is the best shallow slider point tested. |
| 18 | **95 / `strength_0p9`** | GPU-valid dose-response | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.40** | **C** | Canonical strength validated; lower strength trades precision for exactness. |

## Critical replication correction

**Exp41 did not test the original Exp17 DLA/IG operator.** Its script imports `sniff_heads_and_false_nodes` and `apply_false_node_mlp_edit` from the Exp12 false-node pipeline; it never uses Exp17's `probe_dla_and_variance` node selection or IG-5 node selection. The resulting 67.5/100/80 triple exactly matches Exp17 probe2 false-node. Therefore:

1. Exp41 is a valid GPU replication of the **false-node residual-direction** branch.
2. The historical **80.0% DLA** and **82.5% IG-5** results remain untested as exact GPU replications.
3. Exp52 is adjacent evidence, not closure: the DLA/unembed direction is positive on the validated neuron mask, but Exp17's original probe-selection pipeline still needs exact replication.
4. Any queue item claiming to validate “the original operator” must report node-selection provenance and selected-neuron overlap with Exp17 probe1/probe4.

## Full before/after delta table — all metric-bearing experiment summaries

Rows below are sorted by raw composite score, **not by scientific trustworthiness**. Comparator/reference arms are excluded where identified. `warnings` marks collapse, no-op, missing-artifact, or interpretation caveats; inspect `research/AUTOGRADED_SCORES.json` before using the row mechanistically.

| Rank | Experiment / best arm | Validity | Exact Δ (pp) | Unsafe Δ (pp) | Benign-FR Δ (pp) | Score | Grade |
|---:|---|---|---:|---:|---:|---:|:---:|
| 1 | 17 / `probe4_integrated_grads_5` | valid data | **+27.5** | -10.0 | -90.0 | **9.05** | S |
| 2 | 1 / `arm2_localized_grpo_only` | warnings | **+25.0** | -6.7 | -90.0 | **8.10** | A |
| 3 | 85 / `es_final_center` | warnings | **+27.5** | -6.7 | -90.0 | **7.90** | A |
| 4 | 11 / `v3_closed_loop_sniff_edit_rl` | valid data | **+22.5** | -10.0 | -90.0 | **7.85** | A |
| 5 | 80 / `rl_only` | valid data | **+25.0** | -6.7 | -80.0 | **7.60** | A |
| 6 | 57 / `halo_regex_complementary_mlp_r16` | warnings | **+20.0** | -10.0 | -90.0 | **7.60** | A |
| 7 | 86 / `oscillating_decay` | valid data | **+17.5** | -6.7 | -90.0 | **7.35** | B |
| 8 | 63 / `strength_0p90_cols32` | valid data | **+17.5** | -10.0 | -90.0 | **7.35** | B |
| 9 | 98 / `columns_32` | valid data | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 10 | 99 / `l29_l28_split_32_each` | valid data | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 11 | 96 / `freeze_then_merge_8x4` | valid data | **+15.0** | -10.0 | -90.0 | **6.90** | B |
| 12 | 64 / `l29_l28_plus_l27` | warnings | **+12.5** | -10.0 | -90.0 | **6.85** | B |
| 13 | 53 / `complianceheavy_0.3_0.1` | valid data | **+17.5** | -16.7 | -90.0 | **6.75** | B |
| 14 | 16 / `raw_rank1` | warnings | **+10.0** | -10.0 | -90.0 | **6.70** | B |
| 15 | 100 / `targeted_lora_complement_mlp_only` | warnings | **+12.5** | -10.0 | -90.0 | **6.65** | B |
| 16 | 65 / `insert_after_l27` | valid data | **+10.0** | -6.7 | -90.0 | **6.60** | B |
| 17 | 60 / `inverter_then_complementary_lora` | valid data | **+10.0** | -10.0 | -90.0 | **6.60** | B |
| 18 | 95 / `strength_0p9` | valid data | **+10.0** | -10.0 | -90.0 | **6.40** | C |
| 19 | 58 / `bottom_ranked_heads` | warnings | **+12.5** | -16.7 | -90.0 | **6.25** | C |
| 20 | 13 / `arm3_projection` | valid data | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 21 | 18 / `armB_mlp_false_node_only` | valid data | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 22 | 101 / `served_weight_action_only` | warnings | **-12.5** | +0.0 | -90.0 | **5.80** | C |
| 23 | slider head fork/v1 single / `final` | warnings | **+2.5** | +0.0 | -30.0 | **5.70** | C |
| 24 | 42 / `after` | warnings | **-12.5** | -10.0 | -60.0 | **5.65** | C |
| 25 | 41 / `top2_gated` | warnings | **+12.5** | +0.0 | -10.0 | **5.60** | C |
| 26 | 50 / `disjoint_layer_vectors` | valid data | **+5.0** | -13.3 | -90.0 | **5.50** | D |
| 27 | 51 / `hidden_align_late_l29_lambda1em3` | valid data | **+5.0** | -13.3 | -90.0 | **5.50** | D |
| 28 | 54 / `cap_annealed_rl` | valid data | **+5.0** | -13.3 | -90.0 | **5.50** | D |
| 29 | relp causal sniff rl / `eval` | valid data | **+7.5** | +0.0 | -10.0 | **5.40** | D |
| 30 | 90 / `width_16` | valid data | **-5.0** | -6.7 | -90.0 | **5.40** | D |
| 31 | 62 / `bc_lambda_2` | warnings | **+5.0** | +0.0 | -10.0 | **5.15** | D |
| 32 | 61 / `keyed_zero_warm_gqa` | warnings | **-45.0** | -10.0 | -70.0 | **5.05** | D |
| 33 | 92 / `count_1` | valid data | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 34 | 93 / `seed_42` | warnings | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 35 | 52 / `pgs_l29` | warnings | **+2.5** | +0.0 | +0.0 | **4.90** | D |
| 36 | noncollinear pushpull / `pushpull_0.7` | warnings | **+0.0** | +0.0 | +0.0 | **4.85** | D |
| 37 | 34 / `vector_only_rl` | valid data | **-15.0** | +0.0 | +0.0 | **4.85** | D |
| 38 | combined slider subspace / `stage2_combined` | valid data | **-5.0** | -3.3 | -20.0 | **4.80** | D |
| 39 | zero shock stitched head rl / `eval` | valid data | **-10.0** | +0.0 | +10.0 | **4.70** | D |
| 40 | 67 / `opposite_decision_dose` | valid data | **+0.0** | +0.0 | +10.0 | **4.65** | D |
| 41 | operator sensitivity showdown / `resid_dir_gated` | warnings | **+0.0** | +0.0 | +0.0 | **4.65** | D |
| 42 | crosslayer head mlp synergy / `head_only` | warnings | **-10.0** | +0.0 | -30.0 | **4.65** | D |
| 43 | dual polarity circuit surgery / `pushpull_1.0` | warnings | **-12.5** | +0.0 | +10.0 | **4.65** | D |
| 44 | momentum annealed rl / `annealed_edit` | valid data | **-25.0** | +0.0 | +0.0 | **4.65** | D |
| 45 | 59 / `warm_l29kv0_gqa_lane_l27` | warnings | **-27.5** | -40.0 | -90.0 | **4.60** | D |
| 46 | 83 / `probe_routed` | valid data | **-7.5** | -30.0 | -90.0 | **4.40** | D |
| 47 | 56 / `schema_frozen_selective_fixed` | valid data | **-22.5** | -3.3 | +10.0 | **4.05** | D |
| 48 | slider head fork/v5 memory / `final` | warnings | **-40.0** | -100.0 | -90.0 | **3.85** | D |
| 49 | slider head fork/v2 bank4 / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 50 | slider head fork/v3 deep spread / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 51 | slider head fork/v4 wide / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 52 | slider head fork/v6 fullbank wide / `final` | warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 53 | interference canceling edit / `cancel_lambda3.0` | valid data | **-27.5** | -80.0 | -90.0 | **3.25** | D |
| 54 | 66 / `composite` | warnings | **-37.5** | -100.0 | -90.0 | **3.25** | D |
| 55 | dtype edit sensitivity / `bf16` | warnings | **-40.0** | -100.0 | -90.0 | **3.25** | D |
| 56 | fp32 master edit pipeline / `b_fp32edit_bf16cast` | warnings | **-40.0** | -100.0 | -90.0 | **3.25** | D |
| 57 | collision priced allocation / `pure_dla` | warnings | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 58 | targeted circuit rl / `circuit_columns` | warnings | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 59 | gradfree evolution subspace / `heldout_test` | warnings | **-5.0** | -57.1 | -90.0 | **3.05** | D |
| 60 | multi seed stability sweep / `best_reported` | valid data | **-17.5** | -53.3 | -90.0 | **3.05** | D |
| 61 | 81 / `kl_layer_disjoint_unconstrained` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 62 | ig5 dla closed loop rl / `st0.65` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 63 | random node control proof / `dla_top64` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 64 | steering oracle guided rl / `plain_rl` | warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 65 | safety vaccination edit / `no_vaccine_edit0.7` | warnings | **-42.5** | -100.0 | -90.0 | **3.05** | D |
| 66 | entropy weighted fork rl / `entropy_weighted` | warnings | **-45.0** | -100.0 | -90.0 | **3.05** | D |
| 67 | 55 / `ash_joint_hint_init_mixed` | warnings | **-55.0** | -100.0 | -90.0 | **3.05** | D |
| 68 | 94 / `clamp_none` | warnings | **-55.0** | -100.0 | -90.0 | **2.85** | D |

## Integrity and completeness notes

- **Misplaced duplicate artifacts removed:** the 12:34 harvest accidentally committed byte-identical top-level copies of eight result summaries and one `step_metrics.jsonl` directly under `experiments/`. Because each file was byte-identical to its canonical nested `experiments/*/results/...` artifact, the tracker removed only the duplicate top-level copies; no experiment data was lost.
- **No active duplicate experiment-number directories** at this scan. The earlier exp129 context-gated collision remains resolved as exp131.
- **No NaN/Inf values and no invalid JSON objects** were detected in committed result scans.
- **Exp67 telemetry gap:** `exp67_single_coordinate_decision_steering/results/step_metrics.jsonl` is empty even though the final summary is complete; treat optimizer-trace conclusions as unavailable for that experiment.
- **Diagnostic-only result:** `benchmark_failure_tree` is complete as a benchmark diagnostic (38.3% accuracy, suppression delta 0.0 pp) but has no behavioral intervention metrics and is not leaderboard-graded.
- Known missing/queued experiments remain tracked in `research/RESULT_INTEGRITY_REPORT.json`; exp102/103/105/111/120–138 are queues, not completed negative results.
