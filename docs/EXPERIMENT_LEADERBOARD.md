# Experiment Leaderboard — Quality-Controlled Rubric Grades

Last updated: **2026-11-26 12:05 UTC** by the tracker/post-processor.

Composite = `0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R` (see `research/RUBRIC.md`). Machine-generated scores and the integrity scan are in `research/AUTOGRADED_SCORES.json` and `research/RESULT_INTEGRITY_REPORT.json`.

**Canonical baseline:** Halo LoRA SFT checkpoint-36 on the 40-task held-out eval = **55.0% exact / 100.0% unsafe refusal / 90.0% benign false refusal** (also 77.5% action accuracy, 57.5% policy accuracy). Positive Δ benign-FR is worse; negative Δ benign-FR is better only when unsafe refusal remains high.

## Interpretable technique ranking

This table separates historically interpretable results from the wave-1/2/3 operator-failure measurements. **The top CPU-era probe row is not yet GPU-replicated.** The strongest new GPU result is Exp 1 `arm2_localized_grpo_only` (80/93.3/0), but its same-run baseline shifted to 47.5/100/80 and the CPU artifact was overwritten before being restored side-by-side. The best GPU-validated surgical operator remains Exp 16 `raw_rank1`; the best fully safety-preserving GPU-validated edit is the Exp 12/41 false-node residual operator.

| Rank | Experiment / best arm | Regime & validity | Exact 55→after (Δ) | Unsafe 100→after (Δ) | Benign-FR 90→after (Δ) | Score | Grade | Interpretation |
|---:|---|---|---:|---:|---:|---:|:---:|---|
| 1 | **17 / `probe4_integrated_grads_5`** | legacy CPU-era; **GPU replication still outstanding** | 55→82.5 (**+27.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **9.05** | **S** | Historical best zero-grad probe (IG-5); still not reproduced. |
| 2 | **1 / `arm2_localized_grpo_only`** | GPU-valid but baseline-shifted/overwrite-restored | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **8.10** | **A** | Targeted localized GRPO; now directionally supported by exp80. |
| 3 | **85 / `es_final_center`** | **GPU-valid n=40; selection-noise caution** | 55→82.5 (**+27.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.90** | **A** | Strongest new final metric, but ES dev selection did not beat fixed-dose/reference cleanly. |
| 4 | **11 / `v3_closed_loop_sniff_edit_rl`** | legacy CPU-era | 55→77.5 (**+22.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.85** | **A** | Strongest historical closed-loop composition. |
| 5 | **80 / `rl_only`** | **GPU-valid canonical-baseline replication** | 55→80 (**+25**) | 100→93.3 (**-6.7**) | 90→10 (**-80**) | **7.60** | **A** | Major replication: targeted RL alone reaches the Exp1-class operating point without overwrite/baseline shift. |
| 6 | **86 / `oscillating_decay`** | **GPU-valid positive** | 55→72.5 (**+17.5**) | 100→93.3 (**-6.7**) | 90→0 (**-90**) | **7.35** | **B** | Oscillating polarity strongly outperforms constant-polarity controls. |
| 7 | **98 / `columns_32`** | **GPU-valid dose-response** | 55→72.5 (**+17.5**) | 100→90 (**-10**) | 90→0 (**-90**) | **7.15** | **B** | Narrower 32-column validated edit beats 64/128 columns. |
| 8 | **96 / `freeze_then_merge_8x4`** | **GPU-valid schedule result** | 55→70 (**+15**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.90** | **B** | Freeze-then-merge avoids the joint multi-layer collapse. |
| 9 | **16 / `raw_rank1`** | GPU-validated positive operator | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.70** | **B** | Prior best validated surgical operator, now superseded by parameterized variants. |
| 10 | **95 / `strength_0p9`** | GPU-valid dose-response | 55→65 (**+10**) | 100→90 (**-10**) | 90→0 (**-90**) | **6.40** | **C** | Validates the canonical edit dose; lower strength 0.5 gives 77.5 exact but 40% benign-FR. |

## Critical replication correction

**Exp 41 did not test the original Exp 17 DLA/IG operator.** Its script imports `sniff_heads_and_false_nodes` and `apply_false_node_mlp_edit` from the Exp 12 false-node pipeline; it never uses Exp 17's `probe_dla_and_variance` node selection or IG-5 node selection. The resulting 67.5/100/80 triple exactly matches Exp 17 probe2 false-node. Therefore:

1. Exp 41 is a valid GPU replication of the **false-node residual-direction** branch.
2. The historical **80.0% DLA** and **82.5% IG-5** results remain untested on GPU.
3. Any queue item claiming to validate “the original operator” must report node-selection provenance and selected-neuron overlap with Exp 17 probe1/probe4.

## Full before/after delta table — all metric-bearing experiment summaries

Rows below are sorted by raw composite score, **not by scientific trustworthiness**. `operator-invalid` rows mostly measure the broken unembedding-direction/always-call collapse regime and must not be read as mechanism rankings. `n=20` marks a noncanonical half-split eval.

| Rank | Experiment / best arm | Validity | Exact Δ (pp) | Unsafe Δ (pp) | Benign-FR Δ (pp) | Score | Grade |
|---:|---|---|---:|---:|---:|---:|:---:|
| 1 | 17 / `probe4_integrated_grads_5` | see notes | **+27.5** | -10.0 | -90.0 | **9.05** | S |
| 2 | 1 / `arm2_localized_grpo_only` | valid data; see warnings | **+25.0** | -6.7 | -90.0 | **8.10** | A |
| 3 | 85 / `es_final_center` | valid data; see warnings | **+27.5** | -6.7 | -90.0 | **7.90** | A |
| 4 | 11 / `v3_closed_loop_sniff_edit_rl` | see notes | **+22.5** | -10.0 | -90.0 | **7.85** | A |
| 5 | 80 / `rl_only` | GPU-valid | **+25.0** | -6.7 | -80.0 | **7.60** | A |
| 6 | 86 / `oscillating_decay` | GPU-valid | **+17.5** | -6.7 | -90.0 | **7.35** | B |
| 7 | 98 / `columns_32` | GPU-valid | **+17.5** | -10.0 | -90.0 | **7.15** | B |
| 8 | 96 / `freeze_then_merge_8x4` | GPU-valid | **+15.0** | -10.0 | -90.0 | **6.90** | B |
| 9 | 16 / `raw_rank1` | valid data; see warnings | **+10.0** | -10.0 | -90.0 | **6.70** | B |
| 10 | 95 / `strength_0p9` | GPU-valid | **+10.0** | -10.0 | -90.0 | **6.40** | C |
| 11 | 13 / `arm3_projection` | see notes | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 12 | 12 / `armB_mlp_false_node_only` | see notes | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 13 | slider-fork v1 single | valid data; see warnings | **+2.5** | +0.0 | -30.0 | **5.70** | C |
| 14 | 42 / `after` | valid data; see warnings | **-12.5** | -10.0 | -60.0 | **5.65** | C |
| 15 | 41 / `top2_gated` | valid data; see warnings | **+12.5** | +0.0 | -10.0 | **5.60** | C |
| 16 | 15 / `eval` | see notes | **+7.5** | +0.0 | -10.0 | **5.40** | D |
| 17 | 90 / `width_16` | GPU-valid | **-5.0** | -6.7 | -90.0 | **5.40** | D |
| 18 | 92 / `count_1` | GPU-valid | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 19 | 93 / `seed_42` | valid data; see warnings | **+5.0** | +0.0 | -40.0 | **4.95** | D |
| 20 | noncollinear_pushpull / `pushpull_0.7` | valid data; see warnings | **+0.0** | +0.0 | +0.0 | **4.85** | D |
| 21 | 34 / `vector_only_rl` | GPU-valid | **-15.0** | +0.0 | +0.0 | **4.85** | D |
| 22 | combined_slider_subspace / `stage2_combined` | see notes | **-5.0** | -3.3 | -20.0 | **4.80** | D |
| 23 | zero_shock_stitched_head_rl / `eval` | see notes | **-10.0** | +0.0 | +10.0 | **4.70** | D |
| 24 | operator_sensitivity_showdown / `resid_dir_gated` | valid data; see warnings | **+0.0** | +0.0 | +0.0 | **4.65** | D |
| 25 | crosslayer_head_mlp_synergy / `head_only` | valid data; see warnings | **-10.0** | +0.0 | -30.0 | **4.65** | D |
| 26 | dual_polarity_circuit_surgery / `pushpull_1.0` | valid data; see warnings | **-12.5** | +0.0 | +10.0 | **4.65** | D |
| 27 | momentum_annealed_rl / `annealed_edit` | see notes | **-25.0** | +0.0 | +0.0 | **4.65** | D |
| 28 | 83 / `probe_routed` | GPU-valid | **-7.5** | -30.0 | -90.0 | **4.40** | D |
| 29 | slider-fork v5 memory | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.85** | D |
| 30 | slider-fork v2 bank4 | valid data; see warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 31 | slider-fork v3 deep spread | valid data; see warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 32 | slider-fork v4 wide | valid data; see warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 33 | slider-fork v6 fullbank wide | valid data; see warnings | **-55.0** | -100.0 | -90.0 | **3.85** | D |
| 34 | interference_canceling_edit / `cancel_lambda3.0` | collapse/invalid operator regime | **-27.5** | -80.0 | -90.0 | **3.25** | D |
| 35 | dtype_edit_sensitivity / `bf16` | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.25** | D |
| 36 | fp32_master_edit_pipeline / `b_fp32edit_bf16cast` | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.25** | D |
| 37 | collision_priced_allocation / `pure_dla` | valid data; see warnings | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 38 | targeted_circuit_rl / `circuit_columns` | valid data; see warnings | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 39 | gradfree_evolution_subspace / `heldout_test` | valid data; see warnings | **-5.0** | -57.1 | -90.0 | **3.05** | D |
| 40 | multi_seed_stability_sweep / `best_reported` | collapse/invalid operator regime | **-17.5** | -53.3 | -90.0 | **3.05** | D |
| 41 | 81 / `kl_layer_disjoint_unconstrained` | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 42 | ig5_dla_closed_loop_rl / `st0.65` | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 43 | random_node_control_proof / `dla_top64` | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 44 | steering_oracle_guided_rl / `plain_rl` | valid data; see warnings | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 45 | safety_vaccination_edit / `no_vaccine_edit0.7` | valid data; see warnings | **-42.5** | -100.0 | -90.0 | **3.05** | D |
| 46 | entropy_weighted_fork_rl / `entropy_weighted` | valid data; see warnings | **-45.0** | -100.0 | -90.0 | **3.05** | D |
| 47 | 94 / `clamp_none` | valid data; see warnings | **-55.0** | -100.0 | -90.0 | **2.85** | D |

## Current integrity and queue status

- **No duplicate experiment numbers** are present in active `experiments/*/README.md` files. A transient exp103/104 numbering collision introduced at 10:53–10:54 UTC was renumbered upstream within about one minute (rotation lanes are now exp120; alpha-penalty is exp111).
- **Numbering collision fixed:** commit `cfc91d7` accidentally relabeled fp32-master as Exp 34, colliding with the resurrected activation-vector Exp 34, and added a stray top-level `experiments/README.md`. The tracker restored fp32-master to Exp 40 and removed the stray file. Current gaps are 44–49 and 68–79.
- **Missing result artifacts:** `inverse_causal_patching` (exp27), `adaptive_k_proportional_edit` (exp24), exp50–67, exp80–87, exp90–103, exp105, exp111, exp120–131, and `benchmark_failure_tree`. Slider-fork iteration-1 summaries have landed and are graded above.
- **Degenerate/invalid artifacts:** Exp 27's all-zero identity artifact has been preserved under `results/backup_invalid_identity/` and its script corrected/requeued; Exp 25 uses only 20-row dev/test splits; several gated edits are exact baseline copies because the gate rejected deployment or selected strength 0.0.
- **In-place overwrite warnings:** commit `cfc91d7` overwrote Exp 1 CPU micro-RL output, the Exp 31 atlas wall time, and Exp 16 wall time in place. The tracker restored the pre-overwrite Exp 1 CPU artifacts and Exp 31 atlas side-by-side under `backup_pre_cfc91d7*`; earlier backup directories now exist for steering/stitched/whitened reruns. Future harvests must back up before overwrite, not after tracker intervention.
- **NaN/crash scan:** no NaN/Inf values or invalid JSON objects were found in the 48 currently committed result JSON files. Exp24 remains the principal missing-output crash; slider-fork collapse rows are valid JSON/behavioral artifacts, not file corruption.

## Pending or not-yet-gradeable

Exp24 still needs a successful rerun; exp27 has been fixed and requeued after backing up the invalid artifact; exp31 needs ground-truth policy-code categories; exp50–67/exp80–87/exp90–103/exp105/exp111/exp120–131 and benchmark-failure-tree are queued/running or awaiting committed results. Slider-fork iteration 1 is graded but incomplete relative to its stated benchmark program: no MMLU/ARC-Easy/HellaSwag scores are present, and trained head weight files are not committed (only metadata JSONs).
