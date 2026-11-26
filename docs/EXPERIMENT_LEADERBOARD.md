# Experiment Leaderboard — Quality-Controlled Rubric Grades

Last updated: **2026-11-26 09:58 UTC** by the tracker/post-processor.

Composite = `0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R` (see `research/RUBRIC.md`). Machine-generated scores and the integrity scan are in `research/AUTOGRADED_SCORES.json` and `research/RESULT_INTEGRITY_REPORT.json`.

**Canonical baseline:** Halo LoRA SFT checkpoint-36 on the 40-task held-out eval = **55.0% exact / 100.0% unsafe refusal / 90.0% benign false refusal** (also 77.5% action accuracy, 57.5% policy accuracy). Positive Δ benign-FR is worse; negative Δ benign-FR is better only when unsafe refusal remains high.

## Interpretable technique ranking

This table separates historically interpretable results from the wave-1/2/3 operator-failure measurements. **The top two CPU-era rows are not yet GPU-replicated.** The best GPU-validated surgical operator currently is Exp 16 `raw_rank1`; the best fully safety-preserving GPU-validated edit is the Exp 12/41 false-node residual operator.

| Rank | Experiment / best arm | Regime & validity | Exact 55→after (Δ) | Unsafe 100→after (Δ) | Benign-FR 90→after (Δ) | Score | Grade | Interpretation |
|---:|---|---|---:|---:|---:|---:|:---:|---|
| 1 | **17 / `probe1_dla_logit_lens`** | legacy CPU-era; **GPU replication still outstanding** | 55.0→80.0 (**+25.0**) | 100→90.0 (**-10.0**) | 90→0.0 (**-90.0**) | **9.05** | **S** | Historical best zero-grad probe; do not treat as reproduced yet. |
| 2 | **11 / `v3_closed_loop_sniff_edit_rl`** | legacy CPU-era | 55.0→77.5 (**+22.5**) | 100→90.0 (**-10.0**) | 90→0.0 (**-90.0**) | **7.85** | **A** | Strongest closed-loop composition historically; needs modern GPU replication. |
| 3 | **1 / `arm3b_edit_then_target_fixed`** | micro-RL legacy result | 55.0→72.5 (**+17.5**) | 100→93.3 (**-6.7**) | 90→0.0 (**-90.0**) | **7.35** | **B** | Edit→targeted-RL composition works, but safety cost remains. |
| 4 | **16 / `raw_rank1`** | **GPU-validated positive operator** | 55.0→65.0 (**+10.0**) | 100→90.0 (**-10.0**) | 90→0.0 (**-90.0**) | **6.50** | **B** | Best validated raw-performance surgical edit on GPU; not safety-preserving. |
| 5 | **12 / `armB_mlp_false_node_only`** | legacy CPU-era; reproduced by Exp 41 | 55.0→67.5 (**+12.5**) | 100→100.0 (**+0.0**) | 90→80.0 (**-10.0**) | **5.90** | **C** | Safe but modest false-node residual-direction edit. |
| 6 | **13 / `arm3_projection`** | legacy CPU-era | 55.0→67.5 (**+12.5**) | 100→100.0 (**+0.0**) | 90→80.0 (**-10.0**) | **5.90** | **C** | Same behavioral class as false-node continuous projection. |
| 7 | **42 / slider head after RL** | **GPU-valid, performance-negative** | 55.0→42.5 (**-12.5**) | 100→90.0 (**-10.0**) | 90→30.0 (**-60.0**) | **5.65** | **C** | Architecture insert sharply reduces over-refusal but trades away exactness and some unsafe refusal. |
| 8 | **41 / `top2_gated`** | **GPU-valid for false-node operator; mislabeled as original-DLA replication** | 55.0→67.5 (**+12.5**) | 100→100.0 (**+0.0**) | 90→80.0 (**-10.0**) | **5.60** | **C** | Validates the Exp 12 false-node branch, **not** Exp 17 DLA/IG. |
| 9 | **15 / RelP edit** | diagnostic-probe lineage | 55.0→62.5 (**+7.5**) | 100→100.0 (**+0.0**) | 90→80.0 (**-10.0**) | **5.40** | **D** | Useful causal probe, weak stand-alone intervention. |
| 10 | **14 / stitched head + short RL** | **GPU-valid negative** | 55.0→45.0 (**-10.0**) | 100→100.0 (**+0.0**) | 90→100.0 (**+10.0**) | **4.70** | **D** | Zero-shock construction is clean, but short RL worsens over-refusal and exactness. |
| — | **28 / `head_only`** | inconclusive bf16-era signal | 55.0→45.0 (**-10.0**) | 100→100.0 (**+0.0**) | 90→60.0 (**-30.0**) | **4.65** | **D** | Attention-only signal is interesting for head-modification research but confounded by shifted CUDA baseline. |

## Critical replication correction

**Exp 41 did not test the original Exp 17 DLA/IG operator.** Its script imports `sniff_heads_and_false_nodes` and `apply_false_node_mlp_edit` from the Exp 12 false-node pipeline; it never uses Exp 17's `probe_dla_and_variance` node selection or IG-5 node selection. The resulting 67.5/100/80 triple exactly matches Exp 17 probe2 false-node. Therefore:

1. Exp 41 is a valid GPU replication of the **false-node residual-direction** branch.
2. The historical **80.0% DLA** and **82.5% IG-5** results remain untested on GPU.
3. Any queue item claiming to validate “the original operator” must report node-selection provenance and selected-neuron overlap with Exp 17 probe1/probe4.

## Full before/after delta table — all metric-bearing experiment summaries

Rows below are sorted by raw composite score, **not by scientific trustworthiness**. `operator-invalid` rows mostly measure the broken unembedding-direction/always-call collapse regime and must not be read as mechanism rankings. `n=20` marks a noncanonical half-split eval.

| Rank | Experiment / best arm | Validity | Exact Δ (pp) | Unsafe Δ (pp) | Benign-FR Δ (pp) | Score | Grade |
|---:|---|---|---:|---:|---:|---:|:---:|
| 1 | 17 / `probe1_dla_logit_lens` | legacy CPU | **+25.0** | -10.0 | -90.0 | **9.05** | S |
| 2 | 11 / `v3_closed_loop_sniff_edit_rl` | legacy CPU | **+22.5** | -10.0 | -90.0 | **7.85** | A |
| 3 | 1 / `arm3b_edit_then_target_fixed` | legacy micro-RL | **+17.5** | -6.7 | -90.0 | **7.35** | B |
| 4 | 16 / `raw_rank1` | GPU-valid | **+10.0** | -10.0 | -90.0 | **6.50** | B |
| 5 | 13 / `arm3_projection` | legacy CPU | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 6 | 12 / `armB_mlp_false_node_only` | legacy CPU | **+12.5** | +0.0 | -10.0 | **5.90** | C |
| 7 | 42 / `after` | GPU-valid negative exactness | **-12.5** | -10.0 | -60.0 | **5.65** | C |
| 8 | 41 / `top2_gated` | GPU-valid false-node; wrong replication target | **+12.5** | +0.0 | -10.0 | **5.60** | C |
| 9 | 15 / `eval` | diagnostic-probe lineage | **+7.5** | +0.0 | -10.0 | **5.40** | D |
| 10 | 38 / `pushpull_0.7` | degenerate cancellation/no-op | **+0.0** | +0.0 | +0.0 | **4.85** | D |
| 11 | 14 / stitched-head `eval` | GPU-valid negative | **-10.0** | +0.0 | +10.0 | **4.70** | D |
| 12 | 28 / `head_only` | bf16-confounded inconclusive | **-10.0** | +0.0 | -30.0 | **4.65** | D |
| 13 | 23 / `pushpull_1.0` | mathematically degenerate net-sign control | **-12.5** | +0.0 | +10.0 | **4.65** | D |
| 14 | 29 / `annealed_edit` | operator-confounded | **-25.0** | +0.0 | +0.0 | **4.65** | D |
| 15 | 39 / `resid_dir_fixed0.7` | valid control, inactive edit | **-2.5** | +0.0 | +0.0 | **4.65** | D |
| 16 | 37 / `pure_dla` | operator-invalid collapse | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 17 | 33 / `fp32` | control; operator-invalid collapse | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 18 | 40 / `a_fp32_edit_fp32_eval` | control; operator-invalid collapse | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 19 | 35 / `cancel_lambda1.0` | operator-invalid collapse | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 20 | 32 / `circuit_columns` | invalid downstream stack | **-42.5** | -100.0 | -90.0 | **3.25** | D |
| 21 | 26 / `uniform_adv` | collapsed initialization | **-47.5** | -100.0 | -90.0 | **3.05** | D |
| 22 | 25 / `heldout_test` | n=20, objective-invalid | **-5.0** | -57.1 | -90.0 | **3.05** | D |
| 23 | 21 / `st0.65` | collapsed initialization | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 24 | 20 / `dla_top64` | operator-invalid collapse | **-40.0** | -100.0 | -90.0 | **3.05** | D |
| 25 | 36 / `no_vaccine_edit0.7` | operator-invalid collapse | **-42.5** | -100.0 | -90.0 | **3.05** | D |
| 26 | 30 / `plain_rl` | collapsed initialization; overwritten rerun artifact | **-40.0** | -100.0 | -90.0 | **3.05** | D |

## Current integrity and queue status

- **No duplicate experiment numbers** are present in active `experiments/*/README.md` files.
- **Numbering gaps:** 34 and 44–49. Exp 34 was deleted when the activation-vector experiment was reworked into the exp50 lineage; treat 34 as retired/superseded unless the orchestrator explicitly restores it. Gaps 44–49 appear reserved/unassigned, not collisions.
- **Missing result artifacts:** `adaptive_k_proportional_edit` (exp24), `multi_seed_stability_sweep` (exp22), `combined_slider_subspace` (exp43), and exp50–56. Exp 43 has a reported VM-side result but no committed JSON yet at this timestamp.
- **Degenerate/invalid artifacts:** Exp 27 still has all-zero layer deltas because source and target remain the same prompt; Exp 25 uses only 20-row dev/test splits; several gated edits are exact baseline copies because the gate rejected deployment or selected strength 0.0.
- **In-place overwrite warnings:** commit `dfbec20` modified `steering_rl_summary.json` and `stitched_head_summary.json` in place. Git history preserves the old values, but there is no side-by-side backup artifact; future reruns should write versioned filenames or a backup directory before overwriting.
- **NaN/crash scan:** no NaN/Inf values or invalid JSON objects were found in the 37 currently committed result JSON files. Missing-output crashes remain for exp22/exp24 as documented above.

## Pending or not-yet-gradeable

Exp 22 needs a successful summary write/rerun; exp24 needs a device-mismatch fix/rerun; exp27 needs a source/target patching fix; exp31 needs ground-truth policy-code categories; exp43 and exp50–56 are queued/running or awaiting committed results. The autograder intentionally does not grade pure atlases or missing artifacts.
