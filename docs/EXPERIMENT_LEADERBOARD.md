# Experiment Leaderboard — Rubric-Graded (see research/RUBRIC.md)

Composite = 0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R (each /10).
Baseline: SFT micro-model = **55.0% exact** / 40 unseen tasks, **100% unsafe refusal**, **90% benign false refusal**. Last regenerated: 2026-11-26 21:52 UTC.

## Validated deployment snapshot

| Recipe | Behavioral result | Validation status | Main caveat |
|---|---|---|---|
| Qwen3-4B fixed-budget scaled-corpus LoRA (Exp152) | **99.2% / 100% / 0% on entity-disjoint v2** | Provisional scale-track breakthrough; autograded composite **8.30/A** | Single seed pending Exp156 replication; 33.03M trainable params; capability replay/transfer remains unresolved |
| Ultra-sparse output-side residual-subspace projection, L29/28, 16 columns/layer, strength 0.9 | **80.0% exact / 93.3% unsafe / 0% benign-FR** | Reproduced by Exp129 strength response and Exp130 column-count response | Exact Exp17 DLA/IG provenance remains unreplicated on GPU; recipe is validated as an operator family, not as the original experiment |
| H22 sparse edit + post-edit anchored complementary MLP-only LoRA | **82.5% / 93.3% / 0% in seeds 7/42/123** | Exported artifact matches adapter eval; Exp149 shows zero legacy benchmark delta and near-zero continuation delta | Random-layer MLP control also reaches 82.5%, and +2.5 pp is only one n=40 task |
| Cache-corrected L20 width-64 inserted attention head | seed42 only: **72.5% / 90.0% / 0%** | **Failed seed replication** | Exp150 seed7 collapses to **0/0/0** with 0% JSON validity; seed123 collapses to **5/0/0** |
| Probe-forced policy readout (Exp102) | **95.0% / 100% / 0%** | Diagnostic upper bound; majority/random forced controls fail | Inference-time scaffold, not a weight-level intervention |

## Rubric composite ranking (raw autograder)

| Rank | Exp | Technique | Key Result | P | S | C | E | N | K | T | R | **Score** | Grade |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **17** | christmas_tree_node_probes | `probe4_integrated_grads_5`: 82.5% exact (+27.5 pp), 90.0% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 10 | 8 | 9 | 8 | 7 | **9.05** | **S** |
| 2 | **152** | exp152_qwen3_behavioral_scale_pivot | `lora_scaled_fixed_updates`: 99.2% exact (+32.5 pp), 100.0% unsafe refusal, 0.0% benign FR † | 10 | 10 | 10 | 2 | 7 | 8 | 7 | 5 | **8.30** | **A** |
| 3 | **1** | edit_then_target_rl | `arm2_localized_grpo_only`: 80.0% exact (+25.0 pp), 93.3% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **8.10** | **A** |
| 4 | **144** | exp144_sparse_edit_anchored_mlp_lora | `h22_seed7`: 82.5% exact (+27.5 pp), 93.3% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 2 | 7 | 8 | 7 | 7 | **8.00** | **A** |
| 5 | **85** | exp85_evolution_strategies_32d_surgical_sub… | `es_final_center`: 82.5% exact (+27.5 pp), 93.3% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.90** | **A** |
| 6 | **129** | exp129_sparse_edit_strength_response | `strength_0p9`: 80.0% exact (+25.0 pp), 93.3% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.90** | **A** |
| 7 | **130** | exp130_sparse_edit_column_count_response | `columns_16`: 80.0% exact (+25.0 pp), 93.3% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.90** | **A** |
| 8 | **11** | seek_sniff_edit_rl | `v3_closed_loop_sniff_edit_rl`: 77.5% exact (+22.5 pp), 90.0% unsafe refusal, 0.0% benign FR | 9 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **7.85** | **A** |
| 9 | **149** | exp149_h22_continuation_followup | `h22_seed7`: 82.5% exact (+27.5 pp), 93.3% unsafe refusal, 0.0% benign FR † | 10 | 8 | 10 | 2 | 4 | 8 | 7 | 7 | **7.70** | **A** |
| 10 | **80** | exp80_edit_strength_scaffold_annealing | `rl_only`: 80.0% exact (+25.0 pp), 93.3% unsafe refusal, 10.0% benign FR | 10 | 8 | 8 | 2 | 7 | 8 | 7 | 5 | **7.60** | **A** |
| 11 | **57** | exp57_halo_native_targeted_lora | `halo_regex_complementary_mlp_r16`: 75.0% exact (+20.0 pp), 90.0% unsafe refusal, 0.0% benign FR † | 8 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **7.60** | **A** |
| 12 | **136** | exp136_sparse_context_gated_dose_cancellati… | `learned_gate_best_decision`: 75.0% exact (+20.0 pp), 93.3% unsafe refusal, 0.0% benign FR † | 8 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.40** | **B** |
| 13 | **63** | exp63_validated_edit_dose_response | `strength_0p90_cols32`: 72.5% exact (+17.5 pp), 90.0% unsafe refusal, 0.0% benign FR | 7 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **7.35** | **B** |
| 14 | **86** | exp86_oscillating_polarity_edit_cycling | `oscillating_decay`: 72.5% exact (+17.5 pp), 93.3% unsafe refusal, 0.0% benign FR | 7 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **7.35** | **B** |
| 15 | **132** | exp132_capability_objective_slider_ablation | `hybrid_capability_primary`: 72.5% exact (+17.5 pp), 93.3% unsafe refusal, 0.0% benign FR † | 7 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.15** | **B** |
| 16 | **98** | exp98_validated_edit_column_count_response | `columns_32`: 72.5% exact (+17.5 pp), 90.0% unsafe refusal, 0.0% benign FR | 7 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.15** | **B** |
| 17 | **99** | exp99_validated_edit_layer_allocation | `l29_l28_split_32_each`: 72.5% exact (+17.5 pp), 90.0% unsafe refusal, 0.0% benign FR | 7 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **7.15** | **B** |
| 18 | **96** | exp96_freeze_then_merge_schedule | `freeze_then_merge_8x4`: 70.0% exact (+15.0 pp), 90.0% unsafe refusal, 0.0% benign FR † | 6 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **6.90** | **B** |
| 19 | **64** | exp64_validated_edit_layer_ablation | `l29_l28_plus_l27`: 67.5% exact (+12.5 pp), 90.0% unsafe refusal, 0.0% benign FR † | 5 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **6.85** | **B** |
| 20 | **53** | exp53_dual_sided_advantage_sculpting | `complianceheavy_0.3_0.1`: 72.5% exact (+17.5 pp), 83.3% unsafe refusal, 0.0% benign FR | 7 | 5 | 10 | 4 | 7 | 8 | 7 | 5 | **6.75** | **B** |
| 21 | **16** | whitened_subspace_rfm_edit | `raw_rank1`: 65.0% exact (+10.0 pp), 90.0% unsafe refusal, 0.0% benign FR † | 4 | 8 | 10 | 4 | 9 | 7 | 6 | 6 | **6.70** | **B** |
| 22 | **100** | exp100_halo_native_transfer_path_showdown | `targeted_lora_complement_mlp_on…`: 67.5% exact (+12.5 pp), 90.0% unsafe refusal, 0.0% benign FR † | 5 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **6.65** | **B** |
| 23 | **60** | exp60_benign_keyed_refusal_inverter | `inverter_then_complementary_lora`: 65.0% exact (+10.0 pp), 90.0% unsafe refusal, 0.0% benign FR | 4 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **6.60** | **B** |
| 24 | **65** | exp65_slider_insertion_layer_sweep | `insert_after_l27`: 65.0% exact (+10.0 pp), 93.3% unsafe refusal, 0.0% benign FR † | 4 | 8 | 10 | 4 | 7 | 8 | 7 | 5 | **6.60** | **B** |
| 25 | **95** | exp95_validated_edit_strength_response | `strength_0p9`: 65.0% exact (+10.0 pp), 90.0% unsafe refusal, 0.0% benign FR | 4 | 8 | 10 | 2 | 7 | 8 | 7 | 5 | **6.40** | **C** |
| 26 | **58** | exp58_attention_head_sparse_rl | `bottom_ranked_heads`: 67.5% exact (+12.5 pp), 83.3% unsafe refusal, 0.0% benign FR † | 5 | 5 | 10 | 4 | 7 | 8 | 7 | 5 | **6.25** | **C** |
| 27 | **13** | bitflip_false_node_and_defense | `arm3_projection`: 67.5% exact (+12.5 pp), 100.0% unsafe refusal, 80.0% benign FR | 5 | 10 | 1 | 4 | 7 | 8 | 7 | 5 | **5.90** | **C** |
| 28 | **12** | head_and_node_sniff_rl | `armB_mlp_false_node_only`: 67.5% exact (+12.5 pp), 100.0% unsafe refusal, 80.0% benign FR | 5 | 10 | 1 | 4 | 7 | 8 | 7 | 5 | **5.90** | **C** |
| 29 | **101** | exp101_merge_aware_served_weight_finetune | `served_weight_action_only`: 42.5% exact (-12.5 pp), 100.0% unsafe refusal, 0.0% benign FR † | 0 | 10 | 10 | 2 | 7 | 8 | 7 | 5 | **5.80** | **C** |
| 30 | **slider_head_fork** | slider_head_fork | `final`: 57.5% exact (+2.5 pp), 100.0% unsafe refusal, 60.0% benign FR † | 1 | 10 | 1 | 6 | 10 | 10 | 9 | 5 | **5.70** | **C** |

† = validity warning. E is full experiment wall time as recorded by the harness, not always marginal deployment cost. The table is sorted by rubric composite, not by preference for deployment; use the validated snapshot above for the current recipe choice.

## Deployment-relevant interpretation

1. **Best validated zero-grad recipe:** ultra-sparse output-side residual-subspace projection, layers 29/28, 16 columns/layer, strength 0.9 — **80.0/93.3/0**, reproduced through Exp129/Exp130. Its full sweep wall time makes E look poor, but the final edit itself is cheap once hyperparameters are fixed.
2. **Best exact-compliance package:** H22 sparse edit + post-edit anchored MLP-only LoRA — **82.5/93.3/0 in all three seeds**, exported-artifact consistent, with Exp149 continuation deltas mostly zero. The gain over the sparse edit is only one eval task and is matched by random-layer control, so use it as a strong package, not as proof of layer-specific transfer.
3. **Best cheap historical probe:** Exp17 IG-5 remains composite-efficient (**82.5/90/0**, probe time 4.72s), but it permits one more unsafe non-refusal than the modern sparse recipe and is not yet exactly replicated under the current GPU pipeline.
4. **Exp102 is not ranked** because it is an inference-time forced-decoding diagnostic, not a weight-level intervention. Its **95/100/0** result is an upper-bound/readout-bottleneck proof, not a deployable model artifact.

## Diagnostics and upper bounds

| Exp | Diagnostic | Result | Interpretation |
|---:|---|---|---|
| 102 | Probe-forced policy readout | **95.0% exact / 100% unsafe / 0% benign-FR** | Hidden states contain correct decisions; greedy readout is a bottleneck. Not a ranked intervention. |
| 146 | Cache-mode inversion control | L20 legacy cache **0/0/0**, cache-corrected **72.5/90/0** | Pre-fix slider conclusions are generation artifacts. |
| 150 | Cache-corrected L20 seed replication | seed42 **72.5/90/0**, seed7 **0/0/0**, seed123 **5/0/0** | Positive L20 behavior is seed-fragile; attention insertion is not validated. |
| 153 | OLMoE stage-zero | original40 **25/96.7/100**, scaled120 **15.8/83.3/55**, cache diff **0.0** | Feasible scale pivot with large behavior headroom and separable expert routing. |
| 19 | Cross-talk/hub-protection audit | true/false refusal co-movement ratio ≈1.0 | Refusal circuits are entangled; hub protection alone does not decouple them. |
| 27 | Inverse causal patching | all layer deltas 0.0 | Identity-patch artifact; no causal evidence. |
| 31 | Request-type circuit mapper | only pseudo-types `comms_task`/`other_unsafe` | Invalid keyword classifier; atlas must use policy codes. |
| 41 | Original-operator replication | top2-gated **67.5/100/80** | Mislabeled replication target: validates broad contrastive family, not exact Exp17 DLA/IG replication. |

## Key cross-experiment conclusions

1. **Sparse continuous directional surgery wins.** Binary masks/sign flips, broad high-dose edits, and complex blending architectures lose.
2. **Safety-preserving compliance repair is partly solved but capped.** The best interventions remove benign false refusal while retaining 93.3% unsafe refusal, but exact compliance plateaus at 80–82.5%.
3. **Representation is not the bottleneck at 135M.** Probe-forced decoding reaches 95/100/0; ordinary generation does not.
4. **Capability transfer is falsified at 135M for tested inserted-head objectives.** Continuation-token transfer is zero even when validation loss improves.
5. **Attention insertion is not currently robust.** The cache artifact was real, but Exp150 shows the corrected L20 procedure is seed-fragile rather than deployment-ready.
6. **The frontier is now scale + readout.** OLMoE stage-zero passes feasibility/cache/routing gates; watch Exp154 expert adapters, Exp152 Qwen3-4B, and any follow-up that converts Exp102's hidden-state correctness into weight-level greedy decoding.
