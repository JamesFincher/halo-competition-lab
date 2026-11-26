#!/usr/bin/env python3
"""Render the living experiment analysis and leaderboard from AUTOGRADED_SCORES."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES = json.loads((ROOT / "research/AUTOGRADED_SCORES.json").read_text())
INTEGRITY = json.loads((ROOT / "research/RESULT_INTEGRITY_REPORT.json").read_text())

NON_TECHNIQUE_DIAGNOSTICS = {
    "exp102_closed_form_decision_readout",
    "exp145_inversion_logit_diagnostic",
    "exp146_cache_mode_inversion_diagnostic",
    "exp148_legacy_scorer_flattening_audit",
    "exp150_l20_cache_corrected_seed_replication",
    "exp153_olmoe_expert_stage_zero",
}


def exp_number(name: str) -> int:
    m = re.match(r"exp(\d+)", name)
    if m:
        return int(m.group(1))
    legacy = {
        "surgical_crosstalk_audit": 19,
        "random_node_control_proof": 20,
        "ig5_dla_closed_loop_rl": 21,
        "multi_seed_stability_sweep": 22,
        "dual_polarity_circuit_surgery": 23,
        "adaptive_k_proportional_edit": 24,
        "gradfree_evolution_subspace": 25,
        "entropy_weighted_fork_rl": 26,
        "inverse_causal_patching": 27,
        "crosslayer_head_mlp_synergy": 28,
        "momentum_annealed_rl": 29,
        "steering_oracle_guided_rl": 30,
        "request_circuit_mapper": 31,
        "targeted_circuit_rl": 32,
        "dtype_edit_sensitivity": 33,
        "interference_canceling_edit": 35,
        "safety_vaccination_edit": 36,
        "collision_priced_allocation": 37,
        "noncollinear_pushpull": 38,
        "operator_sensitivity_showdown": 39,
        "fp32_master_edit_pipeline": 40,
        "original_operator_replication": 41,
        "slider_gated_attention_head": 42,
        "combined_slider_subspace": 43,
        "edit_then_target_rl": 1,
        "seek_sniff_edit_rl": 11,
        "head_and_node_sniff_rl": 12,
        "bitflip_false_node_and_defense": 13,
        "zero_shock_stitched_head_rl": 14,
        "relp_causal_sniff_rl": 15,
        "whitened_subspace_rfm_edit": 16,
        "christmas_tree_node_probes": 17,
        "behavior_fingerprint_atlas": 18,
    }
    base = name.split("/")[0]
    return legacy.get(base, 999)


def clean(text: object, limit: int = 120) -> str:
    s = str(text).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def warn_marker(row: dict) -> str:
    return " †" if row.get("validity_warnings") else ""


def first_warning(row: dict, limit: int = 150) -> str:
    warnings = row.get("validity_warnings") or []
    # Experiment-specific semantic notes are appended after generic metric warnings;
    # prefer them because they carry the actual interpretation.
    return clean(warnings[-1], limit) if warnings else "—"


def fmt_metric(x: object) -> str:
    return "—" if x is None else f"{float(x):.1f}"


def canonical_rows() -> list[tuple[str, dict]]:
    rows = []
    for name, row in SCORES.items():
        base = name.split("/")[0]
        if base in NON_TECHNIQUE_DIAGNOSTICS:
            continue
        rows.append((name, row))
    return sorted(rows, key=lambda kv: (-kv[1]["composite"], -kv[1]["exact"], kv[0]))


def leaderboard_table(limit: int = 30) -> str:
    lines = [
        "| Rank | Exp | Technique | Key Result | P | S | C | E | N | K | T | R | **Score** | Grade |",
        "|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, (name, row) in enumerate(canonical_rows()[:limit], 1):
        exp = exp_number(name)
        exp_cell = f"**{exp}**" if exp != 999 else f"**{name.split('/')[0]}**"
        key = (
            f"`{clean(row['best_arm'], 32)}`: {fmt_metric(row['exact'])}% exact "
            f"({row['gain']:+.1f} pp), {fmt_metric(row['unsafe'])}% unsafe refusal, "
            f"{fmt_metric(row['benign_fr'])}% benign FR{warn_marker(row)}"
        )
        lines.append(
            f"| {rank} | {exp_cell} | {clean(name.split('/')[0], 44)} | {key} | "
            f"{row['P']:.0f} | {row['S']:.0f} | {row['C']:.0f} | {row['E']:.0f} | "
            f"{row['N']:.0f} | {row['K']:.0f} | {row['T']:.0f} | {row['R']:.0f} | "
            f"**{row['composite']:.2f}** | **{row['grade']}** |"
        )
    return "\n".join(lines)


def ledger_table() -> str:
    rows = sorted(SCORES.items(), key=lambda kv: (exp_number(kv[0]), kv[0]))
    lines = [
        "| Exp | Experiment | Best extracted arm | Exact | Unsafe refusal | Benign FR | Policy | Score | Grade | Main validity note |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, row in rows:
        exp = exp_number(name)
        exp_cell = str(exp) if exp != 999 else "—"
        lines.append(
            f"| {exp_cell} | `{clean(name, 48)}` | `{clean(row['best_arm'], 34)}` | "
            f"{fmt_metric(row['exact'])}% | {fmt_metric(row['unsafe'])}% | {fmt_metric(row['benign_fr'])}% | "
            f"{fmt_metric(row.get('policy_accuracy'))}% | {row['composite']:.2f} | {row['grade']} | {first_warning(row)} |"
        )
    return "\n".join(lines)


def warning_counts() -> tuple[int, int]:
    files = INTEGRITY.get("file_scans", [])
    return len(files), sum(len(f.get("warnings", [])) for f in files)


def missing_list() -> str:
    missing = INTEGRITY.get("missing_result_experiments", [])
    if not missing:
        return "None currently reported by the integrity scan."
    return ", ".join(f"`{x}`" for x in missing)


def result_count(prefix: str) -> int:
    return sum(1 for p in ROOT.glob(f"experiments/{prefix}*/results/**/*.json") if p.is_file())


def result_state(prefix: str, label: str) -> str:
    count = result_count(prefix)
    if count:
        return f"{label}: {count} result JSON{'s' if count != 1 else ''} present"
    return f"{label}: no result JSONs yet"


def frontier_result_status() -> str:
    return "; ".join(
        [
            result_state("exp150", "Exp150 L20 seed replication"),
            result_state("exp151", "Exp151 depth re-ablation"),
            result_state("exp152", "Exp152 Qwen3-4B pivot"),
            result_state("exp153", "Exp153 OLMoE stage-zero"),
            result_state("exp154", "Exp154 OLMoE expert adapters"),
            result_state("exp155", "Exp155 OLMoE surface controls"),
        ]
    ) + "."


def render_analysis(now: str) -> str:
    file_count, warn_count = warning_counts()
    return f"""# Experiment Analysis — Living Research Notes

Last updated: {now} UTC by the results-analysis subagent. This document is regenerated from `research/AUTOGRADED_SCORES.json` plus manual synthesis; do not treat older per-wave interpretations as current when this synthesis contradicts them.

## Executive synthesis — current validated state

### What is now established

1. **Behavioral control at 135M is essentially solved, but only to a ceiling.** The canonical ultra-sparse output-side residual-subspace edit — layers 29/28, 16 columns per layer, strength 0.9 — reaches **80.0% exact / 93.3% unsafe refusal / 0% benign false refusal**, with **95.0% action** and **95.0% policy** accuracy. Exp129 and Exp130 independently hit the same configuration.
2. **The current best exact-compliance package is H22.** Ultra-sparse edit + post-edit KL-anchored complementary MLP-only LoRA reaches **82.5% / 93.3% / 0% in all three seeds (7/42/123)**, with exported merged artifacts matching the adapter eval. Exp149 strengthens the package: legacy benchmark deltas are exactly zero in every seed, and continuation deltas are zero except byte-normalized HellaSwag **-2.0 pp** in seeds 7/123 (one item at n=50). However, the random-layer MLP control also reaches **82.5%**, so the +2.5 pp over sparse edit is generic post-edit adaptation, not evidence of layer-specific repair; it is also only one task at n=40.
3. **Cache-corrected attention insertion is not seed-robust.** Exp150 replicated the seed-42 L20 width-64 result exactly (**72.5/90/0**), but seed 7 collapses to **0/0/0 with 0% JSON validity** and seed 123 collapses to **5/0/0**. The pre-fix cache artifact was real, but fixing it does not make the training procedure reliable.
4. **Exp102 shows the true bottleneck is readout/decoding, not representation.** Hidden-state probes achieve 100% crossfit accuracy for action and tool, and forcing the probe-selected `policy_code` yields **95.0% exact / 100% unsafe / 0% benign-FR**. This is not a deployable weight-level intervention, but it proves the correct decision is internally represented while greedy generation fails to read it out.
5. **The 80–82.5% ceiling is method-independent so far.** Sparse edits, GRPO/RL variants, vector RL, and H22 all cluster near the same ceiling. This points to data/model/eval saturation rather than a single-method failure.
6. **Capability transfer at 135M is falsified for inserted-head objectives tested so far.** Cache-corrected full-text, answer-focused, answer-only, benchmark-format, contrastive-choice, and budget-scaled variants improve validation loss but produce no continuation-benchmark transfer; more capability training actively damages behavior (exact falls from **37.5% at 32 steps** to **20.0% at 512 steps** in Exp142).
7. **The field has pivoted to data/model scale.** Exp153 OLMoE stage-zero is now complete and passes the feasibility gates; Exp152 Qwen3-4B and Exp154 OLMoE adapters remain the next interventional tests. Current frontier artifacts: {frontier_result_status()}

### Critical validity corrections

- **Dtype is not the root cause.** Exp33/Exp40 show fp32 edit + fp32 eval still collapses when using the wrong operator; fp32/bf16 max weight deltas are nearly identical (**0.065996 vs 0.066406**).
- **Unembedding-direction rank-1 projection is destructive.** It produces always-call collapse (typically **12.5–15% exact / 0% unsafe**). Measured residual contrast directions plus gated strengths are the valid family.
- **Exp41 did not replicate the original DLA/IG experiment.** The autograder flags it as a mislabeled replication target: it used Exp12 false-node selection rather than Exp17 probe1-DLA/probe4-IG selection. Its **67.5% / 100% / 80%** top2-gated result validates the broad contrastive operator family, but the original **80–82.5% CPU-era DLA/IG** result still needs exact replication.
- **The slider literature before the cache fix is confounded.** Legacy cached generation made inserted attention heads token-local rather than full-context. Exp146 is decisive: L20 legacy cache scores **0/0/0**, while cache-corrected L20 scores **72.5/90/0**; cache-corrected L27 control scores **45/100/100**.
- **The legacy real benchmark is not credible capability evidence.** Uniform-logit noise nearly matches real baseline because of structural length bias. Only continuation-token scoring is currently trusted.

## Measurement rules

- Canonical legacy behavior eval: **40 tasks**, 30 unsafe and 10 benign. Exact moves in 2.5 pp increments; one unsafe task is 3.33 pp and one benign task is 10 pp.
- Canonical SFT baseline: **55.0% exact**, 77.5% action, 57.5% policy, **100% unsafe refusal**, **90% benign false refusal**.
- New scaled heldout behavior evals use **n=120** and a different distribution; absolute rates must not be compared with n=40 without rebaselining. For scaled-model interventions, the autograder computes P gain against the same-model/same-run baseline when one is present rather than the canonical 135M SFT baseline.
- Scores below use the autograder's rubric implementation. E reflects the full experiment wall clock, including sweeps/audits, not necessarily the marginal cost of deploying the final recipe.
- A dagger (†) in the leaderboard means the row has one or more validity warnings; the exact warning is in the ledger below and in `research/AUTOGRADED_SCORES.json`.

## Deep-dive synthesis by mechanism

### A. Sparse surgical edits — validated winner family

Exp95 strength response: strength 0.5 gives **77.5/93.3/40**, strength 0.9 gives **65.0/90/0**, strength 1.3 collapses safety to **30.0/63.3/0** at the original 64-column dose. The ultra-sparse follow-up changed the operating point. At 16 columns/layer, strength response is **0.5 → 70/100/60**, **0.7 → 77.5/93.3/20**, **0.9 → 80/93.3/0**, **1.2 → 60/86.7/0**. At fixed strength 0.9, column response is **4 → 60/100/70**, **8 → 77.5/93.3/0**, **16 → 80/93.3/0**, **32 → 72.5/90/0**. The optimum is therefore narrow and nonmonotonic: **ultra-sparse, high-strength, output-side projection**, not the original broad 64-column edit.

Exp85 evolution strategies reached **82.5/93.3/0** with only 32 total edited coordinates, but the same file's fixed-dose and random-search references are strong; the autograder correctly warns that ES dev selection did not beat the fixed-dose reference. Treat Exp85 as evidence that a very low-dimensional good solution exists, not yet proof that evolutionary search is necessary.

### B. Post-edit adaptation — useful but generic

Exp57 targeted Halo-native LoRA reached **75.0/90/0** for complementary MLP-only r16. Exp100 shows targeted MLP-only LoRA at **67.5/90/0**, while partial full fine-tuning is worse and vector-only adaptation is much worse. Exp144/H22 is stronger (**82.5/93.3/0** in every seed), and Exp149 now shows the exported artifact is capability-safe within measurement resolution: zero legacy MMLU/ARC/HellaSwag delta in all seeds and continuation deltas mostly zero, with only byte-normalized HellaSwag **-2.0 pp** in seeds 7/123. The adaptation itself is small: **405,504 trainable parameters (0.301%)**, 8 grouped steps, **62.0–94.5s per seed**, and maximum step KL only **0.00207**. The random-layer MLP control still ties behavior. Therefore post-edit adaptation is a useful generic regularizer/repair stage with no observed continuation cost, not validated layer-specific mechanistic repair.

### C. RL and scaffolding — still unreliable

Exp80's RL-only arm reaches **80.0/93.3/10**, close to the sparse edit but with nonzero benign false refusal. Yet many edit-then-RL arms collapse or fail to preserve gains. The consistent pattern is that RL can adapt behavior but does not reliably break the ceiling and often damages the refusal/compliance balance. RL should be treated as a baseline adaptation method, not the mechanism most likely to solve readout.

### D. Attention insertion — cache artifact corrected, seed replication failed

The original Exp42 scalar slider head was exactness-negative (**42.5/90/30**, alpha 0.0309). Exp96 showed sequential freeze-then-merge training beating joint multi-head training (**70/90/0** versus collapse). The cache artifact then invalidated pre-fix mechanistic interpretations. Cache-corrected Exp145/146 show L20 width64 at **72.5/90/0**, while L27 remains poor (**45/100/100**).

Exp150 is the decisive seed test under fixed conditions (layer 20, width 64, 32 steps, grouped REINFORCE, no KL/BC, cache disabled for reward generation/eval). Its cache-off same-run baseline is already slightly regime-shifted at **52.5/93.3/90** with 95% JSON validity, so absolute values are not directly canonical. Seed 42 exactly reproduces the reference **72.5/90/0** with final alpha **-0.01483**. Seed 7 collapses to **0/0/0**, JSON validity **0%**, final alpha **+0.010525**, and mean reward **0.0 for the last 8 steps** — a malformed-generation failure mode. Seed 123 collapses to **5.0/0/0** despite late training rewards around 0.6, with final alpha **-0.034252** — a distinct safety-collapse failure mode. Thus the positive L20 result is real but fragile: tiny seed-dependent alpha trajectories can move between a beneficial negative-alpha operating point, positive-alpha invalidation, and larger-negative-alpha safety collapse. {result_state('exp151', 'Exp151 depth re-ablation')}. Attention insertion should not be deployment-ranked unless stabilization/constraint mechanisms make the seed distribution robust.

### E. Capability transfer — falsified at 135M

Across Exp139–143 and Exp147, validation loss can improve while continuation-benchmark transfer remains zero. Behavior often deteriorates: Exp142 exact compliance falls **37.5% → 40.0% → 20.0%** as budget rises 32 → 128 → 512 steps. This is direct evidence that capability and behavior objectives interfere in the 135M model. The data-scale/model-scale pivot is therefore evidence-driven, not merely a scaling fashion.

### F. Architecture blending — complexity is not paying

Routed width mixture (**25/96.7/100**), capability×behavior blend (**40/100/90**), cross-layer cascade (**37.5/100/80**), and subtractive gates all underperform the plain sparse edit. Exp138's positive gate is important mechanistically: it preserves exactness at **55.0%** while reducing benign false refusal to **0%**, but unsafe falls to **93.3%** and it does not beat the winner. These results argue against adding architectural complexity unless scale or data changes the bottleneck.

### G. Readout bottleneck — the most important diagnostic

Exp102 compared ordinary greedy decoding with probe-forced policy decoding. Ordinary SFT is **55/100/90**. Its local “validated edit” arm is **65/90/0**, but this is not the canonical 16-columns/layer champion: the file edits **64 selected neurons per layer** at layers 29/28 (**73,728 weights**, 0.05481%) versus the canonical **16 per layer** (**18,432 weights**). Probe-forced policy decoding is **95/100/0** for both models. Majority and random forced-policy controls are catastrophic, so the result is not an artifact of forcing tokens per se. The implication is that the model often internally represents correct action/tool/policy decisions but fails to emit them through greedy readout. This reframes the next research question: weight-level methods may be hitting a decoder/readout ceiling rather than a representation ceiling, and Exp102 should not be read as a replication failure of the canonical sparse edit.

### H. Large-model pivot — OLMoE stage-zero passed, interventions pending

`research/LARGE_MODEL_FEASIBILITY.md` estimates Qwen3-4B LoRA experiments at roughly 15–30 minutes per arm and OLMoE stage-zero at 10–20 minutes after download on one H100. Current large-model artifacts: {result_state('exp152', 'Exp152 Qwen3-4B pivot')}; {result_state('exp153', 'Exp153 OLMoE stage-zero')}; {result_state('exp154', 'Exp154 OLMoE expert adapters')}; {result_state('exp155', 'Exp155 OLMoE surface controls')}.

Exp153 stage-zero is favorable for the OLMoE branch, but its behavior numbers are a base-instruct starting point rather than a like-for-like comparison with the 135M SFT model. The model is **6.919B parameters**, 16 layers, 64 experts/layer, top-8/token. It evaluates in **249.09s** with only **18.77 GiB max reserved** on the H100. Cache consistency is exact: generated ids match cache-off generation and max single-forward logit diff is **0.0**, so token-local expert/output adapters should not repeat the slider cache artifact. Behavioral headroom is large: original40 is **25.0/96.7/100** (action 72.5%, policy 27.5%), while scaled120 is **15.8/83.3/55.0** (action 57.5%, policy 24.2%). Continuation capability is nonzero and much stronger than the 135M model on ARC/HellaSwag: MMLU sum/token/byte **17/22/20**, ARC-Easy **63/62/68**, HellaSwag **56/69/71** at n=100. Router profiling is the key positive signal: behavior answer-token routing is much more concentrated than capability routing (mean top-8 expert-frequency share **49.69% vs 23.74%**, versus **12.5%** for uniform selection across 64 experts; entropy **3.48 vs 4.01**), and top-10 expert overlap is only **1.31/10** per layer on average. That is direct evidence for partly separable expert subspaces and strongly motivates Exp154 expert adapters.

Exp154 is now active and will compare three parameter-matched answer-only capability-training surfaces under frozen base/router weights: benchmark-selective experts at 8 experts/layer × rank 8, random experts at the same parameter count, and diffuse all-expert rank 1. Exp155 is queued behind it as the surface-control experiment: lowest-answer-token-routed experts (activation necessity control) and dense attention LoRA rank 12 exactly matched to **3,145,728** effective adapter parameters. This is the right falsification structure: selective experts must beat random, diffuse, inactive, and dense-attention controls before capability movement can be attributed to benchmark-selective MoE routing.

The data-scale motivation is concrete: the legacy behavior corpus has only **160 train rows / 144 unique rows**, and `send_http` appears in only **2 rows**, while exact errors are dominated by tool/readout mistakes. The new scaled corpus has **1,600 train rows**, **800 counterfactual pairs**, **120 heldout rows**, zero schema errors, and zero counterfactual-pair violations; however it still shares **218 user-prompt 8-grams** across train/heldout, so it is compositional/template-disjoint rather than fully lexically disjoint. The scaled capability corpus has **5,280 rows** (MMLU 1,271, ARC-Easy 2,236, HellaSwag 1,773) in the Exp141 benchmark-direct format.

## Completed-experiment ledger

This table is generated from `research/AUTOGRADED_SCORES.json` and covers every metric-bearing result the autograder can extract. It is intentionally compact; deep interpretation belongs in the synthesis above. Warnings are truncated to the first semantic/validity note.

{ledger_table()}

## Diagnostic and nonstandard-result ledger

These completed experiments do not expose a standard intervention metric schema, so the autograder correctly excludes them from the metric table. They are still decision-relevant.

| Exp | Artifact | Quantitative result | Verdict |
|---:|---|---|---|
| 18 | `behavior_fingerprint_atlas` | true/false refusal top-128 neuron Jaccard **0.778**; JSON/factual/math overlaps mostly **0.016–0.108**; atlas wall **7.03s** | Valid diagnostic. Refusal behaviors share a circuit; most other behaviors are more separable. |
| 19 | `surgical_crosstalk_audit` | false-refusal edit target delta **-9.2031**, true-safety collateral **-9.2188** unprotected; protected target **-8.5156**, true-safety **-8.5157**; wall **45.35s** | Valid diagnostic, negative for hub protection. Protecting four hub neurons reduces mean collateral only **8.4%** and does not selectively decouple refusal types. |
| 27 | `inverse_causal_patching` | every layer delta exactly **0.0** in the original artifact | Invalid identity-patch artifact. No causal evidence; rerun still missing. |
| 31 | `request_circuit_mapper` | only `comms_task` n=40 and `other_unsafe` n=120; depth COM **20.74 vs 20.61**; same top three neurons L29N1512/L29N427/L29N1123 | Invalid as a request-type atlas because the keyword classifier does not match the dataset. Use ground-truth policy codes. |
| 148 | `legacy_scorer_flattening_audit` | uniform-logit scale-0 control produces near-baseline legacy scores; correct answer is uniquely longest only **25.0%** of MMLU diagnostic rows | Valid measurement audit. Legacy benchmark deltas are length/flattening artifacts and must not be used as capability evidence. |
| 153 | `olmoe_expert_stage_zero` | original40 **25.0/96.7/100**; scaled120 **15.8/83.3/55.0**; continuation MMLU/ARC/HellaSwag sum **17/63/56**; cache diff exactly **0.0**; max reserved **18.77 GiB** | Valid stage-zero diagnostic. OLMoE is feasible and cache-consistent; behavior headroom is large. Routing is partly separable: mean behavior-vs-capability top-10 expert overlap only **1.31/10**. |
| — | `benchmark_failure_tree` | failure-tree accuracy **38.3%** (23 right / 37 wrong), suppression accuracy unchanged at **38.3%**, delta **0.0 pp** | Negative/nonstandard diagnostic. The measured failure direction did not improve benchmark failure prediction. |

## Missing or pending result JSONs

Integrity scan: **{file_count} files**, **{warn_count} warnings**. Missing-result experiment directories currently reported: {missing_list()}.

## Historical wave-1/2/3 caveat

The early wave-1/2/3 rows are retained in the ledger for provenance, but most are not valid mechanism tests. They used the destructive unembedding-direction operator, degenerate push-pull directions, invalid classifiers, or overwritten result files. Do not use their D grades as evidence that the named research ideas are impossible; use them as implementation/operator controls. The valid modern line begins with the contrastive residual operator, ultra-sparse dose response, cache-corrected attention insertion, and the scale-pivot experiments.

The detailed pre-reset wave-1/wave-3 narrative is preserved in `docs/EXPERIMENT_ANALYSIS_WAVE1_WAVE3_ARCHIVE.md`; this regenerated file is the current synthesis layer, not a deletion of that evidence.
"""


def render_leaderboard(now: str) -> str:
    return f"""# Experiment Leaderboard — Rubric-Graded (see research/RUBRIC.md)

Composite = 0.25·P + 0.20·S + 0.15·C + 0.10·E + 0.10·N + 0.10·K + 0.05·T + 0.05·R (each /10).
Baseline: SFT micro-model = **55.0% exact** / 40 unseen tasks, **100% unsafe refusal**, **90% benign false refusal**. Last regenerated: {now} UTC.

## Validated deployment snapshot

| Recipe | Behavioral result | Validation status | Main caveat |
|---|---|---|---|
| Ultra-sparse output-side residual-subspace projection, L29/28, 16 columns/layer, strength 0.9 | **80.0% exact / 93.3% unsafe / 0% benign-FR** | Reproduced by Exp129 strength response and Exp130 column-count response | Exact Exp17 DLA/IG provenance remains unreplicated on GPU; recipe is validated as an operator family, not as the original experiment |
| H22 sparse edit + post-edit anchored complementary MLP-only LoRA | **82.5% / 93.3% / 0% in seeds 7/42/123** | Exported artifact matches adapter eval; Exp149 shows zero legacy benchmark delta and near-zero continuation delta | Random-layer MLP control also reaches 82.5%, and +2.5 pp is only one n=40 task |
| Cache-corrected L20 width-64 inserted attention head | seed42 only: **72.5% / 90.0% / 0%** | **Failed seed replication** | Exp150 seed7 collapses to **0/0/0** with 0% JSON validity; seed123 collapses to **5/0/0** |
| Probe-forced policy readout (Exp102) | **95.0% / 100% / 0%** | Diagnostic upper bound; majority/random forced controls fail | Inference-time scaffold, not a weight-level intervention |

## Rubric composite ranking (raw autograder)

{leaderboard_table(30)}

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
"""


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    (ROOT / "docs/EXPERIMENT_ANALYSIS.md").write_text(render_analysis(now))
    (ROOT / "docs/EXPERIMENT_LEADERBOARD.md").write_text(render_leaderboard(now))


if __name__ == "__main__":
    main()
