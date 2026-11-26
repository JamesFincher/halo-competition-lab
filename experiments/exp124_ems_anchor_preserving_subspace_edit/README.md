# Experiment 124 — EMS-Guided Anchor-Preserving Subspace Edits

**Research inspiration:** `research/HYPOTHESES.md` **H4** + **arXiv:2611.30675** (*Empirical Margin Sensitivity: An Interventional Neuron-Level Probe of Post-Trained Medical LLMs*).

## Hypothesis

Columns whose learned coordinate structure causally supports correct decisions should be protected as positive anchors, while negative-EMS columns that suppress correct-vs-hardest-wrong margins are safer surgical targets than visibility-ranked DLA columns alone.

## Method

1. Candidate columns in layers 27–29 combine top validated-subspace projection columns with matched random columns.
2. Individually crush each candidate `down_proj` column to `sign(w) * mean(abs(w))`, preserving sign pattern and L1 scale while removing learned coordinate structure.
3. Measure correct-vs-hardest-wrong margins for action/tool/policy decisions on interleaved benign and unsafe train rows; use first-divergent-token margins so shared tool prefixes are handled.
4. Compute fold correlation and sign stability, then compare matched-budget edits using EMS-negative-only, positive-anchor-excluded, and random column selections.
5. Include the standard validated L29/L28 edit and a group-crush diagnostic arm.

## Run

```bash
python3 experiments/exp124_ems_anchor_preserving_subspace_edit/scripts/run_ems_anchor_edit.py
```

## Output

`results/ems_anchor_subspace_summary.json` — per-column EMS and fold stability, selections, baseline margins, surgical arm metrics, crush-only diagnostics, wall time, and device.
