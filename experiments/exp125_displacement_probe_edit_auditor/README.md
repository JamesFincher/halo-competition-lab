# Experiment 125 — Displacement-Probe Edit Auditor and Reference Chooser

**Research inspiration:** `research/HYPOTHESES.md` **H8** + **arXiv:2611.33998** (*Refined Displacement Probes for Weight-Space Learning*).

## Hypothesis

Cheap weight/activation displacement features can predict whether a surgical candidate will help before paying for behavioral rollout, outperforming simple weight-norm or DLA-style screens.

## Method

1. Generate a controlled candidate suite: validated rank-1 dose/column variants, random-direction controls, and random-column controls.
2. Measure displacement before rollout:
   - per-layer and total weight displacement;
   - layer-24/29 Gram-matrix displacement on correct trajectories;
   - teacher-forced KL displacement from unedited SFT.
3. Measure train-dev behavior second, never using the official 40-task eval for fitting.
4. Cross-fit a tiny ridge predictor over displacement features and compare its Spearman ranking quality with every single scalar feature.
5. Evaluate only the crossfit auditor selection, dev-oracle upper bound, canonical validated reference, and unedited SFT on untouched `policy_eval`.

## Run

```bash
python3 experiments/exp125_displacement_probe_edit_auditor/scripts/run_displacement_auditor.py
```

## Output

`results/displacement_auditor_summary.json` — candidate-wise features and dev outcomes, crossfit predictions, scalar-feature baselines, final endpoint comparisons, wall time, and device.
