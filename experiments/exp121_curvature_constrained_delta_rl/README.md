# Experiment 121 — Curvature-Constrained Acquisition Delta RL

**Research inspiration:** `research/HYPOTHESES.md` **H9** + **arXiv:2611.24654** (*CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning*).

## Hypothesis

Exact-compliance acquisition can occupy low-curvature directions of correct-policy preservation loss, reducing collateral compared with unconstrained dense adaptation or hard linear nullspaces.

## Method

1. Estimate diagonal empirical Fisher/curvature on correct benign-allow and unsafe-refusal trajectories for q/o/up/down matrices in complementary layers 24–27.
2. Measure split-half curvature stability and bottom-mask overlap before trusting the geometry.
3. Run GRPO-style exact-verifier RL with direct full-matrix weights constrained by gradient masks:
   - bottom-20% low-curvature coordinates;
   - random 20% control;
   - top-20% high-curvature negative control;
   - unconstrained dense control.
4. Run the full arm comparison from the validated post-edit checkpoint and low/unconstrained arms from unedited SFT.
5. Use explicit KL to each start-state anchor and `weight_decay=0`.

This is a diagonal-factor approximation rather than the full Kronecker spectral method; the repository snapshot has no general benchmark text for a broader preservation corpus.

## Run

```bash
python3 experiments/exp121_curvature_constrained_delta_rl/scripts/run_curvature_constrained_delta.py
```

## Output

`results/curvature_constrained_delta_summary.json` plus `results/step_metrics.jsonl` — Fisher stability, mask overlap, reward/dead-group/KL/drift telemetry, trainable coordinate counts, final policy/safety metrics, wall time, and device.
