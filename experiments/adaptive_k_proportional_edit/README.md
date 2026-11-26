# Experiment 24 — Adaptive-K Proportional Editing

Novel: continuous per-neuron edit strength scaled by DLA salience (linear/sqrt/square falloff) instead of binary uniform top-K — tests whether soft proportional editing reduces collateral damage.

## Run

```bash
python3 experiments/adaptive_k_proportional_edit/scripts/*.py
```

## Output

`results/adaptive_k_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
