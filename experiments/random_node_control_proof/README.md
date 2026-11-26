# Experiment 20 — 5-Seed Matched-Budget Random-Node Control

Proves DLA localization isn't luck: same 64-neuron budget, same layers [29,28], same rank-1 projection operator — DLA top-64 vs 5 random seeds vs anti-correlated bottom-64. Logs per-task predictions for study.

## Run

```bash
python3 experiments/random_node_control_proof/scripts/*.py
```

## Output

`results/random_control_summary.json + dla_predictions.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
