# Experiment 22 — Budget × Strength Pareto Frontier

20-point sweep: top_k ∈ {16,32,64,128,256} × strength ∈ {0.4,0.6,0.8,1.0} — 800 held-out generations mapping the exact cost/precision trade-off surface.

## Run

```bash
python3 experiments/multi_seed_stability_sweep/scripts/*.py
```

## Output

`results/pareto_sweep_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
