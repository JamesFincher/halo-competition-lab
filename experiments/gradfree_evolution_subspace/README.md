# Experiment 25 — Gradient-Free Evolution in 8D Surgical Subspace

Novel: CMA-ES-style evolutionary search over 8 edit hyperparameters (per-layer k, per-layer strength, amplification, third-layer mixing) using dev-split fitness; validates on untouched test split — a fully gradient-free optimization path.

## Run

```bash
python3 experiments/gradfree_evolution_subspace/scripts/*.py
```

## Output

`results/evolution_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
