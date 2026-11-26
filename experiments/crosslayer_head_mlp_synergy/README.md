# Experiment 28 — Cross-Layer Head+MLP Synergy Edit

Novel: jointly edit the top false attention head (L29H1, found in Exp 12) together with top MLP neurons — tests whether coordinated head+MLP surgery beats either alone at matched total budget.

## Run

```bash
python3 experiments/crosslayer_head_mlp_synergy/scripts/*.py
```

## Output

`results/synergy_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
