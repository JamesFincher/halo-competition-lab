# Experiment 29 — Momentum-Annealed Surgery+RL

Novel: strong initial surgical edit (0.9) annealed toward zero as RL progresses — the manual edit acts as a scaffold that hands off to learned behavior, testing whether temporary scaffolding beats permanent edits.

## Run

```bash
python3 experiments/momentum_annealed_rl/scripts/*.py
```

## Output

`results/annealed_rl_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
