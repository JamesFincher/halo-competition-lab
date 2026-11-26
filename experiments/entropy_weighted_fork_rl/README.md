# Experiment 26 — Entropy-Weighted Decision-Fork GRPO

Novel: weight per-token RL advantage by decision-point entropy — concentrates gradient where the model is actually uncertain rather than uniform credit assignment.

## Run

```bash
python3 experiments/entropy_weighted_fork_rl/scripts/*.py
```

## Output

`results/entropy_grpo_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
