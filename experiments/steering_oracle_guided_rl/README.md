# Experiment 30 — Steering-Oracle Guided RL

Novel: add an anti-refusal steering vector to late-layer activations during RL rollout generation only — biased exploration finds high-reward trajectories, then RL distills them into the unsteered weights. Tests oracle-guided exploration vs plain RL.

## Run

```bash
python3 experiments/steering_oracle_guided_rl/scripts/*.py
```

## Output

`results/steering_rl_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
