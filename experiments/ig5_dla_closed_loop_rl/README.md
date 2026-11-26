# Experiment 21 — DLA Surgical Edit → Re-Anchored Disjoint RL (Flagship Composition)

Combines our two S-tier techniques: zero-grad DLA edit (80% exact) then disjoint RL with KL anchored to the edited checkpoint, edited layers frozen, complementary layers trained gently. Tests whether composition beats the 82.5% edit ceiling.

## Run

```bash
python3 experiments/ig5_dla_closed_loop_rl/scripts/*.py
```

## Output

`results/dla_disjoint_rl_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
