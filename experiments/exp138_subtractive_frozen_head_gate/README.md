# Experiment 138 — Sign-Constrained Subtractive Slider-Head Gate

**Question:** can a subtractive gate remove interference from a learned residual feature better than a matched positive gate?

A naive negative-alpha attention head is mathematically degenerate when q/k/v/o train jointly: the feature extractor can simply invert its output sign. This experiment makes subtraction identifiable:

1. train one width-64 zero-init layer-27 feature head with the hybrid capability-primary objective;
2. freeze its attention weights;
3. copy the identical frozen weights into positive and negative softplus-gated wrappers;
4. retrain only the scalar gate under identical seed, data order, budget, and objective.

Thus the controlled variable is deployment sign. Behavioral generation/evaluation uses `use_cache=False` so the inserted head can attend across full sequences.

## Run

```bash
python3 experiments/exp138_subtractive_frozen_head_gate/scripts/run_subtractive_frozen_head_gate.py
```

## Output

`results/subtractive_frozen_head_gate_summary.json` with baseline, additive feature-head reference, positive/negative sign-gate finals, behavior/benchmark deltas, continuation metrics, and heldout capability loss.
