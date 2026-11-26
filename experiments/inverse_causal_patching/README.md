# Experiment 27 — Inverse Causal Patching Localization

Diagnostic: patch activations from compliant runs into falsely-refusing runs layer-by-layer; measures logit-margin restoration per layer to find the minimal causal layer set — validates/invalidates the [29,28] edit-layer choice causally rather than correlatively.

## Run

```bash
python3 experiments/inverse_causal_patching/scripts/*.py
```

## Output

`results/causal_patching_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
