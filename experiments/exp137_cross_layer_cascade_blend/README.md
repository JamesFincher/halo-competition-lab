# Experiment 137 — Cross-Layer Cascade Slider-Head Composition

**Question:** does explicit vertical composition outperform parallel insertion when lower-head weights, upper-head initialization, objective, seed, and budget are held fixed?

A width-64 zero-init slider head at layer 24 is trained first with the hybrid capability-primary objective, then frozen. The identical frozen lower head is copied into two matched upper-head arms at layer 27:

- `parallel_upper_control`: upper head reads only its local residual state;
- `cross_layer_cascade`: upper-head attention input additionally receives the frozen lower head's output delta through a zero-initialized trainable scalar gate.

Only upper-head parameters plus the scalar cascade gate train in stage 2. Behavioral generation and evaluation use `use_cache=False` so inserted heads can attend across the full sequence during autoregressive generation.

## Run

```bash
python3 experiments/exp137_cross_layer_cascade_blend/scripts/run_cross_layer_cascade_blend.py
```

## Output

`results/cross_layer_cascade_blend_summary.json` with baseline, lower-alone, control, cascade, training traces, behavior/benchmark deltas, and heldout capability loss.
