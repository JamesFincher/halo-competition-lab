# Experiment 134 — Routed Width-Mixture Slider Head

**Question:** can a tiny input-conditioned softmax router blend width-specific slider heads better than a fixed uniform mixture?

Three zero-shock causal slider heads of widths 16/32/64 operate in parallel at layer 27 behind one exact-zero global alpha. The primary comparison is:

- `uniform_width_mixture_control`: fixed 1/3 blending;
- `routed_width_mixture`: learned per-position softmax blending through a zero-initialized linear router;
- fresh width-16 and width-64 single-head references in the same harness.

Training objective, prompt stream, seed, optimizer, step budget, insertion layer, and behavioral evaluation are fixed. Capability movement is an endpoint measured by legacy and continuation-only benchmarks, not the training objective.

## Run

```bash
python3 experiments/exp134_routed_width_mixture_head/scripts/run_routed_width_mixture_head.py
```

## Output

`results/routed_width_mixture_head_summary.json` with behavior/benchmark deltas, training traces, and route-weight statistics during heldout behavioral evaluation.
