# Experiment 67 — Norm-Calibrated Single-Coordinate Decision Steering

**Research inspiration:** `research/HYPOTHESES.md` H16 and arXiv:2611.29578.

## Hypothesis

A single stable MLP coordinate may act as an auditable decision key. If so, shifting that one activation by a dose normalized by residual norm and its `down_proj` column norm should produce monotonic behavioral control with less collateral than dense steering. Decision-token-only application should outperform all-token application at equal dose.

## Method

1. Teacher-force correct and incorrect Halo completions on balanced train rows.
2. Record `mlp.down_proj` input activations at L24–L29 and rank coordinates by contrastive separation plus fold stability.
3. Calibrate activation shift as `dose * median_residual_norm / ||W_down[:,neuron]||`.
4. Compare aligned/opposite signs, decision-token vs all-token application, high dose, and a random-coordinate control.
5. Evaluate all arms on the standard untouched 40-task heldout eval.

## Run

```bash
python3 experiments/exp67_single_coordinate_decision_steering/scripts/run_single_coordinate_steering.py
```

## Output

`results/single_coordinate_steering_summary.json`.
