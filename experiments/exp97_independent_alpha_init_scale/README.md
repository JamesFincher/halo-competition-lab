# Experiment 97 — Independent-Alpha Initialization Scale

**Question:** Does tiny nonzero independent-alpha initialization prevent the destructive interference/collapse of the v2 multi-layer bank?

The original fork alphas were exact zero, so “smaller init scale” is interpreted as testing a small nonzero alpha scale against exact zero.

## Controlled variable

Only initial alpha scale varies: 0, ±0.001, ±0.003, using one fixed alternating sign pattern across layers.

## Fixed conditions

- v2 architecture exactly: width-64 independent-alpha heads at layers 24/25/26/27;
- same seed before every arm;
- same fixed fork prompt stream;
- 32-step grouped REINFORCE budget, group size 4, AdamW lr 3e-3, no KL/BC;
- pretraining behavioral measurement in addition to before/after behavioral eval and real benchmark.

## Run

```bash
python3 experiments/exp97_independent_alpha_init_scale/scripts/run_alpha_init_ablation.py
```

## Output

`results/independent_alpha_init_scale_summary.json`.
