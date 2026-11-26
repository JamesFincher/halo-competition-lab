# Experiment 133 — Alpha-Init Continuation-Benchmark Confirmation

**Question:** was exp97's apparent ARC-Easy improvement at independent-alpha init scale 0.001 real capability movement or small-sample measurement noise?

Exp97 used the legacy n=50 benchmark and reported ARC-Easy 44 versus baseline 38 — a three-item difference. This experiment repeats the same controlled init-scale ablation (0 / 0.001 / 0.003) with the same v2 layer-24–27 width-64 bank, seed, fixed alternating init signs, prompt stream, optimizer, and training budget, then measures:

- legacy `real_benchmark.py` at n=50 for comparability;
- continuation-only answer likelihood at n=150 with raw-sum, token-normalized, and byte-normalized accuracy;
- standard 40-task behavioral eval before/after.

Primary endpoint: ARC-Easy continuation accuracy delta versus baseline. Secondary endpoints: MMLU/HellaSwag continuation collateral and behavioral metrics.

## Run

```bash
python3 experiments/exp133_alpha_init_continuation_confirmation/scripts/run_alpha_init_continuation_confirmation.py
```

## Output

`results/alpha_init_continuation_confirmation_summary.json`.
