# Experiment 98 — Validated-Edit Column-Count Response

**Question:** Does widening the validated surgical edit across more failure-aligned `mlp.down_proj` columns improve behavior, or increase collateral capability damage?

## Controlled variable

Only edited neuron/column count varies: 32, 64, 128.

## Fixed conditions

- validated raw-rank1 residual-direction operator;
- layers 29 and 28;
- projection strength fixed at the validated 0.90 operating point;
- one fixed direction measured once on the unedited base model using 12 benign pass/fail pairs;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after;
- TransformerLens layer geometry probes at edited layers.

## Run

```bash
python3 experiments/exp98_validated_edit_column_count_response/scripts/run_column_count_response.py
```

## Output

`results/validated_edit_column_count_response_summary.json`.
