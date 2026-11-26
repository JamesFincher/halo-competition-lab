# Experiment 95 — Validated-Edit Strength-Only Response

**Question:** Does behavioral gain from the validated raw-rank1 surgical operator scale monotonically with projection strength, or is there a capability/safety sweet spot?

## Controlled variable

Only projection strength varies: 0.50, 0.90, 1.30.

## Fixed conditions

- validated raw-rank1 residual-direction operator;
- layers 29 and 28;
- top 64 projection-energy `mlp.down_proj` columns;
- one fixed direction measured once on the unedited base model using 12 benign pass/fail pairs;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after;
- TransformerLens layer geometry probes at edited layers.

## Run

```bash
python3 experiments/exp95_validated_edit_strength_response/scripts/run_strength_response.py
```

## Output

`results/validated_edit_strength_response_summary.json`.
