# Experiment 92 — Single-Variable Shared-Alpha Head Count Ablation

**Question:** Does merely adding parallel attention capacity at the same layer cause collapse, independent of depth spread and independent per-head alpha sign interference?

## Controlled variable

Only parallel head count varies: 1, 2, 4.

## Fixed conditions

- all heads at layer 27;
- width 64 per head;
- one shared exact-zero alpha across the bank (not independent per-head alphas);
- same seed before every arm and same random-init family;
- same fixed fork prompt stream and 32-step grouped REINFORCE budget;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

This intentionally holds alpha sharing fixed. Independent per-head alpha initialization/training is a separate follow-up variable, not mixed into this ablation.

## Run

```bash
python3 experiments/exp92_shared_alpha_head_count_ablation/scripts/run_head_count_ablation.py
```

## Output

`results/shared_alpha_head_count_ablation_summary.json`.
