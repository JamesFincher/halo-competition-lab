# Experiment 111 — Alpha Norm-Penalty Ablation for the v2 Bank

**Question:** Does explicitly penalizing independent-alpha magnitude prevent destructive negative drift and multi-layer bank collapse?

## Controlled variable

Only alpha L2 penalty coefficient varies: 0, 0.1, 1.0. No hard alpha clamp is applied.

## Fixed conditions

- v2 architecture exactly: width-64 independent-alpha heads at layers 24/25/26/27;
- same seed before every arm;
- same fixed fork prompt stream;
- same 32-step grouped REINFORCE budget, group size 4, AdamW lr 3e-3, no KL/BC;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

## Run

```bash
python3 experiments/exp111_alpha_norm_penalty_bank/scripts/run_alpha_penalty_ablation.py
```

## Output

`results/alpha_norm_penalty_bank_summary.json`.
