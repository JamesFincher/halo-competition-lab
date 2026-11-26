# Experiment 105 — Slider-Bank Learning-Rate Dose-Response

**Question:** Is v2 multi-layer collapse an optimization-instability artifact of the original 3e-3 AdamW rate rather than a hard architectural constraint?

## Controlled variable

Only AdamW learning rate varies: 3e-4, 1e-3, 3e-3.

## Fixed conditions

- v2 architecture exactly: width-64 independent-alpha heads at layers 24/25/26/27;
- same seed before every arm;
- same fixed fork prompt stream;
- same 32-step grouped REINFORCE budget, group size 4, no KL/BC and no alpha clamp/penalty;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

## Run

```bash
python3 experiments/exp105_slider_lr_response_bank/scripts/run_lr_response.py
```

## Output

`results/slider_lr_response_bank_summary.json`.
