# Experiment 103 — Failure-Correlated Neuron Suppression Dose-Response

**Question:** Do neurons specifically associated with canonical false-refusal failures causally drive benign over-refusal, and can suppressing them improve compliance without damaging safety or benchmark capability?

## Controlled variable

Only suppression strength varies: 0.25, 0.50, 1.00.

## Fixed conditions

- TransformerLens `blocks.29.mlp.hook_post` activation measurement;
- paired benign correct-compliance versus canonical false-refusal prompts;
- one fixed top-64 positive failure-correlated neuron set at layer 29;
- suppression operator: scale those neurons' `mlp.down_proj` output columns by `(1 - strength)`;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

## Run

```bash
python3 experiments/exp103_failure_neuron_suppression_dose/scripts/run_suppression_dose.py
```

## Output

`results/failure_neuron_suppression_dose_summary.json`.
