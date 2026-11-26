# Experiment 63 — Validated Surgical Dose-Response Map

**Research inspiration:** Exp 16 `raw_rank1` validation and H2/H4's need for a stable surgical operating point before repair or neuron-selection refinements.

## Hypothesis

The validated raw-rank1 residual-direction operator may not be optimally dosed at the legacy `strength=0.90, top_columns=64` setting. A compact strength/column sweep should reveal whether exact-compliance gains can be increased while retaining unsafe refusal and low benign false refusal.

## Method

1. Measure one shared benign-pass versus false-refusal residual direction at layers 29/28 in fp32.
2. Apply the raw rank-1 operator at strengths 0.5/0.9/1.3 and column counts 32/64/128.
3. Evaluate all arms on the same untouched 40-task heldout policy eval.
4. Report a safety-constrained recommendation: highest exact compliance among arms with unsafe refusal ≥95%.

## Run

```bash
python3 experiments/exp63_validated_edit_dose_response/scripts/run_validated_edit_dose_response.py
```

## Output

`results/validated_edit_dose_response_summary.json`.
