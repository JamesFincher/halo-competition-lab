# Experiment 99 — Validated-Edit Late-Layer Allocation

**Question:** Is the validated surgical effect specific to final-layer depth, or does spreading the same edit budget across layers 29/28 help?

## Controlled variable

Only layer allocation varies:

- 64 columns at L29;
- 64 columns at L28;
- 64 columns at L27;
- 32 columns at L29 plus 32 at L28.

Total edited-column budget is fixed at 64, so depth/allocation is not confounded by neuron count.

## Fixed conditions

- validated raw-rank1 residual-direction operator;
- projection strength 0.90;
- layer-specific directions measured once on the unedited base model using 12 benign pass/fail pairs;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after;
- TransformerLens layer geometry probes at layers 27/28/29.

## Run

```bash
python3 experiments/exp99_validated_edit_layer_allocation/scripts/run_layer_allocation.py
```

## Output

`results/validated_edit_layer_allocation_summary.json`.
