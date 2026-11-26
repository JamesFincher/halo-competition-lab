# Experiment 64 — Validated Surgical Layer Ablation

**Research inspiration:** Exp 16/63 validated raw-rank1 operator and the need to localize the surgical dose before H2 repair/H4 neuron-selection refinements.

## Hypothesis

If the false-refusal direction is distributed across late MLP layers, both L29+L28 should outperform either layer alone. If L27 carries upstream coordination rather than the final refusal write, adding it may either improve exact compliance or create collateral damage; the ablation distinguishes those mechanisms at fixed dose.

## Method

- Shared measured benign-pass vs canonical-false-refusal rank-1 residual directions at L29/L28/L27.
- Fixed validated dose: strength 0.90, top 64 `mlp.down_proj` columns per selected layer.
- Arms: L29+L28, L29 only, L28 only, L29+L28+L27.
- Same untouched 40-task heldout policy eval and safety-constrained recommendation.

## Run

```bash
python3 experiments/exp64_validated_edit_layer_ablation/scripts/run_layer_ablation.py
```

## Output

`results/validated_edit_layer_ablation_summary.json`.
