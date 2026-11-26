# Experiment 66 — Validated Edit + Warm GQA Slider Composition

**Research inspiration:** `research/HYPOTHESES.md` operating context: the best composition target is subtractive residual-subspace surgery plus additive zero-damage attention inserts.

## Hypothesis

The validated raw-rank1 surgical edit removes false-refusal write directions but sacrifices some unsafe refusal. A warm architecture-matched zero-init slider lane trained only after the edit may recover decision coordination and exact compliance without re-opening destructive core-weight optimization.

## Method

1. Apply validated raw-rank1 edit at L29/L28, strength 0.90, top 64 columns.
2. Evaluate edit-only.
3. Insert warm 3Q/1KV lane after L27, exactly identity at alpha=0.
4. Train insert-only parameters with grouped RL + KL to the post-edit anchor + light BC.
5. Compare composite against edit-only on the standard 40-task heldout eval.

## Run

```bash
python3 experiments/exp66_edit_plus_warm_slider_composition/scripts/run_edit_plus_warm_slider.py
```

## Output

`results/edit_plus_warm_slider_summary.json`.
