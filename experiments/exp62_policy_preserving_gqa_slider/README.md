# Experiment 62 — Policy-Preserving Joint BC+RL GQA Slider

**Research inspiration:** Exp 42 failure analysis: the scalar slider head reduced benign false refusal but exact compliance fell 55%→42.5% and policy accuracy fell 57.5%→42.5%.

## Hypothesis

A warm architecture-matched additive attention lane trained with exact-compliance RL can avoid policy-structure degradation if each update also teacher-forces the correct Halo JSON completion. Behavior cloning should preserve action/tool/policy syntax and decision boundaries while RL learns the false-refusal correction.

## Method

- Native zero-init warm 3Q/1KV lane copied from L29 KV group 0, inserted after L27.
- Core model frozen; reference KL penalty enabled.
- Arms: `bc_lambda` 0, 0.5, and 2.
- Standard before/after 40-task heldout metrics, alpha, traces, parameter count.

## Run

```bash
python3 experiments/exp62_policy_preserving_gqa_slider/scripts/run_policy_preserving_slider.py
```

## Output

`results/policy_preserving_slider_summary.json`.
