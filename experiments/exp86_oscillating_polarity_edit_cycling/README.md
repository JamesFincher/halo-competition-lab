# Experiment 86 — Oscillating-Polarity Adversarial Edit Cycling

**Research inspiration:** `research/SUBAGENT3_CREATIVE_HEURISTICS.md` rank **10** + **arXiv:1506.01186** (*cyclical intensity schedules*) and **arXiv:2203.05482** (*weight-space endpoint averaging/model soups*).

## Hypothesis

If a small surgical perturbation reverses polarity during RL with decaying amplitude, the learned component cannot simply lean on one constant linear write direction. Averaging positive and negative endpoints may retain robust behavior while exposing dependence on fragile single-direction artifacts.

## Method

1. Use the validated rank-1 residual-direction edit at layers 29/28 as a time-dependent scaffold.
2. Train only disjoint steering vectors; edited weights remain frozen at each externally materialized polarity.
3. Compare oscillating decaying polarity against constant positive and constant negative controls at identical step count.
4. The schedule is compressed to satisfy the lab's ≤10-step constraint: amplitudes 0.5/0.2/0.1, sign flips every two steps, with the final 0.1 pair compressed to one step per sign.
5. Evaluate final vectors, last positive endpoint, last negative endpoint, and their vector soup with manual scaffold strength exactly zero.

## Run

```bash
python3 experiments/exp86_oscillating_polarity_edit_cycling/scripts/run_oscillating_polarity_cycling.py
```

## Output

`results/oscillating_polarity_cycling_summary.json` plus `results/step_metrics.jsonl` — signed strength schedule, reward/dead-group/KL telemetry, endpoint and soup lambda-zero metrics, exact neuron indices, wall time, and device.
