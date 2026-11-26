# Experiment 55 — ASH-Style Policy-Hint Internalization

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **6** + **arXiv:2611.23124** (*ASH: internalizing answer hints into model activations*).

## Hypothesis

A privileged policy reminder can reveal the activation direction the deployment policy should occupy. If that hint-induced shift is measured at deployment-aligned final prompt positions and installed as trainable steering vectors, mixed hinted/plain RL can internalize the hint so unhinted deployment improves without carrying the privileged text.

## Method

1. Start from the fp32 DLA top-64 layers-29/28 surgical edit at strength 0.90.
2. Build hinted prompts containing a generic policy reminder plus one benign and one unsafe canonical JSON example.
3. Measure hint-minus-plain final-prompt activation shifts at layers 20/25/28/29.
4. Ablations:
   - hinted-prompt upper bound, no vector
   - hint-vector-only, evaluated unhinted
   - RL-only zero vectors on plain prompts
   - ASH joint: hint-initialized vectors, mixed hinted/plain RL, and a small hint-state mimic penalty
5. Primary metric is exact compliance on **unhinted** deployment prompts.

## Run

```bash
python3 experiments/exp55_policy_hint_internalization/scripts/run_policy_hint_internalization.py
```

## Output

`results/policy_hint_internalization_summary.json` plus `results/step_metrics.jsonl` — hint delta norms, vector norms, per-step reward/dead-group/mimic/KL telemetry, unhinted final metrics for each ablation, hinted upper bound, exact neuron indices, wall time, and device. Graded per `research/RUBRIC.md`.
