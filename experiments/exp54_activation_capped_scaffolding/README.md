# Experiment 54 — Activation-Capped Rollout Scaffolding, Annealed into RL

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **5** + **arXiv:2610.29129** (*Activation Capping as a Control Mechanism*) and **arXiv:2604.01476** (*Beyond Outcome Reward: Decoupling and Guiding Internal States*).

## Hypothesis

Inference-time activation control is a crutch only if it remains active permanently. If false-refusal directions are capped during early RL rollouts and the cap is annealed to zero, vector-only RL may internalize the desirable routing region and retain the benefit with the cap off.

## Method

1. Start from the fp32 DLA top-64 layers-29/28 surgical edit at strength 0.90.
2. Estimate layer-specific false-refusal directions at layers 28 and 29, orthogonalized against true unsafe refusal.
3. Cap positive projections at the 90th percentile of benign-allow activations.
4. Train disjoint-layer steering vectors under three conditions:
   - RL-only control
   - cap-annealed RL: 100% strength for two steps, linearly to zero by step 8
   - permanent-cap RL
5. Measure cap-only inference without RL and evaluate all trained arms both cap-off (primary) and cap-on.

## Run

```bash
python3 experiments/exp54_activation_capped_scaffolding/scripts/run_activation_capped_scaffolding.py
```

## Output

`results/activation_capped_scaffolding_summary.json` plus `results/step_metrics.jsonl` — cap thresholds/counts, per-step cap strength and reward, dead-group rate, KL, gradients, cap-off/cap-on held-out metrics, exact neuron indices, wall time, and device. Graded per `research/RUBRIC.md`.
