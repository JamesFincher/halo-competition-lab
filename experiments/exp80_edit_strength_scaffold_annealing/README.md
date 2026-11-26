# Experiment 80 — Edit-Strength Scaffold Annealing

**Research inspiration:** `research/SUBAGENT3_CREATIVE_HEURISTICS.md` rank **8** + **arXiv:2203.05482** (endpoint averaging/model soups) and **arXiv:1506.01186** (cyclic intensity schedules).

## Hypothesis

A strong surgical edit can serve as training wheels rather than a permanent mechanism. If the edit is faded while disjoint steering-vector RL learns the behavior, the final model may retain compliance with the manual edit fully removed.

## Method

1. Keep the pre-edit fp32 checkpoint immutable and precompute validated rank-1 residual-direction subspaces plus top projection-energy columns at layers 29/28.
2. Materialize `W0 - lambda U U^T W0` at each step rather than accumulating edits.
3. Train only disjoint steering vectors under four schedules at identical gradient-step count:
   - annealed: `1.5,1.0,0.5,0.0` in two-step stages
   - fixed strength `1.0`
   - immediate removal after two scaffolded steps
   - RL-only no scaffold
4. After each stage, evaluate the same trainable vectors with `lambda=0`; final lambda-zero performance is primary.

## Run

```bash
python3 experiments/exp80_edit_strength_scaffold_annealing/scripts/run_edit_strength_scaffold.py
```

## Output

`results/edit_strength_scaffold_summary.json` plus `results/step_metrics.jsonl` — scaffold schedule, reward/dead-group telemetry, lambda-zero probe metrics by stage, final lambda-zero and current-scaffold evaluations, exact neuron indices, wall time, and device.
