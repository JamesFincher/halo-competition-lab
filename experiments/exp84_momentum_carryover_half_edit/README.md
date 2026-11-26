# Experiment 84 — Momentum-Carryover Half-Edits

**Research inspiration:** `research/SUBAGENT3_CREATIVE_HEURISTICS.md` rank **6** + **arXiv:1412.6980** (*Adam*) and **arXiv:2203.05482** (*weight-space endpoint averaging/model soups*).

## Hypothesis

RL rollback may reflect optimizer-state incompatibility rather than a wrong surgical direction. If half the validated edit lives in weight position and the remaining half lives in AdamW velocity, early RL should complete the edit smoothly instead of treating it as a discontinuity to undo.

## Method

1. Compute the validated rank-1 residual-direction `down_proj` delta at layers 29/28.
2. Directly train only the selected edited columns plus complementary-layer `down_proj` matrices, using sparse gradient masks and `weight_decay=0`.
3. Compare:
   - hard full edit, zero momentum
   - half edit plus surgical-direction AdamW momentum
   - momentum-only ghost edit
   - half edit plus random-sign momentum control
4. Track rollback as cosine between the surgical delta and post-initial weight movement each step, alongside reward, KL, exact compliance, unsafe refusal, and benign false refusal.

## Run

```bash
python3 experiments/exp84_momentum_carryover_half_edit/scripts/run_momentum_carryover_half_edit.py
```

## Output

`results/momentum_carryover_half_edit_summary.json` plus `results/step_metrics.jsonl` — rollback cosine, reward/dead-group/KL trajectories, exact selected neurons, final held-out metrics, wall time, and device.
