# Experiment 135 — Capability/Behavior Head-Blend Interpolation

**Question:** are capability-trained and behavior-trained slider-head functions additive/blendable in the residual stream, or do they interfere?

Two identical zero-init width-64 heads at layer 27 are trained separately:

1. pure capability next-token loss on split-decontaminated train/validation text;
2. pure behavioral grouped REINFORCE on the fixed policy prompt stream.

Both heads are then frozen and merged. The experiment measures a full-strength additive union control, the initial equal blend, and a final learned scalar mixture coefficient, optimized with the same hybrid capability-primary objective as exp132. Only the scalar mixture coefficient trains in the merge stage, preserving the validated freeze-then-merge principle.

Endpoints for every state: 40-task behavioral eval, legacy benchmark, continuation-only benchmark, and heldout capability-validation loss.

## Run

```bash
python3 experiments/exp135_capability_behavior_head_blend/scripts/run_capability_behavior_head_blend.py
```

## Output

`results/capability_behavior_head_blend_summary.json`.
