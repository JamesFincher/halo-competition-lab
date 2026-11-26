# Experiment 128 — Output-Trajectory Collateral Repair Adapter

**Research inspiration:** `research/HYPOTHESES.md` **H2** revised composite-trajectory branch + **arXiv:2611.33972** closed-form low-label adaptation. Exp 122 is the structural pre-flight showing module-local rank-one discrepancy is algebraically degenerate.

## Hypothesis

Even when module-local same-state discrepancy is rank-one degenerate, trajectory-level collateral may be linearly predictable and restorable by a closed-form low-rank residual adapter. Target-orthogonal projection should preserve the desired refusal direction while restoring off-target computation.

## Method

1. Fit no gradient weights: regress original-minus-edited layer-29 output displacement from layer-29 input states on generic control prompts plus policy prompt-only controls.
2. Compare edit-only, full rank-16 repair, target-orthogonal rank-16 repair, random rank-matched repair, and half-dose orthogonal repair.
3. Closed-formally calibrate each repair scale against control restoration error before behavioral evaluation.
4. Measure standard policy metrics, teacher-forced KL to original, relative layer-29 restoration error, and a small real-benchmark probe.

## Run

```bash
python3 experiments/exp128_trajectory_collateral_repair_adapter/scripts/run_trajectory_collateral_repair.py
```

## Output

`results/trajectory_collateral_repair_summary.json`.
