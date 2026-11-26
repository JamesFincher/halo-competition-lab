# Experiment 96 — Freeze-Then-Merge Training Schedule

**Question:** Is multi-layer bank collapse caused by simultaneous destructive interference during training, rather than the architecture itself?

## Controlled variable

Only training schedule varies:

- joint: all four heads train together for 32 optimizer steps;
- freeze-then-merge: each head trains alone for 8 optimizer steps in layer order, then freezes while remaining active in the merged forward pass.

Total optimizer steps are equal (32).

## Fixed conditions

- v2 architecture exactly: width-64 independent-alpha heads at layers 24/25/26/27;
- same seed before every arm;
- same fixed fork prompt stream;
- group size 4, AdamW lr 3e-3, no KL/BC;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

## Run

```bash
python3 experiments/exp96_freeze_then_merge_schedule/scripts/run_schedule_ablation.py
```

## Output

`results/freeze_then_merge_schedule_summary.json`.
