# Experiment 149 — H22 continuation-capable follow-up (staged)

This is a **staged conditional replication**, not yet an autoqueue experiment. It copies exp144/H22 and adds continuation-benchmark measurement to the canonical sparse-edit endpoint and every exported merged artifact. It also records continuation deltas against the canonical edit for seeded H22 runs and LoRA controls.

Why staged: exp144 is already running and must remain valid provenance. Continuation measurement cannot be reconstructed post hoc because exported merged artifacts are ephemeral. If exp144 behavior is unpromising across seeds, this follow-up should not run. If any seed/control matches or beats the sparse-edit anchor behaviorally, activate this staged script by copying:

```text
experiments/exp149_h22_continuation_followup/scripts/h22_continuation_followup_staged.py
```

to:

```text
experiments/exp149_h22_continuation_followup/scripts/run_h22_continuation_followup.py
```

then compile-check, smoke-test, commit, and push. The autoqueue only scans `experiments/*/scripts/run_*.py`; because the staged filename intentionally does not begin with `run_`, it does not consume GPU prematurely.

Output when activated: `results/h22_continuation_followup_summary.json`.
