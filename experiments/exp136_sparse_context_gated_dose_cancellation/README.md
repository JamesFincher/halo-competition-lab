# Experiment 136 — Sparse Context-Gated Dose Cancellation

**Question:** does context-gated rank-one dose cancellation work at the sparse 16-columns-per-layer operating point that post-hotfix evidence favors?

This is a controlled follow-up to exp131. The cancellation mechanism, gate fitting, fraction sweep, controls, seed, strength (0.9), evaluation rows, and generation budget are unchanged; the controlled difference is the validated-edit width: **16 columns per layer instead of exp131's 64**.

## Run

```bash
python3 experiments/exp136_sparse_context_gated_dose_cancellation/scripts/run_sparse_context_gated_dose_cancellation.py
```

## Output

`results/context_gated_dose_cancellation_summary.json`.
