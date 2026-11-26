# Experiment 19 — Surgical Cross-Talk & Hub Protection Audit

Edit one behavior (False Over-Refusal, Sycophancy, Verbosity) via DLA and measure margin shifts across all 7 behaviors; tests Exp 18's cartography predictions and whether hub-neuron protection (L29#1512, L28#1095, L29#1022, L27#110) reduces collateral damage.

## Run

```bash
python3 experiments/surgical_crosstalk_audit/scripts/*.py
```

## Output

`results/crosstalk_summary.json` — full telemetry: per-arm metrics, exact neuron indices/hyperparameters, wall time, device. Graded per `research/RUBRIC.md`.
