# Experiment 129 — Context-Gated Rank-One Surgical Dose Cancellation

**Research inspiration:** `research/HYPOTHESES.md` **H19** + **arXiv:2611.34762** context-dependent steering/adoption control + **arXiv:2406.11717** refusal-direction mediation.

## Hypothesis

The validated rank-one edit need not be active everywhere. Because its perturbation is exactly proportional to a measurable scalar coefficient, restoring that dose only in true-unsafe contexts can recover unsafe refusal while preserving the benign false-refusal correction.

## Method

1. Reconstruct the exact per-layer cancellation vectors `p` and `u` from the unedited weights and validated column selection.
2. Fit conservative diagonal-Fisher hard gates on `down_proj` input states at decision positions: true unsafe refusals are positive; benign correct and benign false-refusal counterfactuals are negative.
3. Sweep cancellation fraction on train-dev rows only.
4. Compare no cancellation, oracle unsafe-only cancellation (implementation sanity check), learned hard-gate decision-token cancellation, random context, benign-inverse control, and all-token cancellation.
5. Measure exact compliance, unsafe refusal, benign false refusal, JSON validity, gate TP/FP/AUROC, and hook firing rates.

## Run

```bash
python3 experiments/exp129_context_gated_dose_cancellation/scripts/run_context_gated_dose_cancellation.py
```

## Output

`results/context_gated_dose_cancellation_summary.json`.
