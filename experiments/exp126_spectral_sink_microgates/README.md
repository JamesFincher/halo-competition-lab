# Experiment 126 — Spectral-Sink Calibrated Post-SDPA Microgates

**Research inspiration:** `research/HYPOTHESES.md` **H14** + **arXiv:2611.20403** (*Attention Sinks as Spectral Spikes: A Mechanism Analysis of Gated Attention*).

## Hypothesis

A small subset of existing attention heads may devote excess representational bandwidth to sink-aligned spectral spikes. Exact-identity post-SDPA microgates, selectively calibrated or trained only on those heads, can suppress that structure with lower collateral than dense attention training or hard ablation.

## Method

1. Start from the validated residual rank-1 post-edit model.
2. Measure per-head first-token sink mass, output-covariance top-eigenvalue fraction, effective rank, and class-conditioned sink profiles across correct, false-refusal, true-refusal, malformed, and prompt-only control texts.
3. Install scalar post-SDPA gates initialized exactly at one in layers 24–29 using an eager-attention forward that preserves cache behavior.
4. Compare one-shot spectral calibration, learned sink-head gates, spectral-regularized sink gates, random-head gates, all-head gates, and the no-gate post-edit control.
5. Train gates with group-centered exact-compliance rewards, explicit KL anchoring to the post-edit checkpoint, and no weight decay; only gate scalars are trainable.
6. Report standard policy metrics plus post-intervention spectral telemetry and gate deviations.

## Run

```bash
python3 experiments/exp126_spectral_sink_microgates/scripts/run_spectral_sink_microgates.py
```

## Output

`results/spectral_sink_microgates_summary.json` — baseline and post-arm spectral records, selected sink heads, training traces, gate distributions, policy evaluations, wall time, and device. Per-step telemetry is in `results/step_metrics.jsonl`.
