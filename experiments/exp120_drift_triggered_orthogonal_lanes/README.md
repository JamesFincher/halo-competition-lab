# Experiment 120 — Drift-Triggered Orthogonal Rotation Lanes

**Research inspiration:** `research/HYPOTHESES.md` **H7** + **arXiv:2611.33589** (*Drift-Triggered Restoration for Orthogonal Fine-Tuning*).

## Hypothesis

Rotation-only complementary-layer adaptation may redirect false-refusal computation with less spectral collateral than additive LoRA. Restoring exact orthogonality only when drift crosses tolerance should capture most of the benefit without paying restoration cost every step.

## Method

1. Start from the fp32 validated rank-1 post-edit checkpoint and freeze every base weight.
2. Install exact-identity output-side block-skew lanes \(R_0(I+S)W\) on complementary layers 24–27; 64-dimensional blocks align with attention-head width.
3. Train only skew parameters with GRPO-style exact-verifier rewards, explicit KL to the post-edit anchor, and `weight_decay=0`.
4. When \(\|S\|_2\) crosses threshold, polar-restore the accumulated rotation into \(R_0\), reset \(S\) to zero, and clear stale Adam state.
5. Compare drift-triggered, no-restoration, restore-every-step, MLP-only, random-skew initialization, and matched targeted-LoRA control.

## Run

```bash
python3 experiments/exp120_drift_triggered_orthogonal_lanes/scripts/run_drift_triggered_rotations.py
```

## Output

`results/drift_triggered_rotations_summary.json` plus `results/step_metrics.jsonl` — reward/dead-group/KL traces, restoration counts/timing, spectral drift, singular-value drift, final policy/safety metrics, wall time, and device.
