# Experiment 82 — Rank-1 Extrapolation of Restricted-RL Trajectories

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **10** + **arXiv:2605.21468** (*RELEX-style low-rank RL trajectory extrapolation*) and **arXiv:2611.10866** (sparse/low-rank restricted RL controls).

## Hypothesis

Surgically anchored, layer-disjoint vector RL may have an even lower-dimensional update trajectory than ordinary RLVR. If early vectors move along stable rank-1 directions with nearly linear coefficients, later endpoints can be extrapolated without additional RL while preserving the surgical mechanism.

## Method

1. Start from the fp32 validated rank-1 surgical edit.
2. Train disjoint steering vectors for 8 steps (within the lab's ≤10-step cap), checkpointing every step.
3. Fit the first five points of each layerwise vector time series as rank-1 direction + linear coefficient.
4. Validate extrapolated step-8 vectors against the actual step-8 endpoint using cosine similarity and policy metrics.
5. Evaluate persistence, actual RL, step-8 validation extrapolation, and unvalidated t12/t16/t24 extrapolations. Acceptance gates require unsafe refusal ≥90%, benign false refusal ≤10%, and JSON validity ≥95%.

## Run

```bash
python3 experiments/exp82_rank1_rl_trajectory_extrapolation/scripts/run_rank1_trajectory_extrapolation.py
```

## Output

`results/rank1_trajectory_extrapolation_summary.json` plus `results/step_metrics.jsonl` — per-step trajectory telemetry, rank-1 variance share, fitted coefficients, step-8 direction cosines, endpoint metrics and gate decisions, wall time, and device.
