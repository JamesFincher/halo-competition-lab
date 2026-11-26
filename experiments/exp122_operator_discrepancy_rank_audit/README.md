# Experiment 122 — Operator-Discrepancy Rank Audit for H2 Repairability

**Research inspiration:** `research/HYPOTHESES.md` **H2** + **arXiv:2611.32755** (*ReQOR: Operator-Discrepancy Subspace Repair*).

## Hypothesis / pre-flight concern

ReQOR-style target-orthogonal collateral repair only has something to repair when the same-state operator discrepancy spans multiple directions. Both leading Halo surgical edits are explicitly rank-1 in output direction, so the collateral subspace may be structurally empty. This experiment tests that gate before spending GPU time on a full H2 adapter.

## Method

1. Reconstruct the Exp1-style contrastive edit (top-2 divergence layers, top-32 `down_proj` columns, strength 0.5).
2. Reconstruct the validated residual rank-1 edit (layers 29/28, top-64 columns, strength 0.9).
3. Cache `down_proj` inputs along correct edited trajectories.
4. Compute same-state discrepancy matrices analytically as \((W_\text{orig}-W_\text{edit})x\), then measure singular-energy spectra, effective rank, alignment to the weight-delta direction, and non-target collateral energy.
5. Report behavioral endpoints for both operators alongside the structural conclusion.

## Run

```bash
python3 experiments/exp122_operator_discrepancy_rank_audit/scripts/run_operator_discrepancy_audit.py
```

## Output

`results/operator_discrepancy_rank_audit.json` — per-layer discrepancy spectra, effective rank, target-direction alignment, collateral rank/energy, behavioral endpoints, and a binary H2 repairability conclusion.
