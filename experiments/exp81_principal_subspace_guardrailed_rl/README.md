# Experiment 81 — Principal-Subspace Guardrails for Edit-Safe RL

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **9** + **arXiv:2608.11674** (*GCPO bilateral orthogonal-complement constraints*), **arXiv:2609.34344** (*Alpha-Stabler-style stability geometry*), and **arXiv:2606.31813** (*geometry-preserving orthonormal initialization*).

## Hypothesis

Layer-disjoint RL controls *which matrices* move but not *which directions inside those matrices* move. Projecting updates away from dominant pretrained singular subspaces should reduce destructive interference with the surgically anchored policy, even if it does not maximize short-term reward.

## Method

1. Start from the fp32 validated rank-1 surgical edit at layers 29/28.
2. Precompute top-16 left/right singular subspaces for every attention and MLP matrix in complementary layers 24–27.
3. Train those full matrices with exact-compliance GRPO and `weight_decay=0` under:
   - unconstrained KL+layer-disjoint control
   - bilateral orthogonal-complement projection
   - left-subspace-only projection control
4. Measure forbidden-subspace gradient overlap every step, including in the unconstrained arm, and correlate it with exact compliance, unsafe refusal, benign false refusal, and KL.

## Run

```bash
python3 experiments/exp81_principal_subspace_guardrailed_rl/scripts/run_principal_subspace_guardrailed_rl.py
```

## Output

`results/principal_subspace_guardrailed_summary.json` plus `results/step_metrics.jsonl` — forbidden-gradient overlap, reward/dead-group/KL trajectories, final held-out metrics, validated-edit metadata, wall time, and device.
