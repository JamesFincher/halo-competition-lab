# Experiment 101 — Merge-Aware Served-Weight Action-Token Fine-Tuning

**Research inspiration:** `research/HYPOTHESES.md` **H10** + **arXiv:2611.32064** (*Fix How It Acts, Not How It Thinks: Surgical Fine-Tuning of Frontier LLMs*).

## Hypothesis

The artifact that matters is the merged, BF16-rounded served checkpoint—not the fp32 hook-time approximation. Training adapters through a straight-through rounded merged-weight forward should preserve more gain after export, while action/tool/policy-only loss should reduce schema and trajectory collateral.

## Method

1. Start from the fp32 validated post-edit checkpoint and freeze base weights.
2. Install matched custom rank-16 LoRA wrappers on complementary layers 24–27.
3. Factorially compare:
   - ordinary unmerged/unrounded fp32 forward vs straight-through BF16-rounded merged forward;
   - all-completion-token loss vs action/tool/policy value-token-only loss.
4. Anchor sequence drift to the post-edit checkpoint and use `weight_decay=0`.
5. Evaluate both the in-process training representation and a merged+BF16-rounded export for every arm.

## Run

```bash
python3 experiments/exp101_merge_aware_served_weight_finetune/scripts/run_merge_aware_finetune.py
```

## Output

`results/merge_aware_finetune_summary.json` plus `results/step_metrics.jsonl` — CE/KL/mask telemetry, hook-vs-export metrics, exact export gaps, tensor round-trip error, wall time, and device.
