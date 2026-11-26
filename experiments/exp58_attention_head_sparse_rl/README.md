# Experiment 58 — Attention-Head Sparse RL Complementary to MLP Surgery

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **8** + **arXiv:2610.21336** and **arXiv:2611.10866** (head-ranked sparse RL / random sparse controls).

## Hypothesis

The validated MLP edit supplies the refusal/tool direction, but context-dependent routing may live in attention. Restricting RL to gradient-ranked attention-head q/output slices could add contextual routing without overwriting the surgical mechanism.

## Method

1. Start from the fp32 validated raw-rank-1 residual-direction edit at layers 29/28.
2. Teacher-force correct action tokens and rank every non-edited attention head by initial q_proj-row + o_proj-column gradient norm.
3. Select the top 10% of heads; compare bottom-ranked, random, and dense complementary-layer controls at matched effective slice budgets.
4. Freeze all embeddings/MLPs/edited layers. Train only selected q/output slices with exact-compliance GRPO, KL to the post-edit checkpoint, and `weight_decay=0`.
5. Report reward, dead-group fraction, KL, exact compliance, unsafe refusal, benign false refusal, and selected-head layer distribution.

## Run

```bash
python3 experiments/exp58_attention_head_sparse_rl/scripts/run_attention_head_sparse_rl.py
```

## Output

`results/attention_head_sparse_rl_summary.json` plus `results/step_metrics.jsonl` — ranked-head preview, exact selected heads, effective slice parameter counts, per-step telemetry, held-out metrics, wall time, and device. Graded per `research/RUBRIC.md`.
