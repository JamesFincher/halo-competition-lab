# Experiment 59 — Architecture-Matched Warm GQA Slider Heads

**Research inspiration:** `research/HYPOTHESES.md` H1/H3, arXiv:2611.28483 (exact-zero gated weight inserts), arXiv:2611.31615 (attention-head functional interventions), and local Exp 42.

## Hypothesis

Exp 42 proved a zero-init scalar-gated attention insert can reduce false refusal, but its random single-head shape is not architecture-matched and exact compliance/policy accuracy degraded. A SmolLM2-native GQA lane (3 query heads sharing 1 KV head) warm-copied from the behaviorally important L29 KV group should give RL a useful additive attention circuit while preserving exact zero impact at alpha=0.

## Method

1. Load merged Halo SFT checkpoint in fp32 and evaluate the standard 40-task heldout set.
2. Insert additive lanes after layer 27; all core weights frozen.
3. Compare random-init vs warm-copied 3Q/1KV lane, each behind scalar alpha initialized exactly to zero.
4. Train only insert parameters with grouped exact-compliance RL plus KL to the unedited reference.
5. Report before/after exact compliance, unsafe refusal, benign false refusal, alpha, parameter count, and traces.

## Run

```bash
python3 experiments/exp59_warm_gqa_slider_heads/scripts/run_warm_gqa_slider.py
```

## Output

`results/warm_gqa_slider_summary.json`.
