# Experiment 140 — Answer-Focused Capability Objective

**Question:** does aligning capability training with answer-continuation scoring improve benchmark generalization relative to uniform full-text causal loss?

Exp132's cache-independent capability-only arm reduced heldout validation loss but did not produce robust continuation-benchmark gains. A plausible cause is objective mismatch: uniform causal loss spends most capacity predicting question/context tokens, while the benchmark scores only answer continuations.

## Controlled variable

Only relative context-vs-answer token weighting varies:

1. `full_text_uniform_reference` — context weight 1, answer weight 1;
2. `answer_upweighted_4x` — context weight 1, answer weight 4;
3. `answer_only` — context weight 0, answer weight 1.

The answer span begins after the final `Answer: ` or `Continuation: ` marker. Architecture, insertion layer, width, seed, optimizer, corpus order, and step budget are fixed.

## Contamination control

Uses only MMLU validation, ARC-Easy train, and HellaSwag train. Benchmarks measure MMLU test, ARC-Easy test, and HellaSwag validation.

## Run

```bash
python3 experiments/exp140_answer_focused_capability_objective/scripts/run_answer_focused_capability_objective.py
```

## Output

`results/answer_focused_capability_objective_summary.json` with behavioral eval, legacy benchmark, continuation-only benchmark, full validation loss, answer-only validation loss, deltas, alphas, and traces.
