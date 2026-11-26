# Experiment 143 — contrastive continuation-choice capability objective

**Controlled variable:** capability objective only, with benchmark-direct format, answer-bearing rows, row order, architecture, seed, optimizer, learning rate, batch size, and step budget fixed.

## Motivation

Correct-answer causal training is format-aligned but still not fully benchmark-aligned. The continuation benchmark is discriminative: it compares several candidate continuations in the same context. Merely increasing the probability of the correct continuation may not improve relative ranking if incorrect continuations move together.

This experiment tests the missing alignment directly.

## Arms

1. `correct_answer_causal_reference` — answer-only causal LM loss on `context + " " + correct_answer`.
2. `contrastive_choice_objective` — token-normalized continuation logprob for every candidate, followed by a softmax cross-entropy loss that rewards ranking the correct continuation above incorrect alternatives.

The causal-reference records are derived directly from the multiple-choice records, so row identity, answer payload, source order, and data order are paired by construction rather than reconstructed across two independent loaders. Objective magnitudes inherently differ, so the comparison is at matched optimizer budget rather than matched raw loss scale.

## Endpoints

Before/after behavioral collateral, legacy benchmark, continuation benchmark (`n=150` default), and a heldout contrastive-validation loss/accuracy measured on split-disjoint validation rows. Training remains restricted to MMLU validation, ARC-Easy train, and HellaSwag train; benchmark evaluation uses MMLU test, ARC-Easy test, and HellaSwag validation.

## Output

`results/contrastive_choice_capability_summary.json`
