# Experiment 142 — capability-training budget dose-response

**Controlled variable:** optimization budget only (`1x` vs `4x` vs `16x` steps), with architecture, data, direct continuation-style format, answer-only token weighting, seed, optimizer, learning rate, and batch size fixed.

## Motivation

The first capability-objective experiment used only 32 optimizer steps — 128 examples, less than one pass over even the small split-decontaminated corpus. If benchmark transfer is absent under that budget, there are two very different explanations:

1. inserted heads cannot add transferable capability in the saturated 135M residual stream, or
2. capability training was too brief to produce transfer.

This experiment separates those explanations using benchmark-direct answer-only training, the current best-aligned objective candidate.

## Arms

- `budget_1x_32_steps` — original budget reference.
- `budget_4x_128_steps` — four times the original budget.
- `budget_16x_512_steps` — sixteen times the original budget.

All arms start from identical randomly initialized zero-alpha heads and consume the same deterministic data-order prefix, so shorter arms are exact prefixes of longer arms. Endpoints include behavioral collateral, legacy benchmark, continuation benchmark (`n=150` by default), and full/answer-only validation loss.

## Output

`results/capability_training_budget_response_summary.json`
