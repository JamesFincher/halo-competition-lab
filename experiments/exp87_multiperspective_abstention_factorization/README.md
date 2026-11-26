# Experiment 87 — Multi-Perspective Abstention-Direction Factorization

**Research inspiration:** `research/HYPOTHESES.md` **H15** + **arXiv:2306.11681** (*inference-time intervention / truthfulQA-style activation geometry*) and **arXiv:2410.12405** (*linearity and geometry of refusal/abstention representations*).

## Hypothesis

False refusal, true harmfulness refusal, and ambiguity/unknown-tool abstention may not be one monolithic direction. If their factorization differs, a selective false-refusal correction can avoid weakening the true-harmfulness refusal direction.

## Method

1. Construct approximate harmful/harmless lexical counterfactuals constrained by the existing three-policy schema.
2. Collect same-prompt behavior-conditioned states for correct benign tool calls versus false-refusal rollouts.
3. Construct ambiguity/unknown-tool prompt variants as measurement-only probes (the schema has no explicit unknown-tool refusal label).
4. Measure cross-family cosine alignment, linear CKA, split-half stability, diagonal-whitened stability, and residual alignment.
5. Validate selectivity with a small layer-24 steering comparison: no steering, negative raw false-margin direction, positive control, negative refusal-orthogonalized direction, and random control.

## Run

```bash
python3 experiments/exp87_multiperspective_abstention_factorization/scripts/run_abstention_factorization.py
```

## Output

`results/abstention_factorization_summary.json` — counterfactual counts, construct-validity caveat, per-layer geometry metrics, steering endpoint metrics, wall time, and device.
