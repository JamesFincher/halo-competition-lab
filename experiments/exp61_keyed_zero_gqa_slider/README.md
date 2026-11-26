# Experiment 61 — Exact-Zero Keyed Warm GQA Slider

**Research inspiration:** `research/HYPOTHESES.md` H1 (exact-zero keyed modular implant bank) and arXiv:2611.28483.

## Hypothesis

Exp 42/59-style scalar-slider heads become globally active once alpha moves away from zero. An input-conditioned ReLU key initialized from measured benign-vs-unsafe residual boundaries should activate the added attention lane primarily on false-refusal/benign-control states while remaining exactly zero in negative-key regions. This should preserve the false-refusal benefit with less unsafe-refusal and policy-accuracy collateral.

## Method

1. Capture layer-27 final-token residual activations on balanced benign/unsafe training rows.
2. Calibrate a linear key and threshold; gate is `relu((x·key - threshold)/scale)`.
3. Warm-copy one native 3Q/1KV attention lane from L29 KV group 0.
4. Compare always-on warm lane vs keyed warm lane, both exactly identity at alpha=0.
5. Train insert-only parameters with grouped exact-compliance RL plus KL anchoring and evaluate the standard 40-task heldout set.

## Run

```bash
python3 experiments/exp61_keyed_zero_gqa_slider/scripts/run_keyed_zero_slider.py
```

## Output

`results/keyed_zero_gqa_slider_summary.json`.
