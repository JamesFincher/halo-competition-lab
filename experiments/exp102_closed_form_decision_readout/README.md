# Experiment 102 — Closed-Form Decision-Readout Correction

**Research inspiration:** `research/HYPOTHESES.md` **H12** + **arXiv:2611.33972** (*Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder*).

## Hypothesis

Correct action/tool/policy decisions may already be linearly or kernel-linearly readable in late hidden states immediately before the decision value. If so, a deterministic closed-form readout can correct only those few tokens without gradient training or schema-wide drift.

## Method

1. Teacher-force train rows through frozen SFT and validated-edit models.
2. Cache hidden states at layers 20–29 immediately before `action`, `tool`, and `policy_code` values.
3. Fit exact shrinkage-LDA, linear-ridge, and kernel-ridge classifiers; select the layer/method by 4-fold cross-fitted train accuracy.
4. During manual greedy decoding, detect entry into a quoted decision field and force the probe-predicted canonical value tokens only inside that field.
5. Compare no correction, probe correction, majority-label forcing, and random-label forcing on SFT and validated-edit models.

## Run

```bash
python3 experiments/exp102_closed_form_decision_readout/scripts/run_closed_form_readout.py
```

## Output

`results/closed_form_readout_summary.json` — cross-fitted probe diagnostics, label counts, built-in/manual greedy metrics, forced-token counts, correction-control comparisons, wall time, and device.
