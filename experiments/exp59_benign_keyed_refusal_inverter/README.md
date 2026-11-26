# Experiment 59 — Benign-Keyed Negative-Image Refusal Inverter

**Research inspiration:** `research/SUBAGENT3_CREATIVE_HEURISTICS.md` rank **1** + **arXiv:2406.11717** (single-direction refusal mediation) and **arXiv:2012.14913** (feed-forward layers as key-value memories).

## Hypothesis

Refusal need not be globally erased. One carefully chosen MLP neuron can act as a signed benign/unsafe discriminator: positive benign activation writes the negative image of refusal, while negative unsafe activation writes positive refusal.

## Method

1. Collect decision-token MLP inputs at layers 27–29 for benign `call_tool` and unsafe `refuse` examples.
2. Compute a centered Fisher/LDA key separating benign from unsafe contexts.
3. Choose a low-collision non-hub neuron, overwrite its gate/up rows with mean-context and Fisher-key directions, and write `-alpha * refusal_direction` through its down_proj column.
4. Grid layers 27/28/29 and alpha `{0.1,0.25,0.5,0.75}`.
5. Re-anchor the accepted checkpoint and run at most 4 targeted complementary-LoRA GRPO steps.

## Run

```bash
python3 experiments/exp59_benign_keyed_refusal_inverter/scripts/run_benign_keyed_inverter.py
```

## Output

`results/benign_keyed_inverter_summary.json` plus `results/step_metrics.jsonl` — input counts, selected neuron, grid metrics, accepted edit, post-edit RL telemetry, held-out exact/safety metrics, wall time, and device.
