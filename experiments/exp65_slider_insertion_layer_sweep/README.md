# Experiment 65 — Warm GQA Slider Insertion-Layer Sweep

**Research inspiration:** H3 functional/localization question plus Exp 42/59/61/62, which all use late-layer insertion but have not isolated insertion depth.

## Hypothesis

If false refusal is a late readout failure, L29 insertion should dominate. If it is a coordination failure that needs upstream correction, L24/L27 should preserve policy accuracy better. A fixed-dose sweep distinguishes these mechanisms without confounding architecture or initialization.

## Method

- Same warm 3Q/1KV lane copied from L29 KV group 0.
- Insert after L24, L27, or L29.
- Exact zero at alpha=0; core frozen; joint grouped RL + KL + light behavior cloning (`bc_lambda=0.5`).
- Standard 40-task heldout metrics and traces for every arm.

## Run

```bash
python3 experiments/exp65_slider_insertion_layer_sweep/scripts/run_insertion_layer_sweep.py
```

## Output

`results/slider_insertion_layer_sweep_summary.json`.
