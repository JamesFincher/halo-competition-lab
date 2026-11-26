# Experiment 91 — Single-Variable Slider Depth Ablation

**Question:** Is the surviving v1 effect specific to layer 27, or does insertion depth alone determine stability?

## Controlled variable

Only insertion layer varies: 16, 20, 24, 27, 29.

## Fixed conditions

- one width-64 head;
- same seed before every arm;
- same exact-zero alpha and random-init family;
- same fixed fork prompt stream;
- same 32-step grouped REINFORCE budget, group size 4, AdamW lr 3e-3, no KL/BC;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

TransformerLens probes residual geometry and attention entropy at every candidate layer in one cache pass per prompt.

## Run

```bash
python3 experiments/exp91_single_slider_depth_ablation/scripts/run_depth_ablation.py
```

## Output

`results/single_slider_depth_ablation_summary.json`.
