# Experiment 90 — Single-Variable Slider Width Ablation

**Fork finding:** v1 single narrow L27 head survived (57.5% exact, alpha +0.0117); v4 single dim128 head collapsed (0% exact, alpha negative). This experiment asks whether width alone causes that transition.

## Controlled variable

Only slider head width varies: 16, 32, 64, 128.

## Fixed conditions

- one head at layer 27 only;
- same seed before every arm;
- same random-initialization family and exact zero alpha;
- same fixed prompt stream used by `slider_head_fork` (12 benign then 12 unsafe);
- same 32-step grouped REINFORCE budget, group size 4, AdamW lr 3e-3, no KL/BC;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

TransformerLens probes L27 residual geometry/attention entropy once on the unedited model for interpretability context.

## Run

```bash
python3 experiments/exp90_single_slider_width_ablation/scripts/run_width_ablation.py
```

## Output

`results/single_slider_width_ablation_summary.json`.
