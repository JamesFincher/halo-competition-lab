# Experiment 94 — Alpha-Clamp Ablation for the v2 Multi-Layer Bank

**Fork finding:** multi-head banks collapsed and showed negative alpha drift. This experiment tests whether unbounded alpha magnitude/sign movement is the causal failure variable.

## Controlled variable

Only post-step alpha clamp magnitude varies: none, ±0.03, ±0.01.

## Fixed conditions

- v2 architecture exactly: width-64 heads at layers 24/25/26/27;
- independent alpha per layer;
- same seed before every arm;
- same fixed fork prompt stream;
- same 32-step grouped REINFORCE budget, group size 4, AdamW lr 3e-3, no KL/BC;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

## Run

```bash
python3 experiments/exp94_alpha_clamp_multilayer_bank/scripts/run_alpha_clamp_ablation.py
```

## Output

`results/alpha_clamp_multilayer_bank_summary.json`.
