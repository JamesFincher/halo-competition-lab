# Experiment 93 — v1 Single-Slider Seed Replication

**Question:** Is the surviving v1 single narrow late-layer result robust, or a lucky random initialization/rollout seed?

## Controlled variable

Only random seed varies: 7, 42, 123. The seed controls both head initialization and rollout sampling together, matching replication of the original stochastic pipeline.

## Fixed conditions

- one width-64 head at layer 27;
- exact-zero alpha;
- same fixed fork prompt stream;
- same 32-step grouped REINFORCE budget, group size 4, AdamW lr 3e-3, no KL/BC;
- same behavioral 40-task eval and real MMLU/ARC-Easy/HellaSwag benchmark before/after.

## Run

```bash
python3 experiments/exp93_v1_seed_replication/scripts/run_seed_replication.py
```

## Output

`results/v1_seed_replication_summary.json`.
