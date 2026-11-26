# Experiment 139 — Cache-Corrected Capability-Objective Slider Replication

**Iteration-4 question:** can an inserted zero-init slider head add measurable capability in a saturated 135M residual stream when it is trained with a capability signal rather than only policy-behavior reward?

## Controlled variable

Only the training objective varies:

1. `capability_only` — next-token loss on split-disjoint benchmark-style train text;
2. `behavior_only` — exact slider-fork grouped behavioral REINFORCE;
3. `hybrid_capability_primary` — capability loss plus a behavior regularizer (`capability_loss + 10 * behavior_loss`).

## Fixed conditions

- one zero-init width-64 plain slider head at layer 27;
- same seed before every arm;
- same AdamW hyperparameters, 32 steps, behavior prompt stream, and group size;
- same capability corpus ordering and batch size;
- same before/after 40-task behavioral eval, real benchmark, and held-out capability-validation loss.

## Contamination control

Capability text is drawn only from:

- MMLU `validation` while benchmark measures MMLU `test`;
- ARC-Easy `train` while benchmark measures ARC-Easy `test`;
- HellaSwag `train` while benchmark measures HellaSwag `validation`.

No measured benchmark split is used for training. Corpus provenance, load errors, counts, and SHA-256 fingerprint are written into the result JSON.

## Run

```bash
python3 experiments/exp139_cache_corrected_capability_objective/scripts/run_cache_corrected_capability_objective.py
```

## Output

`results/cache_corrected_capability_objective_summary.json` with per-arm legacy benchmark deltas, continuation-scored benchmark deltas (sum/token-normalized/byte-normalized answer likelihood), behavioral deltas, held-out capability-loss deltas, final alpha, and full training traces.


This is a cache-corrected replication of exp132: reward generation and behavioral evaluation use `use_cache=False` so inserted heads can attend across the full generated sequence. The original exp132 artifact ran before that correction and is retained for comparison rather than overwritten.
