# Experiment 57 — Halo-Native Post-Edit Anchored Targeted LoRA

**Research inspiration:** `research/SUBAGENT4_HALO_NATIVE_RL.md` top transfer recommendation + **arXiv:2403.10704** (*PE-RLHF: parameter-efficient RLHF*) and **arXiv:2609.18587** (*label-free small-parameter-space RL*).

## Hypothesis

The verified micro loop should transfer when expressed the way Halo expresses it natively: the post-edit checkpoint is both policy initialization and KL anchor, PEFT freezes all base weights including edited layers, a bare regex targets only complementary layers, and `weight_decay=0` prevents silent drift.

## Method

1. Start from the fp32 validated raw-rank-1 residual-direction edit at layers 29/28.
2. Anchor the KL reference to that exact post-edit model.
3. Compare:
   - complementary layers 24–27, attention+MLP LoRA r16
   - complementary layers 24–27, MLP-only LoRA r16
   - negative control that also targets edited layers 28/29
   - full-model LoRA r8 control
4. Train at reduced Stage-2 LR `1e-4`, at most 8 steps, with `weight_decay=0` inherited from the base GRPO harness.

## Run

```bash
python3 experiments/exp57_halo_native_targeted_lora/scripts/run_halo_native_targeted_lora.py
```

## Output

`results/halo_native_targeted_lora_summary.json` plus `results/step_metrics.jsonl` — trainable parameter counts, reward/KL trajectories, validated-edit telemetry, held-out exact compliance and safety metrics, wall time, and device. Graded per `research/RUBRIC.md`.
