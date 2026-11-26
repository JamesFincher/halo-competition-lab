# Experiment 100 — Halo-Native Post-Edit Transfer-Path Showdown

**Research inspiration:** `research/SUBAGENT4_HALO_NATIVE_RL.md` + **arXiv:2403.10704** (*PE-RLHF / parameter-efficient RLHF*) and **arXiv:2609.18587** (*small-parameter-space RL adaptation*).

## Hypothesis

Halo's targeted-LoRA transfer path and partial full fine-tuning path may preserve the surgical edit differently even when initialized from the same post-edit checkpoint. A same-seed showdown is needed before choosing the default restricted-RL transfer mechanism.

## Method

1. Start every arm from the same fp32 validated rank-1 edit at layers 29/28.
2. Freeze the edit itself and anchor KL explicitly to the post-edit checkpoint.
3. Train only complementary layers 24–27 or disjoint steering vectors.
4. Compare vector-only lower bound, targeted LoRA all-projection, targeted LoRA MLP-only, partial full fine-tuning all-projection, and partial full fine-tuning MLP-only.
5. Use identical prompts, seeds, group size, exact-verifier reward, `weight_decay=0`, and ≤10-step budget.

## Run

```bash
python3 experiments/exp100_halo_native_transfer_path_showdown/scripts/run_transfer_path_showdown.py
```

## Output

`results/transfer_path_showdown_summary.json` plus `results/step_metrics.jsonl` — reward/dead-group/KL traces, trainable parameter counts, full-FT drift norms, final policy/safety metrics, wall time, and device.
