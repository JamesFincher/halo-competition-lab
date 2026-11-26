# Micro-Model Targeted RLVR/GRPO Benchmark Report (`SmolLM2-135M-Instruct`)

- **Device:** `cuda` · **Starting Checkpoint:** Halo v1.3.0rc1 SFT (`checkpoint-36`, merged)
- **Protocol:** 8 GRPO steps/arm × 2 balanced prompts/step × G=4 rollouts, `lr=0.0003`, `beta=0.02`, `weight_decay=0.0`, `seed=42`
- **Localized Top-8/30 Layers:** `[0, 1, 11, 20, 24, 27, 28, 29]` (Random Control: `[2, 5, 6, 9, 10, 14, 21, 25]`)
- **Stage-1 Surgical Pre-Edit:** Top-2 contrastive MLP `down_proj` layers `[29, 28]`, `36,864` weights (`0.02741%`)

## Held-Out 40-Task Unseen Evaluation Results

| Arm | Trainable Params | Wall Time | Mean Reward | JSON Valid | Action Acc | Unsafe Refusal | Benign False-Refusal | Exact Compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Starting Halo SFT (`checkpoint-36`) | 0 (0.0%) | 0.0s | 0.775 | 100.0% | 80.0% | 100.0% | 80.0% | **47.5%** |
| Arm 0: Full-Model LoRA GRPO (30 layers, r=8) | 2,442,240 (1.783%) | 110.5s | 0.665 | 100.0% | 60.0% | 46.7% | 0.0% | **47.5%** |
| Arm 1: Stage-1 Surgical Pre-Edit Only (0 steps) | 36,864 (0.02741%) | 0.5s | 0.920 | 100.0% | 95.0% | 93.3% | 0.0% | **75.0%** |
| Arm 2: Localized Targeted GRPO Only (8 layers, r=16) | 1,302,528 (0.959%) | 68.0s | 0.930 | 100.0% | 95.0% | 93.3% | 0.0% | **80.0%** |
| **Arm 3: Flagship Edit-Then-Target GRPO (8 layers, r=16)** | 1,302,528 (0.959%) | 69.4s | 0.725 | 100.0% | 75.0% | 100.0% | 100.0% | **37.5%** |
| Arm 4: Random-Layer Control (8 layers, r=16) | 1,302,528 (0.959%) | 67.8s | 0.845 | 100.0% | 87.5% | 100.0% | 50.0% | **60.0%** |

## Per-Step Training Telemetry

Full per-step logs (reward, active non-degenerate groups, KL from frozen reference, policy loss, grad norm, wall time) are recorded in [`step_metrics.jsonl`](step_metrics.jsonl) and [`micro_rl_summary.json`](micro_rl_summary.json).
