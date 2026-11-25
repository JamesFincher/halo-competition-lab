#!/usr/bin/env bash
set -euo pipefail

# One-command runner for Edit-Then-Target RL ablation suite (run inside Halo Docker image on GPU node)
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PAIRS_FILE="${PAIRS_FILE:-data/policy_train.jsonl}"
GATE_FILE="${GATE_FILE:-data/policy_eval.jsonl}"
HALO_DIR="${HALO_DIR:-/workspace/halo}"
OUT_ROOT="${OUT_ROOT:-outputs/edit_then_target}"

echo "[Stage 1] Contrastive + LS-LoRA + 1-Shot Fisher Localization & Surgical Pre-Edit..."
python experiments/edit_then_target_rl/scripts/localize_and_edit_weights.py \
  --model "$MODEL" \
  --pairs-file "$PAIRS_FILE" \
  --out-dir "$OUT_ROOT/stage1" \
  --apply-surgical-edit

echo "[Stage 1b — V4 Dual-Edit (arXiv:2611.33133)] Half-strength early-layer bias-in-disguise edit..."
python experiments/edit_then_target_rl/scripts/dual_and_iterative_edit.py \
  --mode dual_bias_edit \
  --model "$OUT_ROOT/stage1/checkpoint_edited" \
  --train-file "$PAIRS_FILE" \
  --gate-file "$GATE_FILE" \
  --out-dir "$OUT_ROOT/stage1_dual"

echo "[Stage 1c — V6 Iterative Gated SELFWRITE (arXiv:2611.29513)]..."
python experiments/edit_then_target_rl/scripts/dual_and_iterative_edit.py \
  --mode iterative_gated \
  --model "$MODEL" \
  --train-file "$PAIRS_FILE" \
  --gate-file "$GATE_FILE" \
  --out-dir "$OUT_ROOT/stage1_gated"

echo "[Stage 2] Ready to launch Halo GRPO arms (pass ARM=arm0..arm7):"
for cfg in experiments/edit_then_target_rl/configs/arm*.yaml; do
  echo "  halo launch grpo $cfg"
done
