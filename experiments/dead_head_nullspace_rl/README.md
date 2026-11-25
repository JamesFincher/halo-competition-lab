# Experiment 5: QO-Only Zero-Init Dead-Head Recycling & Null-Space QO-LoRA

**Priority:** Optional Side-Arm (**4/10** — cheap Qwen3-4B falsification gate; never inject new architectural pseudo-attention modules)  
**Target Models:** `Qwen/Qwen3-4B` (cleanest 4:1 GQA pilot) → `unsloth/gpt-oss-20b-BF16` (requires offline `sinks[h]` reset)

## Core Hypothesis
Instead of injecting new pseudo-attention modules into dead MLP/MoE space (mathematically impossible without forward-graph edits because FFNs have zero cross-token Jacobian, and silently dropped by Halo's `_HubForwarder` / vLLM `model.load_weights`), we test the **two architecture-preserving realizations** that keep exact tensor names, shapes, KV-cache layout, and `config.json`:
1. **QO-Only Zero-Init Dead-Head Recycling (True Independent Pseudo-Head):** Identify the bottom 5–10% lowest-impact query heads (by `o_proj` column Frobenius norm + zero-ablation NLL delta), re-initialize ONLY their `q_proj` rows (small orthogonal init) and set their `o_proj` columns to **exact zero** (bit-identical step-0 outputs/logprobs), then train with a head-masked gradient hook + `weight_decay: 0.0`.
2. **Activation/Weight Null-Space QO-LoRA Grafting (`LoRA-Null` / `NB-LoRA`):** Apply LoRA strictly to `q_proj` and `o_proj` initialized in the bottom singular subspace so updates graft orthogonal QK/OV behavior without overwriting pretrained circuits (merges cleanly with Experiment 3 `GRAIL-Orth`).

## Critical GQA & Halo v1.3.0rc1 Safety Rules
- **NEVER recycle `k_proj` or `v_proj`:** `Qwen3-4B` (4:1 GQA), `gpt-oss-20b` (8:1 GQA), and `Gemma 4 26B-A4B` (8:1 GQA + cross-layer KV sharing + `attention_k_eq_v`) share each KV head across 4–8 live query heads. Touching K/V corrupts active sibling heads.
- **`gpt-oss-20b` attention sink trap:** Halo refuses `train_sinks: true` in on-policy RL (`reset_sinks: false`). If a dead head in `gpt-oss-20b` is suppressed by a large learned `sinks[h]` logit, reset `sinks[h]` offline in the checkpoint BEFORE launching RL.
- **Mandatory `weight_decay: 0.0`:** Prevents Halo's `AdamWBF16` from decaying frozen active-head slices.

## Supporting Literature
- **`arXiv:2503.02659` (*LoRA-Null*) & `arXiv:2609.25618` (*NB-LoRA*):** Null-space projection of LoRA adapters preserves pretrained circuits.
- **`arXiv:2603.09616` & `arXiv:2504.03889`:** Selective surgical head recycling requires zero-initialized `W_O^h` and careful warm-up to avoid residual-stream instability.

## Files
- `scripts/recycle_dead_qo_heads.py` — Profiles Q-heads, exports dead-head mask, and saves a step-0 bit-identical `checkpoint_recycled/` (`q_proj` orthogonal re-init + exact zero `o_proj` columns, K/V untouched).
- `configs/qwen3-4b-dead-head-qo-grpo.yaml` — QO-only Halo online GRPO config (`weight_decay: 0.0`).

## Compute & Go/No-Go Criteria
- **Pilot cost (`Qwen3-4B`, 1-day gate):** ~$25–$40.
- **GO:** Recycled QO heads or Null-Space QO-LoRA beat standard QO-LoRA by ≥1.5 pp on held-out pass@1 with ≤50% OOD forgetting; otherwise keep compute on Experiment 1 (Edit-Then-Target RL).
