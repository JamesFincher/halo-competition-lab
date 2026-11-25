# Experiment 14: Zero-Shock Stitched GQA Attention Head + Freeze-Base RL

**Papers:** `arXiv:2410.11038` (*Theory of Function-Preserving Transforms*), `arXiv:2609.37702` (*Width Expansion for Class Incremental Learning*, Sep 2026)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (`9Q/3KV → 12Q/4KV`, verified live) → `Qwen/Qwen3-4B` (`32Q/8KV → 36Q/9KV` or `+4KV` for TP=4)

## Core Math (Exact Zero Shock at Step 0)
Standard multi-head attention sums over heads: \(y = \sum_{h=1}^{H} W_O^{(h)} \text{Attn}^{(h)}(x)\).  
By appending **+1 GQA group** (+1 KV head, +\(g\) Q heads so `num_attention_heads / num_key_value_heads` stays an integer) directly in standard `config.json`:
- Initialize \(W_Q^{\text{new}}, W_K^{\text{new}}, W_V^{\text{new}}\) to small non-zero / SVD directions (so \(\text{Attn}^{\text{new}}(x) \neq 0\)),
- Initialize **\(W_O^{\text{new}} = 0\)** (exact zeros).

At step 0, **\(\max |f_{\text{new}}(x) - f_{\text{old}}(x)| = 0.0\) (zero shock, `KL = 0.000000`)**. On step 1 of RL, gradients flow immediately into \(W_O^{\text{new}}\) while **100% of original model weights stay frozen** — eliminating catastrophic forgetting while adding genuine new cross-token routing capacity, with zero custom modules (100% standard HuggingFace & vLLM checkpoint!).

## Files
- `scripts/run_zero_shock_stitched_head.py` — Widens `SmolLM2-135M` (`9Q/3KV → 12Q/4KV`), proves `max_logit_diff == 0.0`, freezes 100% of base weights, and runs GRPO ONLY on the stitched head group.
- `configs/qwen3-4b-stitched-head-grpo.yaml` — Halo v1.3.0rc1 GPU config for training only stitched GQA head weights.
- `results/STITCHED_HEAD_REPORT.md` — Live zero-shock proof & 40-task eval results.
