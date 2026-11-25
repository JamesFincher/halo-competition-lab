# Experiment 13: 1-Bit Sign-Flip & Zero-Mask False-Node Editing + Safety Bit-Flip Audit

**Priority:** Ultra-Compact Binary Surgery & Security Audit (**8.0/10**)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (verified live) → `Qwen/Qwen3-4B` → `unsloth/gpt-oss-20b-BF16`

## Core Idea (Plain English)
Tests whether **1-bit operations** on individual sniffed neurons can match or beat continuous weight projection, and audits how fragile safety alignment is to targeted single-bit sign flips:

1. **1-Bit Zero-Mask (`w → 0`):** Switch off only the top-K sniffed false nodes (`Layer 29 #453, #1392, #1513`, etc.) with a binary keep/drop bit.
2. **1-Bit Sign-Flip (`w → -w`):** Flip literally only the IEEE-754 / BF16 **sign bit** of the top-K sniffed false-node columns.
3. **Continuous Projection Reference (Exp 12):** Rank-1 projection on top-64 false nodes.
4. **Best False-Node Edit → Early+Mid Disjoint Re-Anchored RL (`layers [0, 1, 11, 19, 20, 27]`):** Tests whether combining 100%-safety false-node editing with early-layer disjoint RL beats our `77.5%` Fork V3 record while preserving higher safety.
5. **Security Track — Safety Bit-Flip Attack (BFA) Probe:** Flips the sign bit of the top `1, 2, 4, 8, 16` **true-safety** neurons to measure the exact minimum sign-bit flips needed to break refusal alignment (motivating Experiment 9 INSEP defense).

## Files
- `scripts/run_bitflip_experiment.py` — Runs all 4 arms + Safety BFA probe on 40 unseen tasks.
- `configs/qwen3-4b-bitflip-false-node-grpo.yaml` — Halo v1.3.0rc1 GPU config for scaling to `Qwen3-4B`.
- `results/BITFLIP_REPORT.md` — Tracked scoreboard, bit-flip counts, and safety audit.
