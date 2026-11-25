# Experiment 15: 1-Pass RelP Causal Attribution Patching Sniff

**Papers:** `arXiv:2508.21258` (*RelP: Faithful and Efficient Circuit Discovery via Relevance Patching*, Neel Nanda et al., `0.006 → 0.956` Pearson faithfulness), `arXiv:2610.05190` (*FORGE: Verification-Gated Behavioral Repair*, Oct 2026)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (verified live) → `Qwen/Qwen3-4B` → `unsloth/gpt-oss-20b-BF16`

## Core Math (True Causal Score in 1 Backward Pass)
Standard activation sniffing ranks neurons/heads by raw activation difference `||a_fail - a_pass||`, which can flag high-norm decoys with zero causal effect on output logits (as observed on attention heads in Experiment 12).  
**RelP-style Causal Attribution** computes the exact first-order causal effect on the target logit margin \(\mathcal{M} = \log p(\text{call\_tool}) - \log p(\text{refuse})\) in **1 backward pass**:
\[\text{CausalScore}(l, j) = \big(a_{l,j}^{\text{pass}} - a_{l,j}^{\text{fail}}\big) \cdot \frac{\partial \mathcal{M}}{\partial a_{l,j}}\]
with stop-grad on RMSNorm variance scaling (following `RelP` `arXiv:2508.21258` to eliminate deep-MLP gradient noise).

## Files
- `scripts/run_relp_causal_sniff.py` — Compares 1-Pass RelP Causal Neuron Selection vs Activation-Only False-Node Selection on 40 unseen tasks.
- `configs/qwen3-4b-relp-causal-grpo.yaml` — Halo v1.3.0rc1 GPU config for scaling to `Qwen3-4B`.
- `results/RELP_CAUSAL_REPORT.md` — Tracked causal neuron rankings and 40-task comparison.
