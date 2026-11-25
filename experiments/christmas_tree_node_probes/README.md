# Experiment 17: 6-Probe "Christmas-Tree" Node-Tracking Benchmark

**Priority:** Mechanistic Localization Bake-Off (**9.0/10**)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (46,080 MLP neurons, 30 layers — verified live) → `Qwen/Qwen3-4B` → `unsloth/gpt-oss-20b-BF16`

## 6 Distinct Ways to Watch a Model Light Up & Track Problem Nodes
All 6 probes are evaluated on the **exact same model**, **exact same top-64 neuron budget per layer**, **exact same projection operator**, and **exact same 40 unseen verifiable tasks** (0 RL steps):

1. **Direct Logit Attribution (`DLA` / Logit Lens, 0 backward passes):** Projects each neuron's output `a_{l,j} * W_down[:, j]` directly through the unembedding difference `W_U[ref] - W_U[call]` — watches which neurons directly write the wrong token into output logits.
2. **Safety-Filtered Activation Spike (`False-Node Ratio`, Exp 12):** Ranks neurons by `|mu_false - mu_pass| - 0.45 * |mu_unsafe - mu_pass|`.
3. **1-Pass RelP Causal Attribution (`arXiv:2508.21258`, Exp 15):** Ranks neurons by `|(a_pass - a_fail) * grad_margin| / ||grad||`.
4. **5-Step Integrated Gradients (`IG-5`):** Integrates gradients along 5 linear steps between passing and failing activations to fix single-point gradient saturation.
5. **Flicker / Variance Spike (`Var-Ratio`):** Finds unstable neurons whose variance explodes on failing prompts relative to passing prompts (`var_fail / (var_pass + eps)`).
6. **Multi-Probe Reciprocal Rank Consensus (`DLA + False-Node + RelP`):** Fuses the top cameras via Reciprocal Rank Fusion (`1 / (60 + rank)`) to test if combining independent cameras beats every single probe alone.

## Files
- `scripts/run_probe_benchmark.py` — Runs all 6 probes head-to-head, computes probe speed, node overlap, 40-task verifiable scores, and generates `results/christmas_tree_heatmap.svg`.
- `results/PROBE_BENCHMARK_REPORT.md` — Full head-to-head leaderboard and takeaways.
- `results/christmas_tree_heatmap.svg` — Visual 30-layer "Christmas tree" heatmap showing where problem nodes light up across probes.
