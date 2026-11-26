# Experiment 31 — Activation-Vector-Only RL on Frozen Weights

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` (steering-vector-guided / activation-space RL) + **arXiv:2611.26627**, *RL Post-Training in Diffusion Transformers Is Activation Steering*.

## Hypothesis

If most of policy-gradient RL's behavioral gain can be represented as a constant residual-stream shift, then a tiny set of trainable activation vectors should be able to improve the policy model while all 139M weights remain frozen. This is a radically cheaper and more composable alternative to weight-space RL and a direct test of whether our DLA edit is simply discovering a steering direction that RL can learn online.

## Method

1. Load the verified Halo SFT checkpoint (`checkpoint-36`) merged into SmolLM2-135M-Instruct.
2. Compare two fast arms, each capped at 8 GRPO-style steps (hard max 10):
   - `vector_only_rl`: train zero-initialized residual-stream vectors at layers 27/28/29 on the unedited SFT model.
   - `edit_plus_vector_rl`: apply the lab's DLA top-64 layers 29/28 surgical edit, then train a layer-27 residual vector.
3. Optimize only vector parameters with AdamW (`weight_decay=0.0`), group-centered advantages, and KL anchoring to each arm's initial checkpoint.
4. Evaluate on the 40-task held-out policy set with the standard exact-compliance / unsafe-refusal / benign-false-refusal metrics.

## Run

```bash
python3 experiments/activation_vector_only_rl/scripts/run_activation_vector_rl.py
```

## Output

`results/activation_vector_rl_summary.json` — baseline metrics, DLA-edit metrics, per-step reward/KL/grad/vector-norm telemetry, learned vectors, exact neuron indices, wall time, and device. Graded per `research/RUBRIC.md`.
