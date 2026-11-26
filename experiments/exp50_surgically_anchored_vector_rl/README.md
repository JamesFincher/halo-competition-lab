# Experiment 50 — Surgically Anchored Bias/Steering-Vector RL

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **1** + **arXiv:2505.18706** (*Steering LLM Reasoning Through Bias-Only Adaptation*), **arXiv:2609.18587** (*Label-free steering: Compressing test-time reinforcement learning into bias-only subspaces*), and **arXiv:2509.06608** (*Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors*).

## Hypothesis

The zero-gradient DLA edit already places the model near the correct refusal-vs-tool-call policy boundary. If RL is prevented from overwriting that mechanism and allowed only to learn additive residual-stream steering vectors, a few thousand parameters may harden the edit better than ordinary LoRA RL.

## Method

1. Load the verified Halo SFT checkpoint in fp32 (required after the wave-1 bf16 edit failure).
2. Apply the standard DLA top-64 neuron edit on layers 29/28 at strength 0.90.
3. Freeze every model weight and set the KL reference to this post-edit checkpoint.
4. Compare capped 8-step GRPO-style training across:
   - `all_layer_vectors`: vectors at all 30 block outputs (17,280 params)
   - `disjoint_layer_vectors`: vectors at layers `[0,1,24,25,26,27]` (3,456 params)
   - `random_layer_control`: six random non-edited layers
   - `lora_disjoint_control`: r=16 LoRA on the same complementary layers
5. Evaluate all arms on the 40-task held-out set with the standard verifier metrics.

## Run

```bash
python3 experiments/exp50_surgically_anchored_vector_rl/scripts/run_surgically_anchored_vector_rl.py
```

Use `--skip-lora-control` for the fastest vector-only sweep.

## Output

`results/surgically_anchored_vector_rl_summary.json` plus `results/step_metrics.jsonl` — baseline/edit metrics, exact layer and neuron indices, vector parameter counts and learned vectors, per-step reward/KL/gradient telemetry, held-out metrics, wall time, and device. Graded per `research/RUBRIC.md`.
