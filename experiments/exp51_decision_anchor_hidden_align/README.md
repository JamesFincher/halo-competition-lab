# Experiment 51 — Decision-Anchor Hidden-Align RL

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **2** + **arXiv:2606.03234** (*Right Makes Might: Aligning Verified Hidden States Empowers RL Reasoning*) with internal-process support from **arXiv:2611.30627** (*From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning*).

## Hypothesis

Strict JSON tool routing has an exact decision boundary: the hidden state immediately before the `call`/`ref` action token. If correct compliant rollouts are explicitly aligned at that anchor, surgically anchored vector RL should become more stable and generalize better without ever treating false-refusal or unsafe-pass rollouts as positive examples.

## Method

1. Start from the fp32 DLA top-64 layers-29/28 surgical edit at strength 0.90.
2. Freeze all weights and train only disjoint-layer steering vectors `[0,1,24,25,26,27]`.
3. For each rollout, locate the first generated `call`/`ref` token and capture the preceding hidden state with a layer hook.
4. Add a pairwise cosine alignment loss over **exact-compliant rollouts only**.
5. Compare vector-only control against late-layer alignment (`L29`, lambda 1e-3 and 1e-2) and middle-layer alignment (`L20`, lambda 1e-3), all capped at 8 RL steps.

## Run

```bash
python3 experiments/exp51_decision_anchor_hidden_align/scripts/run_hidden_align_rl.py
```

## Output

`results/hidden_align_rl_summary.json` plus `results/step_metrics.jsonl` — per-step reward, exact-rollout rate, alignment cosine, aligned-pair count, missing-anchor count, KL, gradients, final vector norms, held-out metrics, exact neuron indices, wall time, and device. Graded per `research/RUBRIC.md`.
