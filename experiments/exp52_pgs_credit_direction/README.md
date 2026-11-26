# Experiment 52 — PGS-Credit Direction Discovery for Surgical Edits

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **3** + **arXiv:2607.27574** (*Policy Gradient Steering: Interventions from Behavioral Objectives*) and **arXiv:2509.06608** (*Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors*).

## Hypothesis

Contrastive and DLA directions identify where refusal behavior is visible, but not necessarily which activation components control reward-relevant action tokens. Return-weighted policy-gradient steering \(v_l=E[A_t\nabla_{h_l}\log\pi(a_t|o_t)]\) may discover a better causal direction for the next surgical edit.

## Method

1. Start from the fp32 DLA top-64 layers-29/28 edited checkpoint.
2. Select high-variance benign and unsafe boundary prompts from the training split.
3. Sample grouped rollouts, compute exact-compliance advantages, and backpropagate only the action-token logprob.
4. Accumulate advantage-weighted hidden gradients at layers 20/28/29.
5. Compare direction cosines and causal effect in the standard rank-1 edit harness:
   - DLA/unembed reference
   - PGS directions from L20/L28/L29
   - rollout correct-vs-incorrect contrast direction
   - random direction control

## Run

```bash
python3 experiments/exp52_pgs_credit_direction/scripts/run_pgs_direction_discovery.py
```

## Output

`results/pgs_direction_summary.json` — prompt-selection telemetry, rollout usage/degenerate-group counts, direction cosines, exact neuron indices, causal held-out metrics for every direction, wall time, and device. Graded per `research/RUBRIC.md`.
