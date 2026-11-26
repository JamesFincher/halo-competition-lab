# Experiment 53 — Dual-Sided Internal Advantage Sculpting

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **4** + **arXiv:2610.40998** (*Think, Then Verify: Internal-Guided Process-Level RL*), **arXiv:2604.01476** (*Beyond Outcome Reward: Decoupling and Guiding Internal States*), and **arXiv:2611.30627** (*Attention Process Reward Model*).

## Hypothesis

Exact-compliance rewards are sparse and confound two failures: missing correct routing and generic over-defensiveness. Small verifier-dominated advantage corrections along two internal directions can reduce dead groups and produce more stable routing without weakening true safety refusals.

## Method

1. Start from the fp32 DLA top-64 layers-29/28 surgical edit at strength 0.90.
2. Estimate layer-29 decision-anchor directions:
   - `v_compliance`: correct routing minus incorrect routing.
   - `v_overrefusal`: benign false refusal minus benign correct allow, orthogonalized against a true-unsafe direction.
3. Train only disjoint-layer steering vectors `[0,1,24,25,26,27]`; all weights remain frozen.
4. Before group normalization, add `lambda_plus * z_compliance` only to verifier-wrong rollouts and subtract `lambda_minus * max(0, z_overrefusal)`.
5. Sweep verifier-only, balanced `(0.1,0.1)`, safety-heavy `(0.1,0.3)`, and compliance-heavy `(0.3,0.1)` for 8 steps.

## Run

```bash
python3 experiments/exp53_dual_sided_advantage_sculpting/scripts/run_dual_sided_advantage_rl.py
```

## Output

`results/dual_sided_advantage_summary.json` plus `results/step_metrics.jsonl` — raw verifier reward, shaped reward, dead-group count, internal z-scores, KL, gradients, exact neuron indices, direction statistics/orthogonalization cosine, final held-out metrics, wall time, and device. Graded per `research/RUBRIC.md`.
