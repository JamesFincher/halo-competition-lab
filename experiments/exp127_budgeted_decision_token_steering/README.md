# Experiment 127 — Trajectory-Aware Budgeted Decision-Token Steering

**Research inspiration:** `research/HYPOTHESES.md` **H11** + **arXiv:2611.18274** (*Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering*).

## Hypothesis

Steering should act like a budgeted per-token intervention, not an always-on vector. Decision-position and trajectory-aware allocation can capture most behavioral benefit with fewer interventions and less schema/distribution damage.

## Method

1. Measure a signed false-refusal direction at layer 28 from benign correct-allow versus counterfactual false-refusal teacher-forced states.
2. Cache rollouts and compare allocation policies at matched token budgets:
   - always-on;
   - prompt-only;
   - fixed positions after `action`, `tool`, and `policy_code` schema markers;
   - memoryless contrastive-alignment gate;
   - closed-loop adaptive Bernoulli steering from recent trajectory alignment;
   - random-token control.
3. Select strength/budget configurations only on offset train-dev rows.
4. Evaluate the best config within each allocation family once on untouched `policy_eval`.
5. Report exact compliance, unsafe refusal, benign false refusal, intervention rates, alignment telemetry, latency proxy, and closed-loop pass@4.

## Run

```bash
python3 experiments/exp127_budgeted_decision_token_steering/scripts/run_budgeted_decision_steering.py
```

## Output

`results/budgeted_decision_steering_summary.json` — direction diagnostics, full dev grid, per-policy selected configs, untouched endpoint metrics, intervention statistics, pass@4, wall time, and device.
