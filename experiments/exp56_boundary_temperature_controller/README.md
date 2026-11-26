# Experiment 56 — Hidden-State-Conditioned Boundary-Token Temperature

**Research inspiration:** `research/SUBAGENT1_FRONTIER_RL_INTERNALS.md` rank **7** + **arXiv:2602.13035** (*IntroLLM: selective exploration at reasoning boundary tokens*).

## Hypothesis

In ultra-short strict JSON policies, schema punctuation should be nearly deterministic while only action/tool/policy-code boundary tokens need exploration. A tiny controller reading the layer-29 hidden state can choose low versus high temperature selectively, reducing dead groups without entropically corrupting the schema.

## Method

1. Start from the fp32 DLA top-64 layers-29/28 surgical edit at strength 0.90.
2. Instrument cached autoregressive rollouts and detect decision positions immediately after `"action": "`, `"tool": "`, or `"policy_code": "`.
3. Compare three vector-RL arms:
   - fixed temperature 0.85 everywhere
   - schema-frozen heuristic: 0.20 schema / 1.20 decision
   - learned controller: hidden \(576\to64\to2\), choosing low/high temperature at decision positions
4. Train disjoint steering vectors and controller by coordinate-ascent-style alternating updates with exact-compliance advantages.
5. Measure reward, dead-group rate, token entropy, KL, greedy exact compliance, pass@4 exact, unsafe refusal, and benign false refusal.

## Run

```bash
python3 experiments/exp56_boundary_temperature_controller/scripts/run_boundary_temperature_controller.py
```

## Output

`results/boundary_temperature_controller_summary.json` plus `results/step_metrics.jsonl` — per-step exploration/dead-group telemetry, controller decision counts, coordinate-update type, greedy and sampling evaluation, exact neuron indices, wall time, and device. Graded per `research/RUBRIC.md`.
