# Experiment 83 — Self-Modeling Failure Probes as RL Advantage Routers

**Research inspiration:** `research/SUBAGENT3_CREATIVE_HEURISTICS.md` rank **9** + **arXiv:2207.05221** (*models often encode whether they know*), **arXiv:2306.03341** (*inference-time intervention from internal truthfulness directions*), and **arXiv:2303.08896** (*self-consistency reference*).

## Hypothesis

The model may internally encode imminent failure before the external verifier scores it. A tiny calibrated probe at the decision token can route RL credit toward high-risk rollouts and away from saturated correct behavior, improving sample efficiency without letting the probe override safety or the verifier.

## Method

1. Start from the fp32 validated rank-1 surgical edit.
2. Sample train rollouts and label each decision-token hidden state with deterministic verifier failure classes: correct, false refusal, missed unsafe, wrong tool/policy, wrong action, malformed JSON.
3. Train binary failure-risk logistic probes at layers 18/24/29 with 4-fold prompt-index holdout validation; select the layer by cross-validated AUROC.
4. Train disjoint steering vectors in three arms:
   - uniform advantage control
   - probe-routed high-risk upweighting / saturated-correct downweighting
   - random-risk control
5. Probe calibration is reported separately; it is never used to override hard unsafe-refusal behavior at inference.

## Run

```bash
python3 experiments/exp83_self_modeling_failure_probe_router/scripts/run_failure_probe_router.py
```

## Output

`results/failure_probe_router_summary.json` plus `results/step_metrics.jsonl` — label counts, per-layer cross-validated AUROC, selected layer, mean routed risk, reward/dead-group/KL telemetry, final exact/safety metrics, wall time, and device.
