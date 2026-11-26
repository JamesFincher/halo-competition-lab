# Experiment 85 — Evolution Strategies in an Interpretable 32-D Surgical Subspace

**Research inspiration:** `research/SUBAGENT3_CREATIVE_HEURISTICS.md` rank **5** + **arXiv:1703.03864** (*Evolution Strategies as a scalable alternative to RL*) and **arXiv:2012.13255** (*forward-only zeroth-order optimization*).

## Hypothesis

If the validated surgical edit is parameterized as independent per-neuron strength coefficients, black-box evolution can find nonuniform dose structure that uniform strength sweeps miss—without gradient RL or backpropagating through the policy objective.

## Method

1. Work in fp32 with validated rank-1 residual-direction subspaces at layers 29/28.
2. Select the top 16 projection-energy `down_proj` columns per layer, yielding an interpretable 32-D coefficient vector.
3. Optimize coefficients in `[0,1.5]` with rank-centered OpenAI-ES on a train-dev split offset from the direction-estimation prefix.
4. Compare uniform dose references, random search with matched evaluation count, ES best, and ES final center.
5. Touch `policy_eval` only for final endpoint evaluation.

## Run

```bash
python3 experiments/exp85_evolution_strategies_32d_surgical_subspace/scripts/run_es32_surgical_subspace.py
```

## Output

`results/es32_surgical_subspace_summary.json` — coordinate map, dev protocol, uniform dose references, full ES/random-search histories, best coefficient vectors, untouched final policy metrics, wall time, and device.
