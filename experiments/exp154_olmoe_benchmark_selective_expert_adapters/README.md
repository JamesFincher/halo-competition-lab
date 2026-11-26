# exp154 — benchmark-selective OLMoE expert adapters (staged)

Not autoqueued: the script is intentionally named `staged_olmoe_expert_adapters.py` rather than `run_*.py`. Activate only after exp153 stage-zero confirms real OLMoE memory, wall-clock, tokenizer, router capture, and cache behavior.

Planned comparison: zero-init low-rank down-projection adapters on benchmark-selective answer-token experts versus parameter-matched random experts and diffuse all-expert rank-1 controls. Base weights and routers frozen; supervised answer-only direct-continuation training; behavior collateral and continuation benchmarks measured before/after; cached-vs-uncached generation equality checked after training.
