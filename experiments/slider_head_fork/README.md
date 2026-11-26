# Slider Head Fork — Iterative Attention-Capability Research

Dedicated fork for pushing zero-init slider-gated attention heads as far as they can go on real benchmarks. Each variant trains additive heads (core weights frozen, exact-zero init verified), logs detailed checkpoints (weights + alphas + telemetry) in `checkpoints/<variant>/`, and gets scored on MMLU/ARC-Easy/HellaSwag + behavioral eval. Variants: v1 single-head baseline, v2 four-head late bank, v3 deep spread, v4 wide heads, v5 memory cross-attention, v6 full wide bank. User-facing iteration log: FORK_LEADERBOARD.md.
