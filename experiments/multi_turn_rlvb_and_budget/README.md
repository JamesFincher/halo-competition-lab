# Experiment 10: Multi-Turn Verifiable Beliefs (RLVB) + Anti-Budget-Internalization RLVR

**Priority:** #1 Multi-Turn Agentic & Test-Time Scaling Track (**8.0/10**)  
**Target Models:** `Qwen/Qwen3-4B` (pilot) → `unsloth/gpt-oss-20b-BF16` & `google/gemma-4-26B-A4B-it` (Halo environmental/async GRPO)

## Core Hypothesis
Two newly discovered Nov 2026 failure modes silently break multi-turn agentic & reasoning GRPO:
1. **Agentic Laziness (`arXiv:2611.32884`, Nov 24, 2026):** Outcome-only GRPO/PPO frequently falls *below the base model* on multi-turn tasks because agents guess prematurely while their own belief state still names multiple candidates. **RLVB** rewards continuing exploration while the environment's exact posterior has >1 plausible answer, and committing only once evidence is sufficient — beating base by **+17.1%** and PPO+PRM by **+9.3%** in **1.6× fewer GPU-hours**, with no judge or critic.
2. **Budget Internalization (`arXiv:2611.26172`, Nov 19, 2026):** Verified directly on **`Qwen3.5-4B`, `Gemma4-E2B`, and `Nemotron-3-Nano-4B`** — training RLVR with a single fixed `max_completion_length` causes the model to internalize that cap and **underperform the base model at larger test-time budgets**. Multi-budget / non-truncated rollout scheduling preserves test-time scaling.

## Supporting Literature
- **`arXiv:2611.32884` (*Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs*, Nov 24, 2026):** Exact environment posterior reward eliminates premature commitment (0.1–2.2% premature commits vs 3–68% base).
- **`arXiv:2611.26172` (*Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy*, Nov 19, 2026):** Links test-time scaling collapse to length-censored training rollouts across Qwen, Gemma 4, and Nemotron families.
- **`arXiv:2611.32013` (*To Critic Or Not To Critic*, Jason Wei et al., Nov 24, 2026):** Clarifies when critic-free GRPO vs critic-backed PPO wins on long-horizon tasks.

## Halo v1.3.0rc1 Compatibility
- Uses Halo's native **async environmental GRPO** (`examples/grpo/environmental/`) with Ray actors for multi-turn tool/observation rollouts.
- Avoids artificially truncated completions by stepping `max_completion_length` across stages (e.g. `1024 → 2048 → 4096`) and masking truncated traces rather than forcing premature answers.

## Files
- `scripts/rlvb_and_budget_schedule.py` — Deterministic RLVB posterior-sufficiency reward function (zero judge/critic) + multi-budget completion-length scheduler to prevent budget internalization.
- `configs/qwen3-4b-multi-budget-rlvb-grpo.yaml` — Halo online/environmental GRPO config with non-truncated rollout settings.

## Compute & Go/No-Go Criteria
- **Pilot cost (`Qwen3-4B`, 150 steps):** ~$20–$40.
- **GO:** RLVB + multi-budget schedule beats outcome-only fixed-cap GRPO on BOTH in-budget accuracy AND 2×/4× test-time token budgets.
