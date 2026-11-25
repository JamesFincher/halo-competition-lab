# Experiment 6: Verifiable On-Policy Distillation (VOPD) + Token Factory Teacher

**Priority:** High-Leverage Data/Distillation Companion (**8.0/10** for Nemotron 3.5 / Mistral / Gemma tracks where API credits shine)  
**Target Models:** Student = `Qwen/Qwen3-4B`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, or `unsloth/gpt-oss-20b-BF16`; Teacher = Nebius Token Factory (`openai/gpt-oss-120b` or `Qwen/Qwen3-235B-A22B-Instruct-2507`)

## Core Hypothesis
Standard RLVR gives only a sparse sequence-level pass/fail reward, while standard On-Policy Distillation (OPD) provides dense next-token teacher feedback that is ungrounded in verified correctness. **Verifiable On-Policy Distillation (VOPD, `arXiv:2611.30325`, Stanford Nov 23, 2026)** projects verifier preference onto the teacher's implicit-reward direction, turning sparse verifier outcomes into dense, verifier-grounded next-token supervision — and directly leverages our **Nebius Token Factory / OpenRouter API credits** on models where online GRPO is restricted (e.g., `Nemotron 3.5 Lightning`).

## Supporting Literature
- **`arXiv:2611.30325` (*Verifiable On-Policy Distillation*, Juntong Shi, Stefano Ermon, Jure Leskovec, Minkai Xu, Nov 23, 2026):** Outperforms both RLVR and vanilla OPD across 8 mathematical reasoning benchmarks by adaptively extrapolating teacher targets based on teacher–verifier agreement. Official code: `https://github.com/juntongshi48/VOPD`.
- **`arXiv:2611.32515` (*Branch and Consolidate: Multi-Teacher On-Policy Distillation*, Nov 24, 2026):** Consolidates domain-specialized teachers cleanly without capability interference.
- **`arXiv:2608.24696` (*On-policy Distillation with Verifiable Reward*, Aug 2026):** Earlier OPD+VR foundation.

## Halo v1.3.0rc1 Compatibility
- Halo natively supports distillation (`scripts/training/distillation.py`) and offline GRPO/SFT across **all** partner models — including `Nemotron 3.5 Lightning` and `Mistral Small 4 119B` where online vLLM rollout sync is unavailable.
- Composes directly with Experiment 1/2: apply VOPD loss restricted to the localized top-25% layers (`unfreeze_layers_patterns` or bare YAML regex LoRA).

## Files
- `scripts/vopd_teacher_logprobs.py` — Queries Nebius Token Factory (`openai/gpt-oss-120b` or `Qwen/Qwen3-30B-A3B-Instruct-2507`) on student rollouts, pairs with deterministic verifier reward, and computes VOPD teacher–verifier agreement weights for Halo distillation / weighted SFT.
- `configs/qwen3-4b-vopd-distill.yaml` — Halo distillation/SFT config consuming VOPD-weighted trajectories.

## Compute & Go/No-Go Criteria
- **Teacher API cost:** <$5 on Nebius Token Factory for 1,000–2,000 verified rollouts.
- **GO:** VOPD beats vanilla SFT/OPD and matches or beats pure GRPO in fewer GPU-hours, especially on `Nemotron 3.5 Lightning`.
