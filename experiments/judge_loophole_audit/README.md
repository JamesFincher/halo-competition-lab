# Experiment 8: Pre-RL Judge Loophole Exposure Audit (Reward-Hacking Defense)

**Priority:** Mandatory Pre-Flight Gate for Any LLM-Judge GRPO Run (**8.5/10** operational value)  
**Target Judges:** White Circle policy reward, OpenRouter LLM judge, Nebius Token Factory judge

## Core Hypothesis
In GRPO against an LLM judge, the **first loophole the policy will exploit can be predicted before RL step 0** via each degenerate pattern's **exposure** (`exposure = base_policy_propensity × judge_FPR`). Auditing exposure before launching a multi-GPU Halo run identifies high-exposure judge loopholes (e.g. lone `{`, `"Thought process:"`, empty refusal boilerplate, length padding) so we can add deterministic rule-masks before spending GPU hours.

## Supporting Literature
- **`arXiv:2611.29651` (*Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges Before RL Training*, Nov 22, 2026):** Across 36 GRPO runs with 7 judges, the first exploited loophole had the **highest exposure in 12/12 runs with exploitation** (vs only 6/12 for raw FPR alone). Exploitation occurs specifically on hard prompts where an accepted exploit gets a large group-normalized advantage.
- **`arXiv:2611.32037` (*Online Environment Repair for Coding Agents*, Nov 24, 2026):** Complementary defense for programmatic coding verifiers with incomplete unit tests.

## Halo v1.3.0rc1 Integration
- Run `scripts/audit_judge_exposure.py` BEFORE launching `configs/gemma4-26b-a4b-policy-grpo-whitecircle.yaml` or any OpenRouter-judged GRPO recipe.
- Any loophole with `exposure > threshold` gets a zero-reward deterministic pre-filter in the reward wrapper (eliminating exploit onset without retraining the judge).

## Files
- `scripts/audit_judge_exposure.py` — Computes base-policy propensity × judge FPR exposure scores across a library of degenerate exploit probes (supports `--dry-run` local check or live Token Factory / White Circle judge).
- `configs/qwen3-4b-hardened-judge-grpo.yaml` — Halo GRPO config paired with pre-audited rule-masked reward filtering.

## Compute & Go/No-Go Criteria
- **Audit cost:** <$1 in API calls + <10 min on 1× GPU before launching multi-GPU RL.
- **GO:** Every high-exposure loophole (`exposure > 0.01`) is rule-masked to `reward = 0.0` before step 0, and 0 exploit collapses occur across 200 GRPO steps on hard prompts.
