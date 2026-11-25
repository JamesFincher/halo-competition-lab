#!/usr/bin/env python3
"""Experiment 10: RLVB Verifiable Belief Reward (arXiv:2611.32884) & Anti-Budget-Internalization
Rollout Scheduler (arXiv:2611.26172).

Provides:
1. `rlvb_step_reward(...)`: exact environment-posterior reward (no judge/critic) that rewards
   gathering more evidence when posterior entropy/candidates > 1 and committing when sufficient.
2. `sample_completion_budget(...)`: avoids fixed-cap budget internalization across Qwen/Gemma/Nemotron.
"""

from __future__ import annotations

import argparse
import json
import random


def rlvb_step_reward(
    committed: bool,
    correct: bool,
    posterior_candidates: int,
    step_bonus: float = 0.1,
    premature_penalty: float = -0.5,
) -> float:
    """Exact verifiable-belief reward (arXiv:2611.32884)."""
    sufficient = posterior_candidates <= 1
    if not committed:
        return step_bonus if not sufficient else -step_bonus
    if sufficient and correct:
        return 1.0
    if not sufficient:
        # Penalize lazy premature guessing even if lucky
        return premature_penalty + (0.2 if correct else 0.0)
    return 0.0


def sample_completion_budget(step: int, total_steps: int, budgets: tuple[int, ...] = (1024, 2048, 4096), seed: int = 42) -> int:
    """Curriculum + randomized rollout cap to prevent budget internalization (arXiv:2611.26172)."""
    rng = random.Random(seed + step)
    progress = step / max(total_steps, 1)
    if progress < 0.3:
        return rng.choice(budgets[:2])
    return rng.choice(budgets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    demo = {
        "rlvb_examples": {
            "explore_when_uncertain": rlvb_step_reward(committed=False, correct=False, posterior_candidates=4),
            "lazy_lucky_guess": rlvb_step_reward(committed=True, correct=True, posterior_candidates=4),
            "evidence_backed_commit": rlvb_step_reward(committed=True, correct=True, posterior_candidates=1),
        },
        "multi_budget_schedule": [sample_completion_budget(s, args.steps) for s in range(args.steps)],
    }
    print(json.dumps(demo, indent=2))


if __name__ == "__main__":
    main()
