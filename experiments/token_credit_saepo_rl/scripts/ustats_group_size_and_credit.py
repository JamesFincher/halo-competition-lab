#!/usr/bin/env python3
"""Experiment 7: U-Statistics Pessimistic Jackknife Group-Size Selector (arXiv:2611.33064)
& Token-Level Credit-Assignment Helper (SAEPO, arXiv:2611.29936).

Given pilot rollout rewards for each prompt, computes the jackknife pessimistic variance-ratio
estimate to allocate larger GRPO group sizes G_i to high-uncertainty prompts and skip/shrink
degenerate all-pass or all-fail prompts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def jackknife_pessimistic_group_size(
    rewards: list[float],
    g_min: int = 4,
    g_max: int = 16,
    z_pessimism: float = 1.0,
) -> dict:
    """Estimate optimal GRPO group size G using leave-one-out jackknife variance uncertainty."""
    n = len(rewards)
    if n < 2:
        return {"mean": rewards[0] if rewards else 0.0, "std": 0.0, "recommended_G": g_max, "degenerate": True}

    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / (n - 1)
    if var < 1e-12:
        # All-pass or all-fail group (zero GRPO advantage) -> use minimum G or filter out
        return {"mean": round(mean, 4), "std": 0.0, "recommended_G": g_min, "degenerate": True}

    # Leave-one-out jackknife variances to capture denominator uncertainty
    jk_vars = []
    for i in range(n):
        sub = [rewards[j] for j in range(n) if j != i]
        m_i = sum(sub) / len(sub)
        v_i = sum((r - m_i) ** 2 for r in sub) / max(len(sub) - 1, 1)
        jk_vars.append(v_i)

    jk_mean = sum(jk_vars) / n
    jk_se = math.sqrt(((n - 1) / n) * sum((v - jk_mean) ** 2 for v in jk_vars))
    # Pessimistic lower bound on variance -> larger G when denominator variance is uncertain
    pessimistic_var = max(var - z_pessimism * jk_se, 1e-3)
    uncertainty_ratio = var / pessimistic_var

    # Map uncertainty ratio [1.0 .. 4.0+] cleanly into [g_min .. g_max] (even integers)
    scaled = g_min + int(round((g_max - g_min) * min(max(uncertainty_ratio - 1.0, 0.0) / 3.0, 1.0)))
    rec_g = max(g_min, min(g_max, scaled + (scaled % 2)))
    return {
        "mean": round(mean, 4),
        "std": round(math.sqrt(var), 4),
        "uncertainty_ratio": round(uncertainty_ratio, 4),
        "recommended_G": rec_g,
        "degenerate": False,
    }


def shape_token_advantages(
    seq_advantage: float,
    token_scores: list[float],
    lam: float = 0.25,
) -> list[float]:
    """SAEPO-style plug-in token-level credit shaping: A_t = A_seq + lam * normalized(s_t)."""
    if not token_scores:
        return []
    mean_s = sum(token_scores) / len(token_scores)
    std_s = math.sqrt(sum((s - mean_s) ** 2 for s in token_scores) / max(len(token_scores), 1)) + 1e-6
    return [round(seq_advantage + lam * ((s - mean_s) / std_s), 5) for s in token_scores]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-rewards-json", required=True, help="JSON list of {prompt_id, rewards: [0/1,...]}")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--g-min", type=int, default=4)
    parser.add_argument("--g-max", type=int, default=16)
    args = parser.parse_args()

    items = json.loads(Path(args.pilot_rewards_json).read_text(encoding="utf-8"))
    results = []
    for item in items:
        stat = jackknife_pessimistic_group_size(item["rewards"], args.g_min, args.g_max)
        results.append({"prompt_id": item["prompt_id"], **stat})

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(results, indent=2), encoding="utf-8")
    active = [r for r in results if not r["degenerate"]]
    print(f"[OK] Scored {len(results)} prompts ({len(active)} non-degenerate) -> {args.out_file}")


if __name__ == "__main__":
    main()
