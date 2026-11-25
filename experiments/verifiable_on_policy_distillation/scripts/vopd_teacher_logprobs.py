#!/usr/bin/env python3
"""Experiment 6: Verifiable On-Policy Distillation (VOPD, arXiv:2611.30325) Data & Weight Builder.

Takes student rollouts (prompt, completion, verified_reward in {0, 1}), queries a stronger teacher
on Nebius Token Factory (e.g. openai/gpt-oss-120b or Qwen/Qwen3-30B-A3B-Instruct-2507) for
critique/corrected reference or logprob agreement, and emits VOPD-weighted training JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_TEACHER = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def compute_vopd_weight(verified_reward: float, teacher_agrees: bool, alpha: float = 0.5) -> float:
    """Adaptive teacher-verifier agreement extrapolation weight (VOPD intuition)."""
    if verified_reward >= 0.5 and teacher_agrees:
        return 1.0 + alpha
    if verified_reward >= 0.5 and not teacher_agrees:
        return 1.0
    if verified_reward < 0.5 and not teacher_agrees:
        return alpha  # Teacher corrects student error -> distill teacher's verified correction
    return 0.0  # Both wrong -> drop from distillation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, help="JSONL with prompt, completion, reward (0/1)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Skip live API calls; verify pipeline locally")
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise SystemExit("NEBIUS_API_KEY is required unless --dry-run is set.")
        client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=api_key)

    rows = [json.loads(l) for l in Path(args.rollouts).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_rows = []
    for row in rows:
        reward = float(row.get("reward", 1.0))
        if args.dry_run:
            teacher_agrees = reward >= 0.5
            target_text = row.get("completion", "")
        else:
            resp = client.chat.completions.create(
                model=args.teacher_model,
                messages=[
                    {"role": "system", "content": "Return JSON: {\"agrees\": true|false, \"better_response\": \"...\"}"},
                    {"role": "user", "content": f"Prompt:\n{row['prompt']}\n\nStudent:\n{row['completion']}"},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            parsed = json.loads(resp.choices[0].message.content)
            teacher_agrees = bool(parsed.get("agrees", False))
            target_text = row["completion"] if reward >= 0.5 else parsed.get("better_response", row["completion"])

        w = compute_vopd_weight(reward, teacher_agrees, args.alpha)
        if w > 0.0:
            out_rows.append(
                {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": target_text},
                    ],
                    "vopd_weight": round(w, 3),
                    "verified_reward": reward,
                    "teacher_agrees": teacher_agrees,
                }
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"[OK] Wrote {len(out_rows)}/{len(rows)} VOPD-weighted rows -> {args.out}")


if __name__ == "__main__":
    main()
