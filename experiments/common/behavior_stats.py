"""Paired behavioral statistics for matched row-level evaluation artifacts."""

from __future__ import annotations

import math
import random
from typing import Iterable


BINARY_METRICS = ("exact", "json_valid", "action_ok", "policy_ok")


def require_matched_rows(rows_a: list[dict], rows_b: list[dict]) -> None:
    if len(rows_a) != len(rows_b):
        raise ValueError(f"paired comparison requires equal row counts: {len(rows_a)} vs {len(rows_b)}")
    for i, (a, b) in enumerate(zip(rows_a, rows_b)):
        if a.get("index", i) != b.get("index", i):
            raise ValueError(f"row order mismatch at position {i}")
        if a.get("target_action") != b.get("target_action") or a.get("target_tool") != b.get("target_tool"):
            raise ValueError(f"matched eval rows diverge at position {i}")


def binary_values(rows: Iterable[dict], metric: str) -> list[int]:
    if metric not in BINARY_METRICS:
        raise ValueError(f"unsupported binary metric {metric!r}; expected one of {BINARY_METRICS}")
    return [int(bool(row.get(metric))) for row in rows]


def mcnemar_exact(rows_a: list[dict], rows_b: list[dict], metric: str = "exact") -> dict:
    """Exact two-sided McNemar test for paired binary row outcomes."""
    require_matched_rows(rows_a, rows_b)
    avals = binary_values(rows_a, metric)
    bvals = binary_values(rows_b, metric)
    b = sum(1 for a, x in zip(avals, bvals) if a == 1 and x == 0)  # a-only successes
    c = sum(1 for a, x in zip(avals, bvals) if a == 0 and x == 1)  # b-only successes
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        cumulative = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p_value = min(1.0, 2.0 * cumulative)
    return {
        "metric": metric,
        "a_only_successes": b,
        "b_only_successes": c,
        "discordant_pairs": n,
        "exact_two_sided_p": round(p_value, 6),
    }


def paired_bootstrap(
    rows_a: list[dict],
    rows_b: list[dict],
    metric: str = "exact",
    iterations: int = 10000,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for rows_a rate minus rows_b rate, in percentage points."""
    require_matched_rows(rows_a, rows_b)
    avals = binary_values(rows_a, metric)
    bvals = binary_values(rows_b, metric)
    n = len(avals)
    if n == 0:
        raise ValueError("paired bootstrap requires nonempty rows")
    observed = 100.0 * (sum(avals) - sum(bvals)) / n
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(100.0 * sum(avals[i] - bvals[i] for i in idx) / n)
    diffs.sort()

    def percentile(q: float) -> float:
        pos = min(len(diffs) - 1, max(0, int(round(q * (len(diffs) - 1)))))
        return diffs[pos]

    return {
        "metric": metric,
        "n_rows": n,
        "observed_diff_pp": round(observed, 3),
        "bootstrap95_pp": [round(percentile(0.025), 3), round(percentile(0.975), 3)],
        "iterations": iterations,
    }


def compare_behavior_rows(rows_a: list[dict], rows_b: list[dict], metric: str = "exact") -> dict:
    return {
        "paired_bootstrap": paired_bootstrap(rows_a, rows_b, metric=metric),
        "mcnemar": mcnemar_exact(rows_a, rows_b, metric=metric),
    }


def comparisons_against_reference(
    arms: list[dict],
    reference_arm: str,
    measurement_key: str = "behavior_scaled120",
    metric: str = "exact",
) -> dict:
    """Compare every arm with matched row outcomes against one reference arm."""
    by_arm = {arm.get("arm"): arm for arm in arms}
    reference = by_arm.get(reference_arm, {}).get("measurement", {}).get(measurement_key, {}).get("rows")
    if not reference:
        return {}
    out = {}
    for arm in arms:
        name = arm.get("arm")
        if name == reference_arm:
            continue
        rows = arm.get("measurement", {}).get(measurement_key, {}).get("rows")
        if not rows:
            continue
        try:
            out[name] = compare_behavior_rows(reference, rows, metric=metric)
            out[name]["direction"] = f"{reference_arm}_minus_{name}"
        except ValueError as exc:
            out[name] = {"error": str(exc)}
    return out
