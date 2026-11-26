"""Defensive JSON serialization for experiment results.

GPU experiments are expensive; serialization must never be the reason a completed
run is lost. These helpers normalize tensors, numpy scalars/arrays, scalar-like
objects, NaN/Inf, and tuples before writing. They are intentionally conservative:
unexpected objects become repr strings rather than raising after compute.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


def scalar_float(value: Any) -> float:
    """Best-effort scalar conversion that accepts python scalars and torch tensors.

    Multi-element tensors are reduced by mean rather than crashing; callers use this
    only for metrics that are intended to be scalar. This prevents post-compute
    serialization/metric type mismatches from destroying an otherwise completed run.
    """
    if torch.is_tensor(value):
        v = value.detach().float().cpu()
        if v.numel() == 0:
            return 0.0
        return float(v.mean().item()) if v.numel() > 1 else float(v.item())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return scalar_float(value.item())
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_round(value: Any, ndigits: int = 5):
    """Round scalar-like values; serialize vector-like tensor/array values as lists."""
    if torch.is_tensor(value):
        v = value.detach().float().cpu()
        if v.numel() == 0:
            return []
        if v.numel() == 1:
            return round(float(v.reshape(-1)[0].item()), ndigits)
        return [round(float(x), ndigits) for x in v.reshape(-1).tolist()]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return safe_round(torch.as_tensor(value), ndigits)
        if isinstance(value, np.generic):
            return round(float(value), ndigits)
    except Exception:
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return round(float(value.item()), ndigits)
        except Exception:
            pass
    try:
        return round(float(value), ndigits)
    except Exception:
        return repr(value)


def json_safe(obj: Any) -> Any:
    if torch.is_tensor(obj):
        v = obj.detach().cpu()
        if v.numel() == 0:
            return []
        if v.numel() == 1:
            return json_safe(v.reshape(-1)[0].item())
        return v.tolist()
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return json_safe(obj.item())
        except Exception:
            pass
    return repr(obj)


def dumps_safe(obj: Any, **kwargs) -> str:
    return json.dumps(json_safe(obj), **kwargs)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_safe(obj, indent=2), encoding="utf-8")
