#!/usr/bin/env python3
"""Hypothesis B — MoE Expert-Routing Localization & EP-Rank-Aware Gradient Masking Utilities.

Grounded in:
- ESFT (arXiv:2407.01906): update only task-specialized MoE experts to prevent OOD forgetting
- Structuring MoE Expert Selection (arXiv:2610.07332): preserve natural routing specialization
- Router Prior Bias / Router-L2 (arXiv:2609.08115): soft router anchoring (never hard-clamp!)
- Router Drift Control / RDC (arXiv:2610.21682): filter high rollout-vs-trainer TV-drift sequences

Halo v1.3.0rc1 Safety Rules:
- EPMoELayerBase stores local experts as fused 3D tensors [E_local, K, N].
- NEVER set requires_grad=False asymmetrically across EP ranks (deadlocks DeepEP combine).
- Instead, register an EP-rank-aware gradient-masking hook keeping requires_grad=True uniformly
  and set weight_decay=0.0 in Halo config (prevents AdamWBF16 silent decay on masked slices).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def select_contrastive_experts(
    pass_counts: dict[int, list[float]],
    fail_counts: dict[int, list[float]],
    top_k_per_layer: int,
    alpha_freq: float = 0.35,
) -> dict[int, list[int]]:
    """Rank experts per layer by contrastive shift (p_pass - p_fail) + frequency mass."""
    selected: dict[int, list[int]] = {}
    for layer_idx, p_cnt in pass_counts.items():
        f_cnt = fail_counts[layer_idx]
        p_tot = sum(p_cnt) + 1e-9
        f_tot = sum(f_cnt) + 1e-9
        scores = []
        for e in range(len(p_cnt)):
            p_e = p_cnt[e] / p_tot
            f_e = f_cnt[e] / f_tot
            score = (p_e - f_e) + alpha_freq * p_e
            scores.append((score, e))
        scores.sort(reverse=True)
        selected[layer_idx] = sorted([e for _s, e in scores[:top_k_per_layer]])
    return selected


def register_ep_expert_grad_mask(
    model: torch.nn.Module,
    selected_global_experts: dict[int, list[int]],
    ep_rank: int,
    experts_per_rank: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach post-backward gradient masks on fused 3D [E_local, ...] expert parameters/LoRA tensors."""
    handles = []
    local_start = ep_rank * experts_per_rank
    local_end = local_start + experts_per_rank

    for name, p in model.named_parameters():
        if not p.requires_grad or p.ndim != 3 or p.shape[0] != experts_per_rank:
            continue
        # Parse layer index from parameter name (e.g. model.layers.12.mlp.experts.down_proj_lora_A)
        parts = name.split(".")
        layer_idx = None
        for i, tok in enumerate(parts[:-1]):
            if tok == "layers" and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None or layer_idx not in selected_global_experts:
            continue

        local_mask = torch.zeros(experts_per_rank, 1, 1, dtype=p.dtype, device=p.device)
        for g_id in selected_global_experts[layer_idx]:
            if local_start <= g_id < local_end:
                local_mask[g_id - local_start] = 1.0

        def _make_hook(m: torch.Tensor):
            return lambda grad: grad * m.to(device=grad.device, dtype=grad.dtype)

        handles.append(p.register_hook(_make_hook(local_mask)))
    return handles


def router_l2_anchor_loss(
    current_router_weights: list[torch.Tensor],
    base_router_weights: list[torch.Tensor],
    lam: float = 1e3,
) -> torch.Tensor:
    """Simple parameter-space Router-L2 anchor (arXiv:2609.08115: matches RPB within 0.6 pts)."""
    loss = torch.zeros((), device=current_router_weights[0].device, dtype=torch.float32)
    for cur, base in zip(current_router_weights, base_router_weights):
        loss = loss + torch.sum((cur.float() - base.to(cur.device).float()) ** 2)
    return lam * loss


def rdc_filter_mask(tv_drifts: torch.Tensor, drop_top_frac: float = 0.10) -> torch.Tensor:
    """Router Drift Control (arXiv:2610.21682): drop top-q highest rollout-vs-trainer TV-drift responses."""
    if tv_drifts.numel() == 0:
        return torch.ones_like(tv_drifts, dtype=torch.bool)
    thresh = torch.quantile(tv_drifts.float(), 1.0 - drop_top_frac)
    return tv_drifts.float() <= thresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-stats-json", required=True, help="JSON with pass_counts & fail_counts per layer")
    parser.add_argument("--top-k-experts", type=int, default=8, help="Experts to unfreeze per layer (e.g. 8 of 32)")
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    raw = json.loads(Path(args.routing_stats_json).read_text(encoding="utf-8"))
    pass_c = {int(k): v for k, v in raw["pass_counts"].items()}
    fail_c = {int(k): v for k, v in raw["fail_counts"].items()}
    selected = select_contrastive_experts(pass_c, fail_c, args.top_k_experts)
    Path(args.out_file).write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"[OK] Selected top-{args.top_k_experts} experts across {len(selected)} layers -> {args.out_file}")


if __name__ == "__main__":
    main()
