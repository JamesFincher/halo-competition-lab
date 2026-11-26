"""Shared validated surgical-edit operator.

The wave-1/2 program established that projecting along the unembedding logit
direction collapses the policy. The validated operator is the raw rank-1
residual-direction edit from experiments/whitened_subspace_rfm_edit: measure
canonical false-refusal minus compliant residual activations on benign training
prompts, take the leading SVD direction at layers 29/28, then project the top
projection-energy mlp.down_proj columns in fp32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/whitened_subspace_rfm_edit/scripts"))
from run_whitened_subspace_edit import apply_rank_k_subspace_edit, collect_sample_hidden_matrices  # noqa: E402


def build_validated_rank1_subspaces(
    model: torch.nn.Module,
    tokenizer,
    train_rows: list[dict],
    device: torch.device,
    edit_layers: list[int] | tuple[int, ...] = (29, 28),
    n_pairs: int = 12,
) -> tuple[dict[int, torch.Tensor], dict[str, list[float]]]:
    mats = collect_sample_hidden_matrices(model, tokenizer, train_rows, device, n_pairs=n_pairs)
    subspaces: dict[int, torch.Tensor] = {}
    singular_values: dict[str, list[float]] = {}
    for layer_id in edit_layers:
        false_minus_pass = mats["h_fail"][layer_id] - mats["h_pass"][layer_id]
        _u, s_vals, vh = torch.linalg.svd(false_minus_pass, full_matrices=False)
        subspaces[layer_id] = vh[:1, :].T.contiguous()
        singular_values[str(layer_id)] = [round(float(x), 4) for x in s_vals[:4].detach().cpu().tolist()]
    return subspaces, singular_values


def top_projection_columns(
    model: torch.nn.Module,
    subspaces: dict[int, torch.Tensor],
    edit_layers: list[int] | tuple[int, ...],
    top_columns: int,
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    with torch.no_grad():
        for layer_id in edit_layers:
            down = model.model.layers[layer_id].mlp.down_proj
            w = down.weight.detach().float()
            u = subspaces[layer_id].to(device=w.device, dtype=torch.float32)
            proj = torch.matmul(u.T, w)
            energy = (proj ** 2).sum(dim=0)
            k = min(top_columns, energy.numel())
            selected[layer_id] = [int(x) for x in torch.topk(energy, k=k).indices.detach().cpu().tolist()]
    return selected


def apply_validated_rank1_edit(
    model: torch.nn.Module,
    tokenizer,
    train_rows: list[dict],
    device: torch.device,
    edit_layers: list[int] | tuple[int, ...] = (29, 28),
    top_columns: int = 64,
    strength: float = 0.90,
    n_pairs: int = 12,
) -> dict:
    """Measure and apply the validated raw-rank-1 residual-direction edit in place."""
    layers = list(edit_layers)
    subspaces, singular_values = build_validated_rank1_subspaces(
        model, tokenizer, train_rows, device, layers, n_pairs=n_pairs
    )
    selected = top_projection_columns(model, subspaces, layers, top_columns)
    info = apply_rank_k_subspace_edit(model, layers, subspaces, top_columns, strength)
    info.update(
        {
            "operator": "validated_raw_rank1_residual_direction",
            "strength": strength,
            "top_columns": top_columns,
            "selected_neurons": selected,
            "top_singular_values": singular_values,
        }
    )
    return info
