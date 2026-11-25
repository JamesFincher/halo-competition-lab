#!/usr/bin/env python3
"""Experiment 16: Safety-Whitened Rank-k Subspace Projection (SmolLM2-135M).

Tests arXiv:2607.02396 (RFM-AGOP multi-dimensional refusal subspaces) + arXiv:2603.27518 +
arXiv:2610.05190 (FORGE null-space repair) on 40 unseen held-out tasks:
  - Compares Unwhitened Rank-1 vs Safety-Nullspace Rank-k (k = 1, 2, 3, 4)
  - Zero RL steps (pure surgical subspace projection) + 1-shot closed-form strength estimate
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402


@torch.no_grad()
def collect_sample_hidden_matrices(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    n_pairs: int = 12,
) -> dict:
    """Collect per-sample [N, D] residual matrices for benign-pass, canonical-false-refusal, and true-unsafe."""
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)
    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_pairs]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][:n_pairs]
    canonical_refusal = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'

    pass_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in benign_rows]
    fail_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + canonical_refusal for r in benign_rows]
    unsafe_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in unsafe_rows]

    def _run(texts: list[str]) -> list[torch.Tensor]:
        mats = [[] for _ in range(n_layers)]
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
            cap = {}
            hooks = []
            for idx, lyr in enumerate(layers):
                def _mk(i: int):
                    def _hk(_m, _inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        cap[i] = h[0, -24:, :].detach().float().mean(dim=0).cpu()
                    return _hk
                hooks.append(lyr.register_forward_hook(_mk(idx)))
            try:
                _ = model(**enc)
            finally:
                for h in hooks:
                    h.remove()
            for idx in range(n_layers):
                mats[idx].append(cap[idx])
        return [torch.stack(m, dim=0) for m in mats]

    h_pass = _run(pass_texts)
    h_fail = _run(fail_texts)
    h_uns = _run(unsafe_texts)
    return {"h_pass": h_pass, "h_fail": h_fail, "h_uns": h_uns}


def apply_rank_k_subspace_edit(
    model: torch.nn.Module,
    edit_layers: list[int],
    subspaces: dict[int, torch.Tensor],
    top_columns: int,
    strength: float,
) -> dict:
    """Project out Rank-k orthonormal subspace U_k [D, k] from top-aligned mlp.down_proj columns."""
    edited = 0
    total = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for li in edit_layers:
            down = model.model.layers[li].mlp.down_proj
            w = down.weight.data.float()
            U = subspaces[li].to(w.device, dtype=torch.float32)  # [D, k] orthonormal
            # Rank columns by projection energy onto U
            proj = torch.matmul(U.T, w)  # [k, inter]
            col_energy = (proj ** 2).sum(dim=0)
            k_cols = min(top_columns, w.shape[1])
            top_idx = torch.topk(col_energy, k=k_cols).indices
            w[:, top_idx] -= strength * torch.matmul(U, proj[:, top_idx])
            down.weight.data.copy_(w.to(down.weight.dtype))
            edited += int(w.shape[0] * k_cols)
    return {"edit_layers": edit_layers, "edited_weights": edited, "edited_pct": round(100.0 * edited / max(total, 1), 5)}


def main() -> None:
    t0 = time.time()
    device = torch.device("cpu")
    out_dir = ROOT / "experiments/whitened_subspace_rfm_edit/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = ben[-5:] + uns[-5:]

    print("[Setup] Loading SmolLM2-135M + Halo SFT checkpoint-36...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft_model = PeftModel.from_pretrained(base_raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload()
    sft_model.eval()

    print("[Subspace] Collecting [12 x D] contrast matrices across all 30 layers...", flush=True)
    mats = collect_sample_hidden_matrices(sft_model, tokenizer, train_rows, device, n_pairs=12)
    edit_layers = [29, 28]

    # Build Safety-Nullspace Rank-k subspaces (k = 1, 2, 3, 4) vs Raw Rank-1
    results = {}
    base_dev = base_rl.evaluate_heldout(sft_model, tokenizer, dev_rows, device)

    for mode, rank_k, null_m in [
        ("raw_rank1", 1, 0),
        ("safety_null_rank1", 1, 1),
        ("safety_null_rank2", 2, 1),
        ("safety_null_rank3", 3, 1),
        ("safety_null_rank4", 4, 2),
    ]:
        subspaces = {}
        sv_info = {}
        for li in edit_layers:
            df = mats["h_fail"][li] - mats["h_pass"][li]  # [N, D]
            ds = mats["h_uns"][li] - mats["h_pass"][li]   # [N, D]
            if null_m > 0:
                _, _, vh_s = torch.linalg.svd(ds, full_matrices=False)
                Q_s = vh_s[:null_m, :].T  # [D, null_m]
                # Project failure deltas into exact orthogonal complement of top-null_m safety directions
                df_clean = df - (df @ Q_s) @ Q_s.T
            else:
                df_clean = df
            _, S_f, vh_f = torch.linalg.svd(df_clean, full_matrices=False)
            U_k = vh_f[:rank_k, :].T  # [D, rank_k] orthonormal
            subspaces[li] = U_k
            sv_info[str(li)] = [round(float(x), 3) for x in S_f[:4].tolist()]

        best_m, best_dev, best_st = sft_model, base_dev, 0.0
        for st in (0.45, 0.70, 0.90):
            cand = copy.deepcopy(sft_model).to(device)
            apply_rank_k_subspace_edit(cand, edit_layers, subspaces, 64, st)
            ev_d = base_rl.evaluate_heldout(cand, tokenizer, dev_rows, device)
            ok = ev_d["unsafe_refusal_rate"] >= base_dev["unsafe_refusal_rate"] - 21.0
            if ok and ev_d["mean_reward"] > best_dev["mean_reward"] + 1e-4:
                best_m, best_dev, best_st = cand, ev_d, st
        ev_40 = base_rl.evaluate_heldout(best_m, tokenizer, eval_rows, device)
        results[mode] = {
            "rank_k": rank_k,
            "safety_null_m": null_m,
            "accepted_strength": best_st,
            "top_singular_values": sv_info,
            "eval": ev_40,
        }
        print(
            f"  [{mode:18s}] st={best_st:.2f} | exact={ev_40['exact_compliance_rate']}% | "
            f"policy={ev_40['policy_accuracy']}% | unsafe={ev_40['unsafe_refusal_rate']}% | "
            f"benign_fr={ev_40['benign_false_refusal_rate']}% | rew={ev_40['mean_reward']:.3f}",
            flush=True,
        )

    summary = {"wall_sec": round(time.time() - t0, 2), "edit_layers": edit_layers, "modes": results}
    (out_dir / "whitened_subspace_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {out_dir / 'whitened_subspace_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
