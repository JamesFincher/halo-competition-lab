#!/usr/bin/env python3
"""Experiment 61: Exact-zero keyed GQA slider lane.

Implements the first local slice of H1: a warm, architecture-matched attention lane
whose output is multiplied by a ReLU key gate. The gate is initialized from measured
benign-vs-unsafe residual-stream boundaries, so it can be exactly zero off-trigger
rather than always-on like Exp 42/59.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from gqa_slider import GqaSliderHead, balanced_rows, count_parameters, train_slider_rl, warm_copy_from_gqa_group  # noqa: E402

INSERT_LAYER = 27
SOURCE_LAYER = 29
KV_GROUP = 0


class KeyedGqaLane(nn.Module):
    def __init__(self, lane: GqaSliderHead, key: torch.Tensor, threshold: float, scale: float):
        super().__init__()
        self.lane = lane
        self.key = nn.Parameter(key.detach().clone())
        self.bias = nn.Parameter(torch.tensor(-float(threshold), dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(max(float(scale), 1e-4), dtype=torch.float32))

    @property
    def alpha(self):
        return self.lane.alpha

    def gate_values(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x, self.key.to(dtype=x.dtype, device=x.device))
        return torch.relu((proj + self.bias.to(dtype=x.dtype, device=x.device)) / self.scale.to(dtype=x.dtype, device=x.device)).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.gate_values(x) * self.lane.attention_delta(x)


def capture_final_hidden(model, tokenizer, rows: list[dict], device: torch.device, layer: int) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured.append(hidden[0, -1, :].detach().float().cpu())

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            for row in rows:
                prompt = base_rl.build_prompt(tokenizer, row["messages"])
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                model(**enc)
    finally:
        handle.remove()
    return torch.stack(captured)


def calibrate_key(model, tokenizer, rows: list[dict], device: torch.device, layer: int) -> tuple[torch.Tensor, float, float, dict]:
    benign = [r for r in rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in rows if r["target"]["action"] == "refuse"]
    hb = capture_final_hidden(model, tokenizer, benign, device, layer)
    hu = capture_final_hidden(model, tokenizer, unsafe, device, layer)
    key = hb.mean(dim=0) - hu.mean(dim=0)
    key = key / (key.norm() + 1e-8)
    pb = hb @ key
    pu = hu @ key
    if float(pb.mean()) < float(pu.mean()):
        key = -key
        pb = hb @ key
        pu = hu @ key
    candidates = torch.linspace(float(min(pb.min(), pu.min())), float(max(pb.max(), pu.max())), steps=64)
    best = None
    for cand in candidates:
        b_pos = (pb > cand).float().mean().item()
        u_pos = (pu > cand).float().mean().item()
        margin = float((pb[pb > cand].mean() - cand).item()) if bool((pb > cand).any()) else 0.0
        score = b_pos - 2.0 * u_pos + 0.01 * margin
        if best is None or score > best[0]:
            best = (score, float(cand), b_pos, u_pos)
    threshold = best[1]
    pooled = torch.cat([pb, pu])
    scale = max(float(pooled.std(unbiased=False).item()), 1e-3)
    stats = {
        "benign_final_token_gate_positive_rate": round(best[2], 4),
        "unsafe_final_token_gate_positive_rate": round(best[3], 4),
        "benign_projection_mean": round(float(pb.mean().item()), 5),
        "unsafe_projection_mean": round(float(pu.mean().item()), 5),
        "threshold": round(threshold, 5),
        "scale": round(scale, 5),
    }
    return key.to(device), threshold, scale, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp61_keyed_zero_gqa_slider/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 61] Exact-zero keyed warm GQA slider on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = balanced_rows(train_rows, n_each=12)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)
    ref_model = copy.deepcopy(sft).to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    baseline = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(
        f"  [baseline] exact={baseline['exact_compliance_rate']:.1f}% unsafe={baseline['unsafe_refusal_rate']:.1f}% benign_fr={baseline['benign_false_refusal_rate']:.1f}%",
        flush=True,
    )

    key, threshold, scale, key_stats = calibrate_key(sft, tokenizer, rows, device, INSERT_LAYER)
    results = {
        "experiment": "exp61_keyed_zero_gqa_slider",
        "research_source": "research/HYPOTHESES.md H1 exact-zero keyed modular implant bank",
        "paper_inspiration": ["arXiv:2611.28483"],
        "config": vars(args),
        "insert_layer": INSERT_LAYER,
        "source_layer": SOURCE_LAYER,
        "baseline": baseline,
        "key_calibration": key_stats,
        "arms": {},
    }

    for arm_name, keyed in [("always_on_warm_gqa", False), ("keyed_zero_warm_gqa", True)]:
        print(f"[Run] {arm_name}", flush=True)
        torch.manual_seed(args.seed)
        lane = GqaSliderHead(hidden_size=sft.config.hidden_size).to(device)
        warm_copy_from_gqa_group(lane, sft, SOURCE_LAYER, KV_GROUP)
        head = KeyedGqaLane(lane, key, threshold, scale).to(device) if keyed else lane
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity_max_diff = float((head(probe) - probe).abs().max().item())
        if identity_max_diff != 0.0:
            raise RuntimeError(f"zero-alpha identity failed: {identity_max_diff}")
        handle, trace = train_slider_rl(
            arm_name,
            sft,
            ref_model,
            tokenizer,
            rows,
            device,
            head,
            INSERT_LAYER,
            steps=args.steps,
            group_size=args.group_size,
            lr=args.lr,
            beta=args.beta,
            bc_lambda=0.0,
            seed=args.seed,
        )
        ev = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
        handle.remove()
        print(
            f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% "
            f"benign_fr={ev['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.5f}",
            flush=True,
        )
        results["arms"][arm_name] = {
            "keyed": keyed,
            "zero_alpha_identity_max_diff": identity_max_diff,
            "parameter_count": count_parameters(head),
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "eval": ev,
            "delta_exact_vs_baseline": round(ev["exact_compliance_rate"] - baseline["exact_compliance_rate"], 2),
            "delta_benign_false_refusal_vs_baseline": round(ev["benign_false_refusal_rate"] - baseline["benign_false_refusal_rate"], 2),
            "trace": trace[::3] + ([trace[-1]] if trace and (len(trace) - 1) % 3 != 0 else []),
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "keyed_zero_gqa_slider_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 61 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
