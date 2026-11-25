#!/usr/bin/env python3
"""Experiment 9: INSEP Refusal-Direction Linear Separability Probe (arXiv:2611.28314).

Measures how easily a white-box abliteration attack can separate harmful vs benign activations
at each transformer layer via difference-in-means projection and Fisher separability ratio:
  J_l = (mu_harm - mu_benign)^2 / (var_harm + var_benign)
Lower J_l (while keeping behavioral refusal accuracy high) = stronger abliteration defense.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_harm_benign(path: Path, max_per_class: int) -> tuple[list[str], list[str]]:
    harm, benign = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages", [])
        if len(msgs) >= 3:
            usr = msgs[1]["content"]
            asst = msgs[2]["content"]
            if '"action": "refuse"' in asst and len(harm) < max_per_class:
                harm.append(usr)
            elif '"action": "call_tool"' in asst and len(benign) < max_per_class:
                benign.append(usr)
    return harm, benign


@torch.no_grad()
def collect_last_token_hiddens(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    max_length: int,
) -> list[torch.Tensor]:
    per_layer: list[list[torch.Tensor]] | None = None
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        hiddens = out.hidden_states[1:]  # skip embedding
        if per_layer is None:
            per_layer = [[] for _ in range(len(hiddens))]
        for li, h in enumerate(hiddens):
            per_layer[li].append(h[0, -1, :].detach().float().cpu())
    return [torch.stack(xs, dim=0) for xs in (per_layer or [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--max-per-class", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    harm, benign = load_harm_benign(Path(args.data_file), args.max_per_class)
    h_harm = collect_last_token_hiddens(model, tokenizer, harm, device, args.max_length)
    h_ben = collect_last_token_hiddens(model, tokenizer, benign, device, args.max_length)

    layer_stats = []
    for li, (hh, hb) in enumerate(zip(h_harm, h_ben)):
        d = hh.mean(dim=0) - hb.mean(dim=0)
        u = d / (d.norm() + 1e-8)
        proj_h = hh @ u
        proj_b = hb @ u
        fisher_j = float(((proj_h.mean() - proj_b.mean()) ** 2) / (proj_h.var() + proj_b.var() + 1e-6))
        layer_stats.append(
            {
                "layer": li,
                "refusal_dir_l2": round(float(d.norm().item()), 5),
                "fisher_separability": round(fisher_j, 5),
            }
        )

    ranked = sorted(layer_stats, key=lambda x: x["fisher_separability"], reverse=True)
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps({"model": args.model, "top_separable_layers": ranked[:8], "all_layers": layer_stats}, indent=2), encoding="utf-8")
    print(f"[OK] Peak refusal separability at layer {ranked[0]['layer']} (J={ranked[0]['fisher_separability']}) -> {args.out_file}")


if __name__ == "__main__":
    main()
