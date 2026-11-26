#!/usr/bin/env python3
"""Experiment 20: 5-Seed Matched-Budget Random-Node Control Proof.

Compares DLA Top-64 vs 5 independent Random-64 seeds and Anti-DLA Bottom-64
in identical layers [29, 28] using the same continuous rank-1 projection operator.
Logs full per-task predictions and exact neuron indices for post-hoc study.
"""

from __future__ import annotations

import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402


def compute_dla_scores(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_layers = model.config.num_hidden_layers
    inter_dim = model.config.intermediate_size
    tid_ref = tokenizer.encode("ref", add_special_tokens=False)[0]
    tid_call = tokenizer.encode("call", add_special_tokens=False)[0]
    lm_w = model.lm_head.weight.detach().float()
    u = lm_w[tid_ref] - lm_w[tid_call]
    u = u / (u.norm() + 1e-8)

    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:10]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][:10]

    def collect_mean_act(rows: list[dict]) -> list[torch.Tensor]:
        sums = [torch.zeros(inter_dim, device=device, dtype=torch.float32) for _ in range(n_layers)]
        hooks = []
        for li in range(n_layers):
            def _mk(idx: int):
                def _hook(_mod, inp, _out):
                    sums[idx] += inp[0][0, -1, :].detach().float()
                return _hook
            hooks.append(model.model.layers[li].mlp.down_proj.register_forward_hook(_mk(li)))
        with torch.no_grad():
            for r in rows:
                prompt = base_rl.build_prompt(tokenizer, r["messages"])
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                model(**ids)
        for h in hooks:
            h.remove()
        return [s / max(len(rows), 1) for s in sums]

    ben_m = collect_mean_act(benign_rows)
    uns_m = collect_mean_act(unsafe_rows)
    scores = torch.zeros(n_layers, inter_dim, dtype=torch.float32)
    for li in range(n_layers):
        w_down = model.model.layers[li].mlp.down_proj.weight.detach().float()
        col_proj = torch.matmul(u.to(w_down.device), w_down)
        ben_dla = (ben_m[li] * col_proj).clamp(min=0.0)
        uns_dla = (uns_m[li] * col_proj).clamp(min=0.0)
        scores[li] = (ben_dla - 0.35 * uns_dla).cpu()
    return scores, u.cpu()


def apply_neuron_subset_edit(
    model: torch.nn.Module,
    selected: dict[int, list[int]],
    u_vec: torch.Tensor,
    strength: float = 0.70,
) -> None:
    with torch.no_grad():
        for li, idxs in selected.items():
            w = model.model.layers[li].mlp.down_proj.weight
            u = u_vec.to(device=w.device, dtype=w.dtype)
            cols = w[:, idxs]
            proj = torch.outer(u, torch.matmul(u, cols))
            w[:, idxs] = cols - strength * proj


def eval_with_predictions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    eval_rows: list[dict],
    device: torch.device,
) -> tuple[dict, list[dict]]:
    metrics = base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
    preds = []
    model.eval()
    with torch.no_grad():
        for idx, row in enumerate(eval_rows):
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out_ids = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rew, info = base_rl.score_completion(gen, row["target"])
            preds.append({"idx": idx, "target_action": row["target"]["action"], "reward": rew, "exact": info["exact"], "gen": gen[:140]})
    return metrics, preds


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = ROOT / "experiments/random_node_control_proof/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 20] Running 5-Seed Random-Node Control Proof on {device} ({dtype})", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()

    base_ev = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [Baseline SFT] exact={base_ev['exact_compliance_rate']:.1f}% | pol={base_ev['policy_accuracy']:.1f}% | unsafe={base_ev['unsafe_refusal_rate']:.1f}%", flush=True)

    dla_scores, u = compute_dla_scores(sft, tokenizer, train_rows, device)
    inter_dim = sft.config.intermediate_size

    dla_sel = {li: torch.topk(dla_scores[li], k=64).indices.tolist() for li in (29, 28)}
    m_dla = copy.deepcopy(sft)
    apply_neuron_subset_edit(m_dla, dla_sel, u, strength=0.70)
    dla_ev, dla_preds = eval_with_predictions(m_dla, tokenizer, eval_rows, device)
    print(f"  [DLA Top-64  ] exact={dla_ev['exact_compliance_rate']:.1f}% | pol={dla_ev['policy_accuracy']:.1f}% | unsafe={dla_ev['unsafe_refusal_rate']:.1f}%", flush=True)
    del m_dla

    anti_sel = {li: torch.topk(-dla_scores[li], k=64).indices.tolist() for li in (29, 28)}
    m_anti = copy.deepcopy(sft)
    apply_neuron_subset_edit(m_anti, anti_sel, u, strength=0.70)
    anti_ev = base_rl.evaluate_heldout(m_anti, tokenizer, eval_rows, device)
    print(f"  [Anti Bot-64 ] exact={anti_ev['exact_compliance_rate']:.1f}% | pol={anti_ev['policy_accuracy']:.1f}% | unsafe={anti_ev['unsafe_refusal_rate']:.1f}%", flush=True)
    del m_anti

    rand_results = []
    for seed in (101, 202, 303, 404, 505):
        rng = random.Random(seed)
        rsel = {li: rng.sample(range(inter_dim), 64) for li in (29, 28)}
        m = copy.deepcopy(sft)
        apply_neuron_subset_edit(m, rsel, u, strength=0.70)
        ev = base_rl.evaluate_heldout(m, tokenizer, eval_rows, device)
        rand_results.append({"seed": seed, "selected_neurons": rsel, **ev})
        print(f"  [Rand seed={seed}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}%", flush=True)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rand_exacts = [r["exact_compliance_rate"] for r in rand_results]
    rand_pols = [r["policy_accuracy"] for r in rand_results]
    summary = {
        "device": str(device),
        "wall_sec": round(time.time() - t0, 2),
        "baseline": base_ev,
        "dla_top64": {"metrics": dla_ev, "selected_neurons": dla_sel},
        "anti_bottom64": anti_ev,
        "random_5seeds": rand_results,
        "random_exact_mean": round(statistics.mean(rand_exacts), 2),
        "random_exact_stdev": round(statistics.stdev(rand_exacts), 2),
        "random_policy_mean": round(statistics.mean(rand_pols), 2),
        "dla_vs_random_exact_gain": round(dla_ev["exact_compliance_rate"] - statistics.mean(rand_exacts), 2),
    }
    (out_dir / "random_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "dla_predictions.json").write_text(json.dumps(dla_preds, indent=2), encoding="utf-8")
    print(
        f"[Exp 20 DONE] DLA={dla_ev['exact_compliance_rate']:.1f}% vs Rand={summary['random_exact_mean']:.1f}%±{summary['random_exact_stdev']:.1f}% "
        f"(gain=+{summary['dla_vs_random_exact_gain']:.1f} pp) in {summary['wall_sec']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
