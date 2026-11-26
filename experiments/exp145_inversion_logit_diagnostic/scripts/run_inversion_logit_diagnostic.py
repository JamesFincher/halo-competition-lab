#!/usr/bin/env python3
"""Experiment 145: L20 inversion logit-degeneracy diagnostic.

This experiment isolates insertion depth only: same width-64 head, prompt stream,
seed, optimizer, training budget, initialization family, and evaluation across layers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "scripts"))
from serialization_utils import dumps_safe, write_json  # noqa: E402
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from methodical_slider import PlainSliderHead, head_parameter_count, ordered_fork_rows, probe_layers_geometry, train_plain_slider  # noqa: E402
from real_benchmark import run_benchmark  # noqa: E402
from continuation_benchmark import continuation_logprob  # noqa: E402

LAYERS = [20, 27]
WIDTH = 64

@torch.no_grad()
def distribution_diagnostic(model, tokenizer, device, n=8):
    from datasets import load_dataset
    rows = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
    entropies=[]; top1s=[]; uniform_kls=[]; sum_margins=[]; token_margins=[]; correct_positive=0
    for r in rows:
        context=r["question"]; choices=list(r["choices"]); correct=choices[int(r["answer"])]
        enc=tokenizer(context,return_tensors="pt",add_special_tokens=False).to(device)
        logits=model(**enc).logits[0,-1].float(); probs=torch.softmax(logits,dim=-1)
        entropies.append(float((-(probs*torch.log(probs.clamp_min(1e-12))).sum()).item()))
        top1s.append(float(probs.max().item()))
        uniform_kls.append(float((probs*(torch.log(probs.clamp_min(1e-12))+torch.log(torch.tensor(float(probs.numel()),device=device)))).sum().item()))
        scored=[(ch,continuation_logprob(model,tokenizer,device,context,ch)) for ch in choices]
        corr=next(lp for ch,lp in scored if ch==correct); wrong=[lp for ch,lp in scored if ch!=correct]
        best=max(wrong,key=lambda x:x['sum']); best_t=max(wrong,key=lambda x:x['token_mean'])
        sum_margins.append(corr['sum']-best['sum']); token_margins.append(corr['token_mean']-best_t['token_mean']); correct_positive+=int(corr['sum']>best['sum'])
    n=max(len(entropies),1)
    return {'n':len(entropies),'mean_next_token_entropy':round(sum(entropies)/n,5),'mean_top1_probability':round(sum(top1s)/n,6),'mean_kl_to_uniform':round(sum(uniform_kls)/n,5),'mean_correct_minus_best_incorrect_sum':round(sum(sum_margins)/n,5),'mean_correct_minus_best_incorrect_token':round(sum(token_margins)/n,5),'correct_sum_positive_rate':round(100.0*correct_positive/n,2)}


def metric_delta(after: dict, before: dict) -> dict:
    return {k: round(after[k] - before[k], 2) for k in after if k in before and isinstance(after[k], (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--bench-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp145_inversion_logit_diagnostic/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 91] Single-slider depth ablation on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    rows = ordered_fork_rows(train_rows, 12)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    for p in sft.parameters():
        p.requires_grad_(False)

    baseline_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, max_new_tokens=args.max_new_tokens, use_cache=False)
    print(f"  [baseline behavior] exact={baseline_eval['exact_compliance_rate']:.1f}% unsafe={baseline_eval['unsafe_refusal_rate']:.1f}% benign_fr={baseline_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    baseline_bench = run_benchmark(sft, tokenizer, device, args.bench_n)
    baseline_diag = distribution_diagnostic(sft, tokenizer, device)
    print(f"  [baseline benchmark] {baseline_bench}", flush=True)
    probe_rows = rows[:4] + rows[-4:]
    try:
        tl_probe = probe_layers_geometry(sft, tokenizer, probe_rows, device, list(LAYERS))
    except Exception as exc:
        tl_probe = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"  [TransformerLens probe failed; continuing] {tl_probe['error']}", flush=True)

    results = {
        "experiment": "exp145_inversion_logit_diagnostic",
        "controlled_variable": "insertion layer only (16/20/24/27/29)",
        "fixed_conditions": {
            "head_dim": WIDTH,
            "head_count": 1,
            "seed": args.seed,
            "optimizer": "AdamW lr=3e-3 default betas",
            "training": "slider_head_fork-style grouped REINFORCE, no KL/BC",
            "prompt_stream": "12 benign then 12 unsafe, fixed order",
        },
        "config": vars(args),
        "baseline_eval": baseline_eval,
        "baseline_benchmark": baseline_bench,
        "baseline_distribution_diagnostic": baseline_diag,
        "transformerlens_probe_by_layer": tl_probe,
        "arms": {},
    }
    for layer in LAYERS:
        arm = f"layer_{layer}"
        print(f"[Run] {arm}: one width-64 head, all other variables fixed", flush=True)
        torch.manual_seed(args.seed)
        head = PlainSliderHead(sft.config.hidden_size, WIDTH).to(device)
        with torch.no_grad():
            probe = torch.randn(1, 5, sft.config.hidden_size, device=device)
            identity_diff = float((head(probe) - probe).abs().max().item())
        if identity_diff != 0.0:
            raise RuntimeError(f"zero-init identity failed for layer {layer}: {identity_diff}")
        handle, trace = train_plain_slider(
            sft, tokenizer, rows, device, head, layer,
            steps=args.steps, group_size=args.group_size, lr=args.lr,
        )
        behavior = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device, max_new_tokens=args.max_new_tokens, use_cache=False)
        bench = run_benchmark(sft, tokenizer, device, args.bench_n)
        diag = distribution_diagnostic(sft, tokenizer, device)
        handle.remove()
        print(f"  [{arm}] exact={behavior['exact_compliance_rate']:.1f}% unsafe={behavior['unsafe_refusal_rate']:.1f}% benign_fr={behavior['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.5f} bench={bench}", flush=True)
        results["arms"][arm] = {
            "insert_layer": layer,
            "head_dim": WIDTH,
            "parameter_count": head_parameter_count(head),
            "zero_init_identity_max_diff": identity_diff,
            "final_alpha": round(float(head.alpha.detach().item()), 6),
            "behavior_eval": behavior,
            "behavior_delta_vs_baseline": metric_delta(behavior, baseline_eval),
            "benchmark": bench,
            "benchmark_delta_vs_baseline": metric_delta(bench, baseline_bench),
            "distribution_diagnostic": diag,
            "diagnostic_delta_vs_baseline": metric_delta(diag, baseline_diag),
            "trace": trace,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "inversion_logit_diagnostic_summary.json"
    write_json(path, results)
    print(f"[Exp 91 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
