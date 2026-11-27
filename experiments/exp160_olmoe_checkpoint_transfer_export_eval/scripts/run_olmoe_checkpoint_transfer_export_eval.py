#!/usr/bin/env python3
"""Exp160: transfer generalization and practical export fidelity for exp154 checkpoints.

No training. For each completed exp154 arm, evaluate the adapted model before and after
merging adapters into expert down_proj weights. Exact servable equivalence was falsified
by exp157; this experiment measures the operationally relevant benchmark-level fidelity
and whether the original gains transfer to benchmarks whose train splits were not used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT))
from experiments.common.serialization_utils import json_safe, write_json  # noqa: E402
from experiments.common.behavior_stats import compare_behavior_rows  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark, run_transfer_benchmark  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp154 = load_module("exp154mod", ROOT / "experiments/exp154_olmoe_benchmark_selective_expert_adapters/scripts/run_olmoe_expert_adapters.py")
exp157 = load_module("exp157mod", ROOT / "experiments/exp157_olmoe_expert_adapter_export_validation/scripts/run_olmoe_expert_adapter_export_validation.py")
EXP_DIR = ROOT / "experiments/exp160_olmoe_checkpoint_transfer_export_eval"
RESULTS_DIR = EXP_DIR / "results"
ARM_JSONL = RESULTS_DIR / "arm_results.jsonl"
ARMS = exp157.ARMS


def smoke() -> bool:
    return os.environ.get("HALO_SMOKE_TEST") == "1"


def load_model(tokenizer, device):
    if smoke():
        return exp154.load_model(tokenizer, device)
    return AutoModelForCausalLM.from_pretrained(
        exp154.MODEL_NAME,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)


def append_arm(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with ARM_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def finished_arms() -> set[str]:
    if not ARM_JSONL.exists():
        return set()
    out = set()
    for line in ARM_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["arm"])
            except Exception:
                pass
    return out


def evaluate_suite(model, tok, device, continuation_n: int) -> dict:
    return {
        "continuation_benchmark": run_continuation_benchmark(model, tok, device, continuation_n, include_rows=True),
        "transfer_benchmark": run_transfer_benchmark(model, tok, device, continuation_n, include_rows=True),
    }


def dataset_scores(suite: dict) -> dict[str, float]:
    out = {}
    for group in ("continuation_benchmark", "transfer_benchmark"):
        for dataset, metrics in suite.get(group, {}).items():
            if isinstance(metrics, dict) and "token_norm_acc" in metrics:
                out[f"{group}/{dataset}"] = float(metrics["token_norm_acc"])
    return out


def paired_export_statistics(pre_suite: dict, post_suite: dict) -> dict:
    """Paired row-level evidence for whether export changes benchmark outcomes."""
    out = {}
    for suite_key in ("continuation_benchmark", "transfer_benchmark"):
        suite_out = {}
        pre_group = pre_suite.get(suite_key, {})
        post_group = post_suite.get(suite_key, {})
        for dataset in sorted(set(pre_group) & set(post_group)):
            pre_rows = pre_group.get(dataset, {}).get("rows") if isinstance(pre_group.get(dataset), dict) else None
            post_rows = post_group.get(dataset, {}).get("rows") if isinstance(post_group.get(dataset), dict) else None
            if not pre_rows or not post_rows:
                continue
            try:
                suite_out[dataset] = compare_behavior_rows(pre_rows, post_rows, metric="correct_sum")
                suite_out[dataset]["direction"] = "pre_merge_minus_post_merge"
            except ValueError as exc:
                suite_out[dataset] = {"error": str(exc)}
        if suite_out:
            out[suite_key] = suite_out
    return out


def compare_scores(pre: dict, post: dict, baseline: dict | None = None) -> dict:
    pre_s, post_s = dataset_scores(pre), dataset_scores(post)
    rows = []
    for key in sorted(set(pre_s) & set(post_s)):
        row = {
            "dataset": key,
            "pre_token_norm_acc": round(pre_s[key], 2),
            "post_token_norm_acc": round(post_s[key], 2),
            "post_minus_pre_pp": round(post_s[key] - pre_s[key], 2),
        }
        if baseline and key in baseline:
            row["pre_minus_baseline_pp"] = round(pre_s[key] - baseline[key], 2)
            row["post_minus_baseline_pp"] = round(post_s[key] - baseline[key], 2)
        rows.append(row)
    transfer_rows = [r for r in rows if r["dataset"].startswith("transfer_benchmark/")]
    return {
        "per_dataset": rows,
        "max_abs_post_minus_pre_pp": round(max((abs(r["post_minus_pre_pp"]) for r in rows), default=0.0), 2),
        "mean_abs_post_minus_pre_pp": round(sum(abs(r["post_minus_pre_pp"]) for r in rows) / max(len(rows), 1), 2),
        "transfer_pre_mean_acc": round(sum(r["pre_token_norm_acc"] for r in transfer_rows) / max(len(transfer_rows), 1), 2),
        "transfer_post_mean_acc": round(sum(r["post_token_norm_acc"] for r in transfer_rows) / max(len(transfer_rows), 1), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuation-n", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    continuation_n = 1 if smoke() else args.continuation_n
    print(f"[Exp160] checkpoint transfer/export evaluation on {device}; transformers={transformers.__version__} torch={torch.__version__}", flush=True)
    tok = exp154.load_tokenizer()
    ckpt_base = ROOT / "experiments/exp154_olmoe_benchmark_selective_expert_adapters" / ("checkpoints_smoke" if smoke() else "checkpoints")
    if smoke():
        exp157.make_smoke_checkpoints(ckpt_base, tok, torch.device("cpu"))

    baseline_t0 = time.time()
    base_model = load_model(tok, device)
    base_model.eval()
    baseline_suite = evaluate_suite(base_model, tok, device, continuation_n)
    baseline_scores = dataset_scores(baseline_suite)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[Exp160] baseline suite done in {round(time.time()-baseline_t0,1)}s", flush=True)

    done = finished_arms()
    missing = []
    for arm in ARMS:
        if arm in done:
            continue
        ckpt = ckpt_base / arm / "final_adapter.pt"
        if not ckpt.exists():
            missing.append(arm)
            continue
        arm_t0 = time.time()
        print(f"[Exp160] arm {arm}", flush=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        n_layers, n_experts, hidden, intermediate, rank = exp157.infer_bank_geometry(state)
        selections = exp157.infer_selections(state, n_layers)
        model = load_model(tok, device)
        model.eval()
        dtype = next(model.parameters()).dtype
        bank = exp154.ExpertAdapterBank(n_layers, n_experts, hidden, intermediate, rank, selections, 16.0, device, dtype).to(device)
        bank.load_state_dict(state["bank"])
        exp154.install_adapters(model, bank)
        pre_suite = evaluate_suite(model, tok, device, continuation_n)
        merge_info = exp157.merge_bank_into_experts(model, bank)
        post_suite = evaluate_suite(model, tok, device, continuation_n)
        comparison = compare_scores(pre_suite, post_suite, baseline_scores)
        paired_export_stats = paired_export_statistics(pre_suite, post_suite)
        rec = {
            "arm": arm,
            "checkpoint_step": int(state.get("step", 0)),
            "merge_info": merge_info,
            "baseline_scores": baseline_scores,
            "pre_merge_scores": dataset_scores(pre_suite),
            "post_merge_scores": dataset_scores(post_suite),
            "comparison": comparison,
            "paired_export_statistics": paired_export_stats,
            "pre_merge_suite": pre_suite,
            "post_merge_suite": post_suite,
            "wall_sec": round(time.time() - arm_t0, 2),
            "smoke_test": smoke(),
        }
        append_arm(rec)
        print(
            f"  {arm}: transfer pre/post={comparison['transfer_pre_mean_acc']}/{comparison['transfer_post_mean_acc']} "
            f"max export score movement={comparison['max_abs_post_minus_pre_pp']}pp",
            flush=True,
        )
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arms = [json.loads(x) for x in ARM_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()] if ARM_JSONL.exists() else []
    write_json(
        RESULTS_DIR / "olmoe_checkpoint_transfer_export_summary.json",
        {
            "experiment": "exp160_olmoe_checkpoint_transfer_export_eval",
            "model": exp154.MODEL_NAME,
            "question": "Do exp154 capability changes generalize to held-out transfer benchmarks, and are they preserved by expert-weight export at benchmark level?",
            "baseline_suite": baseline_suite,
            "missing_dependency_arms": missing,
            "arms": arms,
            "wall_sec": round(time.time() - t0, 2),
            "smoke_test": smoke(),
        },
    )
    if smoke():
        shutil.rmtree(ckpt_base, ignore_errors=True)
    print(f"[DONE] exp160 arms={len(arms)} missing={len(missing)} wall={round(time.time()-t0,2)}s", flush=True)


if __name__ == "__main__":
    main()
