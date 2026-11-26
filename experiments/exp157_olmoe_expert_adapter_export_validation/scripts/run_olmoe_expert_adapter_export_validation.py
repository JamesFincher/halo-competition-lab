#!/usr/bin/env python3
"""Exp157: exact export validation for exp154 OLMoE expert adapters.

The expert adapters are mathematically mergeable because each adapter is linear in the
expert intermediate activation: delta(inter) = (inter @ A.T) @ B.T, equivalent to adding
B @ A to that expert's down_proj. This experiment loads exp154's final adapter
checkpoints, merges them into fresh base weights, and verifies the merged/exported form
against the pre-merge adapted model with logits and greedy-generation equality. It then
measures small behavior/continuation canaries from the merged form. No training is run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import json_safe, write_json  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark  # noqa: E402

EXP154_SCRIPT = ROOT / "experiments/exp154_olmoe_benchmark_selective_expert_adapters/scripts/run_olmoe_expert_adapters.py"
spec = importlib.util.spec_from_file_location("exp154_expert_adapters", EXP154_SCRIPT)
exp154 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp154)

EXP_DIR = ROOT / "experiments/exp157_olmoe_expert_adapter_export_validation"
RESULTS_DIR = EXP_DIR / "results"
ARM_JSONL = RESULTS_DIR / "arm_results.jsonl"
ARMS = [
    "benchmark_selective_rank8",
    "random_expert_parameter_matched",
    "diffuse_all_experts_rank1_parameter_matched",
]


def smoke() -> bool:
    return os.environ.get("HALO_SMOKE_TEST") == "1"


def append_arm(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with ARM_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def completed_arms() -> set[str]:
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


def infer_bank_geometry(state: dict) -> tuple[int, int, int, int, int]:
    a = state["bank"]["a"]
    b = state["bank"]["b"]
    n_layers, n_experts, rank, intermediate = tuple(int(x) for x in a.shape)
    hidden = int(b.shape[-2])
    return n_layers, n_experts, hidden, intermediate, rank


def infer_selections(state: dict, n_layers: int) -> dict[int, list[int]]:
    out = {}
    for layer in range(n_layers):
        keep = state["bank"].get(f"keep_{layer}")
        if keep is None:
            raise KeyError(f"missing keep_{layer} in adapter checkpoint")
        out[layer] = [int(x) for x in torch.nonzero(keep.reshape(-1) > 0).reshape(-1).tolist()]
    return out


@torch.no_grad()
def probe_logits(model, tokenizer, rows: list[dict], device: torch.device, max_rows: int, max_len: int) -> torch.Tensor:
    texts = []
    for row in rows[:max_rows]:
        texts.append((row["context"].strip() + " " + row["answer_text"].strip())[:1200])
    encoded = [tokenizer(t, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"] for t in texts]
    width = max(len(x) for x in encoded)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.tensor([x + [pad_id] * (width - len(x)) for x in encoded], dtype=torch.long, device=device)
    attention = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in encoded], dtype=torch.long, device=device)
    return model(input_ids=input_ids, attention_mask=attention).logits.detach().float().cpu()


@torch.no_grad()
def greedy_ids_for_rows(model, tokenizer, rows: list[dict], device: torch.device, max_rows: int, max_new_tokens: int) -> list[list[int]]:
    if smoke():
        max_new_tokens = min(max_new_tokens, 8)
    out = []
    for row in rows[:max_rows]:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        out.append([int(x) for x in ids[0].detach().cpu().tolist()])
    return out


@torch.no_grad()
def merge_bank_into_experts(model, bank) -> dict:
    """Merge exact linear adapter deltas into expert down_proj and neutralize the external bank."""
    merged_entries = 0
    max_delta_norm = 0.0
    scale = float(bank.alpha) / max(int(bank.rank), 1)
    with torch.no_grad():
        for layer in range(bank.n_layers):
            down = model.model.layers[layer].mlp.experts.down_proj
            for expert in bank.selections[layer]:
                delta = torch.matmul(bank.b[layer, expert], bank.a[layer, expert]) * scale
                down[expert].add_(delta.to(device=down.device, dtype=down.dtype))
                merged_entries += int(delta.numel())
                max_delta_norm = max(max_delta_norm, float(delta.float().norm().item()))
        bank.a.zero_()
        bank.b.zero_()
    return {"merged_weight_entries": merged_entries, "max_delta_frobenius_norm": round(max_delta_norm, 6), "merge_formula": "down_proj[expert] += B @ A * alpha/rank; external bank then zeroed"}


def make_smoke_checkpoints(root: Path, tokenizer, device: torch.device) -> None:
    """Create structurally real tiny adapter checkpoints so full end-to-end smoke exercises loading/merging."""
    if root.exists():
        shutil.rmtree(root)
    model = exp154.load_model(tokenizer, torch.device("cpu"))
    n_layers = len(model.model.layers)
    n_exp = int(model.config.num_experts)
    hidden = int(model.config.hidden_size)
    intermediate = int(model.config.intermediate_size)
    for arm in ARMS:
        rank = 1 if arm.startswith("diffuse") else 2
        if arm.startswith("diffuse"):
            selections = {li: list(range(n_exp)) for li in range(n_layers)}
        elif arm.startswith("random"):
            rng = random.Random(43)
            selections = {li: sorted(rng.sample(range(n_exp), min(2, n_exp))) for li in range(n_layers)}
        else:
            selections = {li: list(range(min(2, n_exp))) for li in range(n_layers)}
        bank = exp154.ExpertAdapterBank(n_layers, n_exp, hidden, intermediate, rank, selections, 16.0, torch.device("cpu"), torch.float32)
        with torch.no_grad():
            bank.b.normal_(mean=0.0, std=0.01)
        path = root / arm / "final_adapter.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": 1, "bank": bank.state_dict()}, path)
    del model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-alpha", type=float, default=16.0)
    ap.add_argument("--probe-rows", type=int, default=4)
    ap.add_argument("--generation-rows", type=int, default=20)
    ap.add_argument("--continuation-n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp157] OLMoE expert-adapter export validation on {device}; transformers={transformers.__version__} torch={torch.__version__}", flush=True)
    tokenizer = exp154.load_tokenizer()
    ckpt_base = ROOT / "experiments/exp154_olmoe_benchmark_selective_expert_adapters" / ("checkpoints_smoke" if smoke() else "checkpoints")
    if smoke():
        make_smoke_checkpoints(ckpt_base, tokenizer, device)
    cap_val = [json.loads(line) for line in (ROOT / "data/capability_validation_scaled.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    behavior_eval = base_rl.load_rows(ROOT / "data/policy_eval_scaled.jsonl")
    if smoke():
        cap_val = cap_val[:2]
        behavior_eval = behavior_eval[:2]

    done = completed_arms()
    missing = []
    for arm in ARMS:
        if arm in done:
            print(f"[Exp157] skipping completed arm {arm}", flush=True)
            continue
        arm_t0 = time.time()
        ckpt = ckpt_base / arm / "final_adapter.pt"
        if not ckpt.exists():
            missing.append(arm)
            print(f"[Exp157] missing dependency checkpoint for {arm}: {ckpt}", flush=True)
            continue
        print(f"[Exp157] arm {arm}", flush=True)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        n_layers, n_experts, hidden, intermediate, rank = infer_bank_geometry(state)
        selections = infer_selections(state, n_layers)
        model = exp154.load_model(tokenizer, device)
        model.eval()
        dtype = next(model.parameters()).dtype
        bank = exp154.ExpertAdapterBank(n_layers, n_experts, hidden, intermediate, rank, selections, args.adapter_alpha, device, dtype).to(device)
        bank.load_state_dict(state["bank"])
        exp154.install_adapters(model, bank)

        before_logits = probe_logits(model, tokenizer, cap_val, device, args.probe_rows if not smoke() else 1, args.max_len)
        before_ids = greedy_ids_for_rows(model, tokenizer, behavior_eval, device, args.generation_rows if not smoke() else 2, args.max_new_tokens)
        merge_info = merge_bank_into_experts(model, bank)
        after_logits = probe_logits(model, tokenizer, cap_val, device, args.probe_rows if not smoke() else 1, args.max_len)
        after_ids = greedy_ids_for_rows(model, tokenizer, behavior_eval, device, args.generation_rows if not smoke() else 2, args.max_new_tokens)
        max_diff = float((before_logits - after_logits).abs().max().item())
        mean_diff = float((before_logits - after_logits).abs().mean().item())
        ids_equal = before_ids == after_ids
        n_equal = sum(int(a == b) for a, b in zip(before_ids, after_ids))

        merged_behavior_rows = behavior_eval[: args.generation_rows if not smoke() else 2]
        merged_behavior = base_rl.evaluate_heldout(model, tokenizer, merged_behavior_rows, device, max_new_tokens=args.max_new_tokens, use_cache=True)
        merged_continuation = run_continuation_benchmark(model, tokenizer, device, args.continuation_n)
        record = {
            "arm": arm,
            "dependency_checkpoint": str(ckpt.relative_to(ROOT)),
            "checkpoint_step": int(state.get("step", 0)),
            "rank": rank,
            "selection_count": sum(len(v) for v in selections.values()),
            "merge_info": merge_info,
            "export_exactness": {
                "max_abs_logit_diff": round(max_diff, 8),
                "mean_abs_logit_diff": round(mean_diff, 10),
                "greedy_generation_ids_equal": bool(ids_equal),
                "greedy_generation_rows_equal": f"{n_equal}/{len(before_ids)}",
                "interpretation": "near-zero differences are floating-point reordering; exact ids on greedy canaries indicate no behavioral export gap",
            },
            "merged_behavior_canary": merged_behavior,
            "merged_continuation_canary": merged_continuation,
            "memory": exp154_memory_stats(device),
            "wall_sec": round(time.time() - arm_t0, 2),
            "smoke_test": smoke(),
        }
        append_arm(record)
        print(f"  {arm}: max_logit_diff={max_diff:.6f} ids_equal={ids_equal} exact_canary={merged_behavior['exact_compliance_rate']}%", flush=True)
        del model, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arms = [json.loads(line) for line in ARM_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()] if ARM_JSONL.exists() else []
    summary = {
        "experiment": "exp157_olmoe_expert_adapter_export_validation",
        "model": exp154.MODEL_NAME,
        "question": "Can exp154 expert adapters be exported exactly into merged expert down_proj weights without a behavior/generation gap?",
        "dependency": "exp154 final_adapter.pt checkpoints in the shared VM checkout; this experiment performs no training",
        "missing_dependency_arms": missing,
        "arms": arms,
        "wall_sec": round(time.time() - t0, 2),
        "smoke_test": smoke(),
    }
    write_json(RESULTS_DIR / "olmoe_expert_adapter_export_summary.json", summary)
    if smoke():
        shutil.rmtree(ckpt_base, ignore_errors=True)
    print(f"[DONE] exp157 arms={len(arms)} missing={len(missing)} wall={summary['wall_sec']}s", flush=True)


def exp154_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(device),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 2),
    }


if __name__ == "__main__":
    main()
