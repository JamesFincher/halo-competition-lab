#!/usr/bin/env python3
"""Exp153: OLMoE expert-adapter stage zero.

Measurement-only baseline for allenai/OLMoE-1B-7B-0125-Instruct before any expert
adapter training: behavior evals, continuation benchmarks, per-layer expert routing
profiles on behavior and capability continuations, actual memory/wall-clock capture,
and a token-local adapter-cache consistency diagnostic. Produces the evidence needed
to choose benchmark-selective experts for the follow-on adapter experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch


def ensure_large_model_transformers() -> None:
    """The stock halo image predates some pivot architectures; upgrade once and re-exec if needed."""
    try:
        import transformers
        from packaging.version import Version
        ready = Version(transformers.__version__.split("+")[0]) >= Version("4.57.0") and hasattr(transformers, "OlmoeForCausalLM")
    except Exception:
        ready = False
    if not ready and os.environ.get("HALO_TRANSFORMERS_UPGRADED") != "1":
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.57,<5", "peft>=0.14", "accelerate>=1.0", "packaging"])
        env = os.environ.copy()
        env["HALO_TRANSFORMERS_UPGRADED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


ensure_large_model_transformers()
from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import write_json  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark  # noqa: E402

MODEL_NAME = "allenai/OLMoE-1B-7B-0125-Instruct"
EXP_DIR = ROOT / "experiments/exp153_olmoe_expert_stage_zero"
RESULTS_DIR = EXP_DIR / "results"


def smoke() -> bool:
    return os.environ.get("HALO_SMOKE_TEST") == "1"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(tokenizer, device):
    if smoke():
        cfg = OlmoeConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            max_position_embeddings=384,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = OlmoeForCausalLM(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
    return model.to(device)


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def memory_stats(device) -> dict:
    if device.type != "cuda":
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(device),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 2),
    }


@torch.no_grad()
def cache_consistency_diagnostic(model, tokenizer, device) -> dict:
    """Verify that token-local output adapters are cache-safe.

    The temporary hook applies a nonzero pointwise transform to every MoE block
    output. Unlike inserted attention heads, it depends only on the current token's
    tensor, so cached and uncached greedy generation should agree exactly.
    """
    def pointwise_hook(_module, _inputs, output):
        return output + 0.001 * torch.tanh(output)

    hooks = [layer.mlp.register_forward_hook(pointwise_hook) for layer in model.model.layers]
    try:
        prompt = base_rl.build_prompt(tokenizer, [{"role": "system", "content": "You are concise."}, {"role": "user", "content": "Say hello."}])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        ids_cached = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        ids_uncached = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=False)
        logits_cached = model(**enc, use_cache=True).logits
        logits_uncached = model(**enc, use_cache=False).logits
        return {
            "generated_ids_equal": bool(torch.equal(ids_cached.cpu(), ids_uncached.cpu())),
            "single_forward_max_abs_logit_diff": round(float((logits_cached - logits_uncached).abs().max().item()), 8),
            "claim_checked": "token-local expert/output adapters preserve normal cached generation",
        }
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def profile_routing(model, tokenizer, device, capability_rows: list[dict], behavior_rows: list[dict], max_rows: int) -> dict:
    caps = [[] for _ in model.model.layers]
    hooks = []

    def make_hook(idx):
        def hook(_module, _inputs, output):
            _logits, scores, indices = output
            caps[idx].append((indices.detach().cpu(), scores.detach().cpu()))
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(make_hook(i)))
    n_layers = len(model.model.layers)
    n_experts = int(model.config.num_experts)
    datasets = {
        "capability_correct_continuation": [],
        "behavior_assistant_response": [],
    }
    try:
        for kind, rows in [("capability_correct_continuation", capability_rows[:max_rows]), ("behavior_assistant_response", behavior_rows[:max_rows])]:
            counts = torch.zeros((n_layers, n_experts), dtype=torch.float64)
            answer_counts = torch.zeros((n_layers, n_experts), dtype=torch.float64)
            mass = torch.zeros((n_layers, n_experts), dtype=torch.float64)
            token_count = 0
            answer_token_count = 0
            for row in rows:
                if kind.startswith("capability"):
                    context = row["context"].strip() + " "
                    text = context + row["answer_text"].strip()
                else:
                    context = base_rl.build_prompt(tokenizer, row["messages"])
                    text = context + row["assistant_text"]
                prefix_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=384, add_special_tokens=False).to(device)
                before = [len(x) for x in caps]
                _ = model(**enc)
                seq_len = int(enc["input_ids"].shape[1])
                answer_positions = torch.arange(seq_len) >= max(prefix_len - 1, 0)
                token_count += seq_len
                answer_token_count += int(answer_positions.sum().item())
                for layer in range(n_layers):
                    new_items = caps[layer][before[layer] :]
                    if not new_items:
                        continue
                    indices, scores = new_items[-1]
                    indices = indices[:seq_len]
                    scores = scores[:seq_len]
                    flat = indices.reshape(-1)
                    counts[layer] += torch.bincount(flat, minlength=n_experts).double()
                    ans_flat = indices[answer_positions].reshape(-1)
                    answer_counts[layer] += torch.bincount(ans_flat, minlength=n_experts).double()
                    mass[layer] += torch.zeros(n_experts, dtype=torch.float64).index_add_(0, flat, scores.reshape(-1).double())
            top_by_answer = []
            for layer in range(n_layers):
                total = max(float(answer_counts[layer].sum().item()), 1.0)
                vals, idxs = torch.topk(answer_counts[layer], k=min(10, n_experts))
                top_by_answer.append([
                    {"expert": int(ix), "answer_token_share": round(float(v) / total, 5)}
                    for v, ix in zip(vals.tolist(), idxs.tolist())
                ])
            probs = answer_counts / answer_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            datasets[kind] = {
                "rows_profiled": len(rows),
                "tokens_profiled": token_count,
                "answer_tokens_profiled": answer_token_count,
                "top_experts_by_answer_token_count": top_by_answer,
                "top8_answer_token_share_by_layer": [round(float(torch.topk(answer_counts[i], k=min(8, n_experts)).values.sum().item() / max(answer_counts[i].sum().item(), 1.0)), 4) for i in range(n_layers)],
                "answer_routing_entropy_by_layer": [round(float(x), 4) for x in entropy.tolist()],
                "total_selection_counts_by_layer": ((counts * 100).round() / 100).tolist() if smoke() else "omitted_for_size",
                "router_weight_mass_by_layer": ((mass * 10000).round() / 10000).tolist() if smoke() else "omitted_for_size",
            }
    finally:
        for h in hooks:
            h.remove()
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation-n", type=int, default=100)
    parser.add_argument("--router-rows", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[Exp153] OLMoE stage zero on {device}", flush=True)
    tokenizer = load_tokenizer()
    model = load_model(tokenizer, device)
    model.eval()
    original_eval = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    scaled_eval = base_rl.load_rows(ROOT / "data/policy_eval_scaled.jsonl")
    scaled_train = base_rl.load_rows(ROOT / "data/policy_train_scaled.jsonl")
    capability_validation = jsonl_rows(ROOT / "data/capability_validation_scaled.jsonl")
    if smoke():
        capability_validation = capability_validation[:2]
        scaled_train = scaled_train[:2]

    phases = {}
    t = time.time()
    behavior = {
        "original40": base_rl.evaluate_heldout(model, tokenizer, original_eval, device, max_new_tokens=args.max_new_tokens, use_cache=True),
        "scaled120": base_rl.evaluate_heldout(model, tokenizer, scaled_eval, device, max_new_tokens=args.max_new_tokens, use_cache=True),
    }
    phases["behavior_eval_wall_sec"] = round(time.time() - t, 2)
    t = time.time()
    continuation = run_continuation_benchmark(model, tokenizer, device, args.continuation_n)
    phases["continuation_benchmark_wall_sec"] = round(time.time() - t, 2)
    t = time.time()
    router = profile_routing(model, tokenizer, device, capability_validation, scaled_train, args.router_rows)
    phases["router_profile_wall_sec"] = round(time.time() - t, 2)
    cache_check = cache_consistency_diagnostic(model, tokenizer, device)

    summary = {
        "experiment": "exp153_olmoe_expert_stage_zero",
        "model": MODEL_NAME,
        "model_facts": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "layers": len(model.model.layers),
            "experts_per_layer": int(model.config.num_experts),
            "top_k_per_token": int(model.config.num_experts_per_tok),
        },
        "behavior_eval": behavior,
        "continuation_benchmark": continuation,
        "router_profiles": router,
        "cache_consistency": cache_check,
        "phase_wall_sec": phases,
        "memory": memory_stats(device),
        "wall_sec": round(time.time() - t0, 2),
        "smoke_test": smoke(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / "olmoe_stage_zero_summary.json", summary)
    print(f"[DONE] exp153 exact_scaled={behavior['scaled120']['exact_compliance_rate']}% wall={summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
