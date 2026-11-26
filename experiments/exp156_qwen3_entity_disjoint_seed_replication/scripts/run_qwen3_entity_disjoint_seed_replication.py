#!/usr/bin/env python3
"""Exp156: Qwen3-4B entity-disjoint seed replication for the critical plain-LoRA reference.

Uses behavior corpus v2, whose train and heldout rows have zero shared context entities
and zero shared user-prompt 8-grams. The experiment asks whether the scaled-data LoRA
result is robust across seeds and decomposes exact-compliance failures into JSON,
action, policy, and tool-field errors plus per-tool/rare-tool metrics. Arm results
append to JSONL immediately; the queue-completing summary JSON is written only after
all arms finish.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch


def ensure_large_model_transformers() -> None:
    """The stock halo image predates some pivot architectures; upgrade once and re-exec if needed."""
    try:
        import transformers
        from packaging.version import Version
        ready = Version(transformers.__version__.split("+")[0]) >= Version("4.57.0") and hasattr(transformers, "Qwen3ForCausalLM")
    except Exception:
        ready = False
    if not ready and os.environ.get("HALO_TRANSFORMERS_UPGRADED") != "1":
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.57,<5", "peft>=0.14", "accelerate>=1.0", "packaging"])
        env = os.environ.copy()
        env["HALO_TRANSFORMERS_UPGRADED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


ensure_large_model_transformers()
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT))
import run_micro_targeted_rl as base_rl  # noqa: E402
from experiments.common.serialization_utils import json_safe, write_json
from experiments.common.behavior_stats import benchmark_comparisons_against_reference, comparisons_against_reference  # noqa: E402
from experiments.common.validated_edit import apply_validated_rank1_edit  # noqa: E402
from scripts.continuation_benchmark import run_continuation_benchmark, run_transfer_benchmark  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
EXP_DIR = ROOT / "experiments/exp156_qwen3_entity_disjoint_seed_replication"
RESULTS_DIR = EXP_DIR / "results"
ARM_JSONL = RESULTS_DIR / "arm_results.jsonl"
ATTN_MLP_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]


def smoke() -> bool:
    return os.environ.get("HALO_SMOKE_TEST") == "1"


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, tokenizer, device: torch.device):
    if smoke():
        cfg = Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=384,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = Qwen3ForCausalLM(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
    return model.to(device)


def architecture_mapped_layers(model) -> tuple[list[int], list[int]]:
    n = len(model.model.layers)
    edit = [n - 1, n - 2]
    complement = list(range(max(0, n - 6), n - 2))
    if not complement:
        complement = [0]
    return edit, complement


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def behavior_example(tokenizer, row: dict, max_len: int) -> dict:
    prompt = base_rl.build_prompt(tokenizer, row["messages"])
    answer = row["assistant_text"] + (tokenizer.eos_token or "")
    pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    aids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (pids + aids)[:max_len]
    labels = ([-100] * len(pids) + aids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def capability_example(tokenizer, row: dict, max_len: int) -> dict:
    context = row["context"].strip()
    answer = row["answer_text"].strip()
    cids = tokenizer(context + " ", add_special_tokens=False)["input_ids"]
    aids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (cids + aids)[:max_len]
    labels = ([-100] * len(cids) + aids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def collate(examples: list[dict], pad_id: int, device: torch.device) -> dict:
    width = max(len(x["input_ids"]) for x in examples)
    input_ids, labels, attention = [], [], []
    for ex in examples:
        pad = width - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad)
        labels.append(ex["labels"] + [-100] * pad)
        attention.append([1] * len(ex["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def cycled_batch(rows: list[dict], step: int, batch_size: int) -> list[dict]:
    return [rows[(step * batch_size + i) % len(rows)] for i in range(batch_size)]


def make_lora(model, targets: list[str], layers: list[int] | None, adapter_path: Path | None, rank: int, alpha: int, dropout: float):
    if adapter_path is not None and (adapter_path / "adapter_config.json").exists():
        return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        layers_to_transform=layers,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def save_checkpoint(model, optimizer, step: int, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tmp = path / "trainer_state.pt.tmp"
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, tmp)
    tmp.replace(path / "trainer_state.pt")


def latest_checkpoint(path: Path) -> tuple[Path | None, int]:
    candidates = sorted(path.glob("checkpoint-step-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]) if p.name.rsplit("-", 1)[1].isdigit() else -1)
    if not candidates:
        return None, 0
    ckpt = candidates[-1]
    state_path = ckpt / "trainer_state.pt"
    step = int(torch.load(state_path, map_location="cpu", weights_only=False)["step"]) if state_path.exists() else int(ckpt.name.rsplit("-", 1)[1])
    return ckpt, step


def train_lora(
    model,
    tokenizer,
    behavior_rows: list[dict],
    capability_rows: list[dict],
    device: torch.device,
    checkpoint_root: Path,
    steps: int,
    batch_size: int,
    max_len: int,
    lr: float,
    warmup: int,
    checkpoint_interval: int,
    mix_replay: bool,
) -> dict:
    final_adapter = checkpoint_root / "final_adapter"
    ckpt, resume_step = latest_checkpoint(checkpoint_root)
    if final_adapter.exists():
        start_step = steps
    else:
        start_step = resume_step if ckpt is not None else 0
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if ckpt is not None and not final_adapter.exists() and (ckpt / "trainer_state.pt").exists():
        try:
            state = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            for state_values in optimizer.state.values():
                for key, value in list(state_values.items()):
                    if torch.is_tensor(value):
                        state_values[key] = value.to(device)
        except Exception as exc:
            print(f"  optimizer resume skipped: {type(exc).__name__}: {exc}", flush=True)
    if device.type == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    log = []
    train_t0 = time.time()
    for step in range(start_step, steps):
        use_capability = mix_replay and capability_rows and step % 4 == 3
        source_rows = capability_rows if use_capability else behavior_rows
        raw_batch = cycled_batch(source_rows, step, batch_size)
        if use_capability:
            examples = [capability_example(tokenizer, r, max_len) for r in raw_batch]
        else:
            examples = [behavior_example(tokenizer, r, max_len) for r in raw_batch]
        batch = collate(examples, int(tokenizer.pad_token_id), device)
        progress = min(1.0, (step + 1) / max(steps, 1))
        lr_now = lr * min(1.0, (step + 1) / max(warmup, 1)) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            rec = {"step": step + 1, "loss": round(float(loss.detach().item()), 5), "lr": lr_now, "grad_norm": round(grad_norm, 4), "batch_type": "capability_replay" if use_capability else "behavior"}
            log.append(rec)
            print(f"    step {step + 1}/{steps}: loss={rec['loss']:.3f} type={rec['batch_type']} grad={rec['grad_norm']:.2f}", flush=True)
        if (step + 1) % checkpoint_interval == 0 and step + 1 < steps:
            save_checkpoint(model, optimizer, step + 1, checkpoint_root / f"checkpoint-step-{step + 1}")
    if steps > 0:
        save_checkpoint(model, optimizer, steps, final_adapter)
    model.config.use_cache = True
    model.eval()
    return {
        "requested_steps": steps,
        "resumed_from_step": start_step,
        "optimizer_steps_completed": steps,
        "train_wall_sec": round(time.time() - train_t0, 2),
        "train_log_tail": log[-8:],
        "trainable_params": sum(p.numel() for p in params),
        "replay_mixture": "75% behavior / 25% direct-continuation capability by step" if mix_replay else "behavior only",
    }


def wilson95(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5 / denom
    return [round(100.0 * max(0.0, center - margin), 2), round(100.0 * min(1.0, center + margin), 2)]


@torch.no_grad()
def evaluate_behavior_detailed(model, tokenizer, eval_rows: list[dict], device: torch.device, max_new_tokens: int, batch_size: int = 4) -> dict:
    """Aggregate behavior metrics plus hierarchical error and per-tool decomposition."""
    if smoke():
        eval_rows = eval_rows[:2]
        max_new_tokens = min(max_new_tokens, 8)
    model.eval()
    n = len(eval_rows)
    json_valid = action_ok_count = policy_ok_count = tool_ok_count = exact_count = 0
    benign_total = benign_false_refusal = unsafe_total = unsafe_refusal = 0
    total_reward = 0.0
    hierarchical = {"json_invalid": 0, "action_error": 0, "policy_error": 0, "tool_error": 0, "exact": 0}
    per_tool = {}
    action_confusion = {}
    row_results = []

    prompts=[base_rl.build_prompt(tokenizer, row["messages"]) for row in eval_rows]
    generations=[]
    old_padding_side=tokenizer.padding_side
    tokenizer.padding_side="left"
    pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    try:
        for start in range(0,len(prompts),batch_size):
            chunk=prompts[start:start+batch_size]
            enc=tokenizer(chunk,return_tensors="pt",padding=True).to(device)
            out_ids=model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                use_cache=True,
            )
            prompt_width=int(enc["input_ids"].shape[1])
            for ids in out_ids:
                generations.append(tokenizer.decode(ids[prompt_width:],skip_special_tokens=True))
    finally:
        tokenizer.padding_side=old_padding_side

    for row_index,(row,gen) in enumerate(zip(eval_rows,generations)):
        reward, info = base_rl.score_completion(gen, row["target"])
        obj = base_rl.parse_json_object(gen)
        target = row["target"]
        pred_action = obj.get("action") if obj else None
        pred_tool = obj.get("tool") if obj else None
        target_tool = target.get("tool") or "null"
        tool_stats = per_tool.setdefault(target_tool, {"n": 0, "tool_correct": 0, "exact": 0, "benign_total": 0, "benign_false_refusal": 0, "unsafe_total": 0, "unsafe_refusal": 0})
        tool_stats["n"] += 1
        total_reward += reward
        json_valid += int(info["json_valid"])
        action_ok_count += int(info["action_ok"])
        policy_ok_count += int(info["policy_ok"])
        tool_ok = bool(obj is not None and pred_tool == target.get("tool"))
        tool_ok_count += int(tool_ok)
        exact_count += int(info["exact"])
        tool_stats["tool_correct"] += int(tool_ok)
        tool_stats["exact"] += int(info["exact"])
        key = f"{target.get('action')}->{pred_action if pred_action is not None else 'invalid_json'}"
        action_confusion[key] = action_confusion.get(key, 0) + 1

        if target["action"] == "call_tool":
            benign_total += 1
            tool_stats["benign_total"] += 1
            if pred_action == "refuse":
                benign_false_refusal += 1
                tool_stats["benign_false_refusal"] += 1
        else:
            unsafe_total += 1
            tool_stats["unsafe_total"] += 1
            if pred_action == "refuse":
                unsafe_refusal += 1
                tool_stats["unsafe_refusal"] += 1

        if obj is None:
            hierarchical["json_invalid"] += 1
        elif not info["action_ok"]:
            hierarchical["action_error"] += 1
        elif not info["policy_ok"]:
            hierarchical["policy_error"] += 1
        elif not tool_ok:
            hierarchical["tool_error"] += 1
        else:
            hierarchical["exact"] += 1

        row_results.append({
            "index": row_index,
            "exact": bool(info["exact"]),
            "json_valid": bool(info["json_valid"]),
            "action_ok": bool(info["action_ok"]),
            "policy_ok": bool(info["policy_ok"]),
            "tool_ok": bool(tool_ok),
            "reward": round(float(reward), 4),
            "target_action": target.get("action"),
            "pred_action": pred_action,
            "target_tool": target.get("tool"),
            "pred_tool": pred_tool,
            "target_policy_code": target.get("policy_code"),
            "pred_policy_code": obj.get("policy_code") if obj else None,
            "generation_prefix": gen[:240],
        })

    per_tool_out = {}
    for tool, st in sorted(per_tool.items()):
        per_tool_out[tool] = {
            "n": st["n"],
            "tool_accuracy": round(100.0 * st["tool_correct"] / max(st["n"], 1), 1),
            "exact_compliance_rate": round(100.0 * st["exact"] / max(st["n"], 1), 1),
            "benign_false_refusal_rate": round(100.0 * st["benign_false_refusal"] / max(st["benign_total"], 1), 1),
            "unsafe_refusal_rate": round(100.0 * st["unsafe_refusal"] / max(st["unsafe_total"], 1), 1),
        }
    return {
        "n_eval": n,
        "mean_reward": round(total_reward / max(n, 1), 4),
        "json_valid_rate": round(100.0 * json_valid / max(n, 1), 1),
        "action_accuracy": round(100.0 * action_ok_count / max(n, 1), 1),
        "policy_accuracy": round(100.0 * policy_ok_count / max(n, 1), 1),
        "tool_accuracy": round(100.0 * tool_ok_count / max(n, 1), 1),
        "unsafe_refusal_rate": round(100.0 * unsafe_refusal / max(unsafe_total, 1), 1),
        "benign_false_refusal_rate": round(100.0 * benign_false_refusal / max(benign_total, 1), 1),
        "exact_compliance_rate": round(100.0 * exact_count / max(n, 1), 1),
        "exact_compliance_wilson95": wilson95(exact_count, n),
        "unsafe_refusal_wilson95": wilson95(unsafe_refusal, unsafe_total),
        "benign_false_refusal_wilson95": wilson95(benign_false_refusal, benign_total),
        "rows": row_results,
        "hierarchical_error_counts": hierarchical,
        "hierarchical_error_rates_pct": {k: round(100.0 * v / max(n, 1), 1) for k, v in hierarchical.items()},
        "per_tool": per_tool_out,
        "rare_tool_send_http": per_tool_out.get("send_http"),
        "action_confusion_counts": action_confusion,
    }


@torch.no_grad()
def endpoint(model, tokenizer, original_eval: list[dict], v2_eval: list[dict], device: torch.device, continuation_n: int, max_new_tokens: int) -> dict:
    return {
        "behavior_original40_bridge": base_rl.evaluate_heldout(model, tokenizer, original_eval, device, max_new_tokens=max_new_tokens, use_cache=True,batch_size=4),
        "behavior_scaled120": evaluate_behavior_detailed(model, tokenizer, v2_eval, device, max_new_tokens=max_new_tokens),
        "continuation_benchmark": run_continuation_benchmark(model, tokenizer, device, continuation_n, include_rows=True),
        "transfer_benchmark": run_transfer_benchmark(model, tokenizer, device, continuation_n, include_rows=True),
    }


def memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": "cpu"}
    return {
        "device": torch.cuda.get_device_name(device),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 2),
    }


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


def append_arm_result(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with ARM_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--continuation-n", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp156] Qwen3 entity-disjoint seed replication on {device}", flush=True)
    tokenizer = load_tokenizer(MODEL_NAME)
    original_train = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    original_eval = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    v2_train = base_rl.load_rows(ROOT / "data/policy_train_scaled_v2.jsonl")
    v2_eval = base_rl.load_rows(ROOT / "data/policy_eval_scaled_v2.jsonl")
    capability_rows = jsonl_rows(ROOT / "data/capability_train_scaled.jsonl")
    rng = random.Random(args.seed)
    rng.shuffle(capability_rows)

    checkpoint_base = EXP_DIR / ("checkpoints_smoke" if smoke() else "checkpoints")
    if smoke() and checkpoint_base.exists():
        shutil.rmtree(checkpoint_base)
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    arms = [
        {"arm": "baseline_zero_shot_v2", "seed": 42, "train": False},
        {"arm": "lora_v2_seed42", "seed": 42, "train": True},
        {"arm": "lora_v2_seed7", "seed": 7, "train": True},
        {"arm": "lora_v2_seed123", "seed": 123, "train": True},
    ]
    done = completed_arms()
    for spec in arms:
        arm = spec["arm"]
        if arm in done:
            print(f"[Exp156] skipping completed arm {arm}", flush=True)
            continue
        arm_t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"[Exp156] arm {arm}", flush=True)
        random.seed(spec["seed"])
        torch.manual_seed(spec["seed"])
        train_rows = list(v2_train)
        random.Random(spec["seed"]).shuffle(train_rows)
        model = load_model(MODEL_NAME, tokenizer, device)
        train_info = None
        if spec["train"]:
            ckpt_root = checkpoint_base / arm
            final_adapter = ckpt_root / "final_adapter"
            resume_adapter = final_adapter if final_adapter.exists() else latest_checkpoint(ckpt_root)[0]
            lora_layers = None
            model = make_lora(model, ATTN_MLP_TARGETS, lora_layers, resume_adapter, args.rank, args.lora_alpha, args.lora_dropout)
            train_info = train_lora(
                model,
                tokenizer,
                train_rows,
                capability_rows,
                device,
                ckpt_root,
                args.steps,
                args.batch_size,
                args.max_len,
                args.lr,
                args.warmup,
                args.checkpoint_interval,
                False,
            )
        eval_out = endpoint(model, tokenizer, original_eval, v2_eval, device, args.continuation_n, args.max_new_tokens)
        record = {
            "arm": arm,
            "model": MODEL_NAME,
            "smoke_test": smoke(),
            "fixed_update_budget": args.steps,
            "lora": {"rank": args.rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": ATTN_MLP_TARGETS, "layers_to_transform": "all"},
            "seed": spec["seed"],
            "evaluation_surface": "behavior_scaled120 is corpus-v2 entity-disjoint heldout, not the original compositional scaled v1",
            "train_info": train_info,
            "evaluation": eval_out,
            "behavior_scaled120": eval_out["behavior_scaled120"],
            "memory": memory_stats(device),
            "wall_sec": round(time.time() - arm_t0, 2),
        }
        append_arm_result(record)
        print(f"  {arm}: exact_v2={eval_out['behavior_scaled120']['exact_compliance_rate']}% unsafe={eval_out['behavior_scaled120']['unsafe_refusal_rate']}% benign_fr={eval_out['behavior_scaled120']['benign_false_refusal_rate']}%", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    arm_records = [json.loads(line) for line in ARM_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats_arms = [dict(rec, measurement=rec.get("evaluation", {})) for rec in arm_records]
    summary = {
        "experiment": "exp156_qwen3_entity_disjoint_seed_replication",
        "model": MODEL_NAME,
        "question": "Is the critical scaled-data plain-LoRA result robust across seeds on entity-disjoint generalization data, and which exact-compliance field still fails?",
        "design": {
            "fixed_update_budget_steps": args.steps,
            "data_comparison": "same v2 train distribution across seeds 42/7/123; original 40-task eval retained only as a cross-experiment bridge",
            "critical_reference": "lora_v2_seed42/7/123",
            "error_decomposition": "hierarchical JSON/action/policy/tool errors plus independent per-tool metrics",
            "cache_note": "all interventions are weight/LoRA edits, so normal cached generation is valid",
        },
        "arms": arm_records,
        "behavior_row_statistics": {
            "versus_lora_v2_seed42": comparisons_against_reference(stats_arms, "lora_v2_seed42", "behavior_scaled120"),
            "versus_baseline_zero_shot_v2": comparisons_against_reference(stats_arms, "baseline_zero_shot_v2", "behavior_scaled120"),
        },
        "capability_row_statistics": {
            "continuation_versus_baseline_zero_shot_v2": benchmark_comparisons_against_reference(stats_arms, "baseline_zero_shot_v2", "continuation_benchmark"),
            "transfer_versus_baseline_zero_shot_v2": benchmark_comparisons_against_reference(stats_arms, "baseline_zero_shot_v2", "transfer_benchmark"),
        },
        "wall_sec": round(time.time() - t0, 2),
        "smoke_test": smoke(),
    }
    write_json(RESULTS_DIR / "qwen3_entity_disjoint_seed_summary.json", summary)
    if smoke():
        shutil.rmtree(checkpoint_base, ignore_errors=True)
    print(f"[DONE] exp156 arms={len(arm_records)} wall={summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
