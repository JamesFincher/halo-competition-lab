#!/usr/bin/env python3
"""Experiment 50: Surgically Anchored Bias/Steering-Vector RL.

Converts SUBAGENT1 rank-1 research idea into a runnable micro-benchmark:
freeze the surgically edited model and train only additive residual-stream
vectors. Tests whether tiny input-invariant steering vectors can harden the
zero-gradient DLA edit without letting RL overwrite the surgical mechanism.

Arms:
  - all_layer_vectors: zero-init vectors at all 30 block outputs (17,280 params)
  - disjoint_layer_vectors: vectors only at the six complementary RL layers (3,456 params)
  - random_layer_control: matched six random non-edited layers
  - lora_disjoint_control: ordinary r=16 LoRA GRPO on complementary layers

Research anchors: arXiv:2505.18706 (bias-only adaptation), arXiv:2609.18587
(label-free steering compression), and arXiv:2509.06608 (small steering vectors).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402

EDIT_LAYERS = [29, 28]
DISJOINT_LAYERS = [0, 1, 24, 25, 26, 27]


def load_sft(device: torch.device, dtype: torch.dtype) -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    model = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    model.eval()
    return model, tokenizer


def freeze_weights(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def steering_hook(vector: torch.Tensor):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        shifted = hidden + vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        return (shifted,) + tuple(out[1:]) if isinstance(out, tuple) else shifted

    return hook


def attach_vectors(model: torch.nn.Module, layer_ids: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    vectors = {}
    for layer_id in layer_ids:
        vector = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32, requires_grad=True)
        vectors[layer_id] = vector
        model.model.layers[layer_id].register_forward_hook(steering_hook(vector))
    return vectors


def make_prompt_stream(train_rows: list[dict]) -> list[dict]:
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"]
    stream = []
    for b_row, u_row in zip(benign, unsafe):
        stream.extend([b_row, u_row])
    return stream


def train_vector_arm(
    arm_name: str,
    init_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_stream: list[dict],
    device: torch.device,
    layer_ids: list[int],
    steps: int,
    group_size: int,
    lr: float,
    beta: float,
    seed: int,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    freeze_weights(model)
    vectors = attach_vectors(model, layer_ids, device)
    optimizer = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)

    trace = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        step_t0 = time.perf_counter()
        batch_rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval()
        all_rewards = []
        rollout_groups = []
        for row in batch_rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.95,
                    num_return_sequences=group_size,
                    pad_token_id=tokenizer.eos_token_id,
                )
            rewards = []
            seqs = []
            for seq in generated:
                completion = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                reward, _info = base_rl.score_completion(completion, row["target"])
                rewards.append(float(reward))
                seqs.append(seq)
            all_rewards.extend(rewards)
            r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            if float(r_t.std(unbiased=False).item()) < 1e-6:
                continue
            adv = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)
            rollout_groups.append((seqs, prompt_len, adv))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        kls = []
        for seqs, prompt_len, advantages in rollout_groups:
            for seq, advantage in zip(seqs, advantages):
                ids = seq.unsqueeze(0)
                lp_cur = base_rl.completion_logprobs(model, ids, prompt_len)
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref_model, ids, prompt_len)
                kl = (lp_cur - lp_ref).abs().mean()
                kls.append(float(kl.detach().item()))
                losses.append(-advantage.detach() * lp_cur + beta * kl)

        loss_val = 0.0
        grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
            optimizer.step()
            loss_val = float(loss.detach().item())

        vec_norms = {str(k): round(float(v.detach().norm().item()), 5) for k, v in vectors.items()}
        rec = {
            "arm": arm_name,
            "step": step,
            "mean_verifier_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4),
            "active_groups": len(rollout_groups),
            "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0,
            "loss": round(loss_val, 5),
            "grad_norm": round(grad_norm, 5),
            "max_vector_l2": max(vec_norms.values()) if vec_norms else 0.0,
            "step_wall_sec": round(time.perf_counter() - step_t0, 2),
        }
        trace.append(rec)
        print(
            f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_verifier_reward']:.3f} | "
            f"active={rec['active_groups']}/2 | kl={rec['mean_abs_kl']:.4f} | max||v||={rec['max_vector_l2']:.3f} | "
            f"grad={rec['grad_norm']:.3f} | {rec['step_wall_sec']:.1f}s",
            flush=True,
        )

    stats = {
        "layer_ids": layer_ids,
        "n_vector_params": sum(v.numel() for v in vectors.values()),
        "vector_param_pct_of_135m": round(100.0 * sum(v.numel() for v in vectors.values()) / 135_000_000.0, 5),
        "final_vector_l2_norms": {str(k): round(float(v.detach().norm().item()), 5) for k, v in vectors.items()},
        "final_vectors": {str(k): [round(x, 6) for x in v.detach().cpu().tolist()] for k, v in vectors.items()},
        "train_wall_sec": round(time.perf_counter() - t0, 2),
        "trace": trace,
    }
    model.eval()
    return model, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--vector-lr", type=float, default=2e-2)
    parser.add_argument("--lora-lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-lora-control", action="store_true")
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # fp32 is mandatory for surgical-edit fidelity (wave-1 bf16 failure)
    out_dir = ROOT / "experiments/exp50_surgically_anchored_vector_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 50] Surgically Anchored Vector RL on {device} ({dtype})", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = make_prompt_stream(train_rows)
    sft, tokenizer = load_sft(device, dtype)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(
        f"  [baseline SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | pol={sft_eval['policy_accuracy']:.1f}% | "
        f"unsafe={sft_eval['unsafe_refusal_rate']:.1f}% | benign_fr={sft_eval['benign_false_refusal_rate']:.1f}%",
        flush=True,
    )

    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=EDIT_LAYERS, top_columns=64, strength=0.90)
    selected = edit_info["selected_neurons"]
    edited_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(
        f"  [anchored validated edit] exact={edited_eval['exact_compliance_rate']:.1f}% | pol={edited_eval['policy_accuracy']:.1f}% | "
        f"unsafe={edited_eval['unsafe_refusal_rate']:.1f}% | benign_fr={edited_eval['benign_false_refusal_rate']:.1f}%",
        flush=True,
    )

    ref_model = copy.deepcopy(edited).to(device).eval()
    freeze_weights(ref_model)
    rng = random.Random(args.seed)
    random_layers = sorted(rng.sample([li for li in range(edited.config.num_hidden_layers) if li not in EDIT_LAYERS], k=len(DISJOINT_LAYERS)))
    vector_arms = [
        ("all_layer_vectors", list(range(edited.config.num_hidden_layers))),
        ("disjoint_layer_vectors", DISJOINT_LAYERS),
        ("random_layer_control", random_layers),
    ]

    results = {
        "experiment": "exp50_surgically_anchored_vector_rl",
        "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 1",
        "paper_inspiration": ["arXiv:2505.18706", "arXiv:2609.18587", "arXiv:2509.06608"],
        "config": vars(args),
        "device": str(device),
        "edit_layers": EDIT_LAYERS,
        "disjoint_layers": DISJOINT_LAYERS,
        "random_layers": random_layers,
        "selected_neurons": selected,
        "validated_edit_operator": edit_info,
        "baseline_sft": sft_eval,
        "anchored_edit_only": edited_eval,
        "arms": {},
    }

    step_log_path = out_dir / "step_metrics.jsonl"
    with step_log_path.open("w", encoding="utf-8") as step_log:
        for arm_name, layer_ids in vector_arms:
            print(f"[Run] {arm_name}: frozen edited weights + vectors at {layer_ids}", flush=True)
            trained, stats = train_vector_arm(
                arm_name,
                edited,
                ref_model,
                tokenizer,
                prompt_stream,
                device,
                layer_ids,
                args.steps,
                args.group_size,
                args.vector_lr,
                args.beta,
                args.seed,
            )
            for rec in stats["trace"]:
                step_log.write(json.dumps(rec) + "\n")
            step_log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            print(
                f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | "
                f"unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%",
                flush=True,
            )
            results["arms"][arm_name] = {"train_stats": stats, "eval": ev}
            del trained
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not args.skip_lora_control:
            target_regex = f".*\\.layers\\.({'|'.join(map(str, DISJOINT_LAYERS))})\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"
            print(f"[Run] lora_disjoint_control: r=16 LoRA GRPO at layers {DISJOINT_LAYERS}", flush=True)
            trained_lora, lora_stats = base_rl.run_grpo_arm(
                "lora_disjoint_control",
                edited,
                ref_model,
                tokenizer,
                prompt_stream,
                device,
                target_regex,
                16,
                args.steps,
                args.group_size,
                args.lora_lr,
                args.beta,
                step_log,
                args.seed,
            )
            ev = base_rl.evaluate_heldout(trained_lora, tokenizer, eval_rows, device)
            print(
                f"  [lora_disjoint_control] exact={ev['exact_compliance_rate']:.1f}% | pol={ev['policy_accuracy']:.1f}% | "
                f"unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}%",
                flush=True,
            )
            results["arms"]["lora_disjoint_control"] = {"train_stats": lora_stats, "eval": ev}
            del trained_lora
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    summary_path = out_dir / "surgically_anchored_vector_rl_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 50 DONE] wrote {summary_path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
