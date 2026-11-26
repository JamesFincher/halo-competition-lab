#!/usr/bin/env python3
"""Experiment 31: Activation-Vector-Only RL on a Frozen Policy Model.

Inspired by arXiv:2611.26627 ("RL Post-Training in Diffusion Transformers Is
Activation Steering"), which found that fixed activation shifts can recover most
of RL's gains in diffusion transformers. This script tests the language-model
policy analogue: all SmolLM2 weights stay frozen, and only tiny per-layer
residual-stream steering vectors are trained by GRPO-style RL.

Arms:
  1. vector_only_rl: SFT checkpoint + trainable steering vectors at layers 27/28/29.
  2. edit_plus_vector_rl: DLA surgical pre-edit + trainable steering vector at layer 27.
Both use <=10 RL steps, group-centered advantages, and a KL anchor to the arm's
initial checkpoint.
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
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402


def load_sft(device: torch.device, dtype: torch.dtype) -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    model = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    model.eval()
    return model, tokenizer


def make_steer_hook(vector: torch.Tensor):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        steered = hidden + vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered

    return hook


def attach_vectors(model: torch.nn.Module, layer_ids: list[int], device: torch.device) -> tuple[dict[int, torch.Tensor], list]:
    vectors = {}
    hooks = []
    for layer_id in layer_ids:
        vec = torch.zeros(model.config.hidden_size, device=device, dtype=torch.float32, requires_grad=True)
        vectors[layer_id] = vec
        hooks.append(model.model.layers[layer_id].register_forward_hook(make_steer_hook(vec)))
    return vectors, hooks


def freeze_model_weights(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def balanced_batch(train_rows: list[dict], step: int) -> list[dict]:
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"]
    return [benign[(2 * step) % len(benign)], benign[(2 * step + 1) % len(benign)], unsafe[step % len(unsafe)], unsafe[(step + 3) % len(unsafe)]]


def train_vector_arm(
    arm_name: str,
    init_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    steer_layers: list[int],
    steps: int,
    group_size: int,
    lr: float,
    beta: float,
    seed: int,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    random.seed(seed)

    model = copy.deepcopy(init_model).to(device)
    freeze_model_weights(model)
    ref_model = copy.deepcopy(init_model).to(device).eval()
    freeze_model_weights(ref_model)

    vectors, hooks = attach_vectors(model, steer_layers, device)
    initial_norms = {str(k): float(v.detach().norm().item()) for k, v in vectors.items()}
    optimizer = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)

    trace = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        rows = balanced_batch(train_rows, step - 1)
        rollouts = []
        model.eval()
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                gens = model.generate(
                    **enc,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    num_return_sequences=group_size,
                    pad_token_id=tokenizer.eos_token_id,
                )
            rewards = []
            seqs = []
            for seq in gens:
                completion = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                reward, _info = base_rl.score_completion(completion, row["target"])
                rewards.append(float(reward))
                seqs.append(seq)
            r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            if float(r_t.std(unbiased=False).item()) < 1e-6:
                continue
            adv = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)
            for seq, reward, advantage in zip(seqs, rewards, adv):
                rollouts.append({"seq": seq, "prompt_len": prompt_len, "adv": advantage.detach(), "reward": reward})

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        kls = []
        for item in rollouts:
            ids = item["seq"].unsqueeze(0)
            lp_cur = base_rl.completion_logprobs(model, ids, item["prompt_len"])
            with torch.no_grad():
                lp_ref = base_rl.completion_logprobs(ref_model, ids, item["prompt_len"])
            kl = (lp_cur - lp_ref).abs().mean()
            kls.append(float(kl.detach().item()))
            losses.append(-item["adv"] * lp_cur + beta * kl)

        loss_val = 0.0
        grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
            optimizer.step()
            loss_val = float(loss.detach().item())

        mean_reward = sum(x["reward"] for x in rollouts) / max(len(rollouts), 1)
        mean_kl = sum(kls) / max(len(kls), 1) if kls else 0.0
        vec_norms = {str(k): round(float(v.detach().norm().item()), 5) for k, v in vectors.items()}
        rec = {
            "step": step,
            "mean_reward": round(mean_reward, 4),
            "active_rollouts": len(rollouts),
            "mean_abs_kl": round(mean_kl, 5),
            "loss": round(loss_val, 5),
            "grad_norm": round(grad_norm, 5),
            "vector_l2_norms": vec_norms,
        }
        trace.append(rec)
        print(
            f"  [{arm_name}] step {step:02d}/{steps} | reward={mean_reward:.3f} | active={len(rollouts)} | "
            f"kl={mean_kl:.4f} | ||v||={max(vec_norms.values()):.3f} | grad={grad_norm:.3f}",
            flush=True,
        )

    stats = {
        "steer_layers": steer_layers,
        "n_vector_params": sum(v.numel() for v in vectors.values()),
        "initial_vector_l2_norms": initial_norms,
        "final_vector_l2_norms": {str(k): round(float(v.detach().norm().item()), 5) for k, v in vectors.items()},
        "final_vectors": {str(k): [round(x, 6) for x in v.detach().cpu().tolist()] for k, v in vectors.items()},
        "train_wall_sec": round(time.perf_counter() - t0, 2),
        "trace": trace,
    }
    for hook in hooks:
        hook.remove()

    # Re-attach the learned vectors for evaluation and for any downstream inspection.
    eval_hooks = []
    for layer_id, vec in vectors.items():
        eval_hooks.append(model.model.layers[layer_id].register_forward_hook(make_steer_hook(vec.detach())))
    stats["eval_hook_count"] = len(eval_hooks)
    model.eval()
    del ref_model
    return model, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = ROOT / "experiments/activation_vector_only_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 31] Activation-Vector-Only RL on {device} ({dtype})", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    sft, tokenizer = load_sft(device, dtype)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(
        f"  [baseline SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | pol={sft_eval['policy_accuracy']:.1f}% | "
        f"unsafe={sft_eval['unsafe_refusal_rate']:.1f}% | benign_fr={sft_eval['benign_false_refusal_rate']:.1f}%",
        flush=True,
    )

    dla_scores, u_refusal = rnd_mod.compute_dla_scores(sft, tokenizer, train_rows, device)
    dla_sel = {li: torch.topk(dla_scores[li], k=64).indices.tolist() for li in (29, 28)}
    edited = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(edited, dla_sel, u_refusal, strength=0.70)
    edited_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(
        f"  [DLA edit only] exact={edited_eval['exact_compliance_rate']:.1f}% | pol={edited_eval['policy_accuracy']:.1f}% | "
        f"unsafe={edited_eval['unsafe_refusal_rate']:.1f}% | benign_fr={edited_eval['benign_false_refusal_rate']:.1f}%",
        flush=True,
    )

    results = {
        "experiment": "exp31_activation_vector_only_rl",
        "paper_inspiration": "arXiv:2611.26627 RL Post-Training in Diffusion Transformers Is Activation Steering",
        "config": vars(args),
        "device": str(device),
        "baseline_sft": sft_eval,
        "dla_edit_only": edited_eval,
        "dla_selected_neurons": dla_sel,
        "arms": {},
    }

    arm_specs = [
        ("vector_only_rl", sft, [27, 28, 29], args.lr),
        ("edit_plus_vector_rl", edited, [27], args.lr * 0.5),
    ]
    for arm_name, init_model, layers, arm_lr in arm_specs:
        print(f"[Run] {arm_name}: frozen weights, trainable activation vectors at layers {layers}", flush=True)
        trained, stats = train_vector_arm(
            arm_name,
            init_model,
            tokenizer,
            train_rows,
            device,
            layers,
            min(args.steps, 10),
            args.group_size,
            arm_lr,
            args.beta,
            args.seed,
        )
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

    results["wall_sec"] = round(time.time() - t0, 2)
    summary_path = out_dir / "activation_vector_rl_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 31 DONE] wrote {summary_path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
