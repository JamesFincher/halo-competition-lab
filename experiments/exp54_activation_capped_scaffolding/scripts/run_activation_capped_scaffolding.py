#!/usr/bin/env python3
"""Experiment 54: Activation-Capped Rollout Scaffolding Annealed into RL.

Implements SUBAGENT1 rank-5: cap false-refusal activation components during
early RL rollouts, linearly anneal the scaffold to zero, and evaluate with the
cap off. Compares permanent-cap RL, RL-only, cap-annealed RL, and cap-only
inference.

Research anchors: arXiv:2610.29129 and arXiv:2604.01476.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp53_dual_sided_advantage_sculpting/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
import run_dual_sided_advantage_rl as sculpt  # noqa: E402

CAP_LAYERS = [28, 29]


def cap_hook(vec: torch.Tensor, threshold: float, strength: float):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        v = vec.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        proj = (hidden * v).sum(dim=-1, keepdim=True)
        excess = torch.clamp(proj - float(threshold), min=0.0)
        capped = hidden - float(strength) * excess * v
        return (capped,) + tuple(out[1:]) if isinstance(out, tuple) else capped
    return hook


class CapController:
    def __init__(self, model, caps: dict[int, dict]):
        self.model = model
        self.caps = caps
        self.handles = []

    def set(self, strength: float) -> None:
        self.remove()
        if strength <= 0.0:
            return
        for layer, cap in self.caps.items():
            self.handles.append(self.model.model.layers[layer].register_forward_hook(cap_hook(cap["vec"], cap["threshold"], strength)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.remove()


def mean_stack(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs).mean(dim=0)


def build_caps(model, tokenizer, train_rows, device, action_ids) -> dict[int, dict]:
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"][:8]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"][:8]
    false_refusal_text = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'
    unsafe_pass_text = '{"action": "call_tool", "tool": "read_file", "policy_code": "ALLOW", "reason": "Called read_file."}'
    caps = {}
    for layer in CAP_LAYERS:
        sculpt.PROBE_LAYER = layer
        false_refusal, benign_allow, true_refusal, unsafe_pass = [], [], [], []
        for row in benign:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            h_correct = sculpt.hidden_at_anchor(model, tokenizer, prompt, row["assistant_text"], device, action_ids)
            h_false = sculpt.hidden_at_anchor(model, tokenizer, prompt, false_refusal_text, device, action_ids)
            if h_correct is not None:
                benign_allow.append(h_correct)
            if h_false is not None:
                false_refusal.append(h_false)
        for row in unsafe:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            h_true = sculpt.hidden_at_anchor(model, tokenizer, prompt, row["assistant_text"], device, action_ids)
            h_pass = sculpt.hidden_at_anchor(model, tokenizer, prompt, unsafe_pass_text, device, action_ids)
            if h_true is not None:
                true_refusal.append(h_true)
            if h_pass is not None:
                unsafe_pass.append(h_pass)
        v_over = mean_stack(false_refusal) - mean_stack(benign_allow)
        v_true = mean_stack(true_refusal) - mean_stack(unsafe_pass)
        v_true = v_true / (v_true.norm() + 1e-8)
        v_over = v_over - torch.dot(v_over, v_true) * v_true
        v_over = v_over / (v_over.norm() + 1e-8)
        projections = torch.stack([h @ v_over for h in benign_allow])
        threshold = float(torch.quantile(projections, 0.90).item())
        caps[layer] = {
            "vec": v_over.detach().cpu(),
            "threshold": threshold,
            "threshold_source": "90th percentile benign-allow projection",
            "n_false_refusal": len(false_refusal),
            "n_benign_allow": len(benign_allow),
            "cos_over_true_after_orthogonalization": float(torch.dot(v_over, v_true).item()),
        }
    sculpt.PROBE_LAYER = 29
    return caps


def cap_strength_for_step(arm: str, step: int, steps: int) -> float:
    if arm == "permanent_cap_rl":
        return 1.0
    if arm != "cap_annealed_rl":
        return 0.0
    if step <= 2:
        return 1.0
    if steps <= 2:
        return 0.0
    return max(0.0, 1.0 - (step - 2) / (steps - 2))


def train_capped_arm(arm_name, init_model, ref_model, tokenizer, prompt_stream, device, caps, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vec_rl.DISJOINT_LAYERS, device)
    opt = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    trace = []
    t0 = time.perf_counter()

    with CapController(model, caps) as cap_controller:
        for step in range(1, steps + 1):
            cap_strength = cap_strength_for_step(arm_name, step, steps)
            rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
            model.eval()
            groups = []
            all_rewards = []
            for row in rows:
                prompt = base_rl.build_prompt(tokenizer, row["messages"])
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
                prompt_len = enc["input_ids"].shape[1]
                cap_controller.set(cap_strength)
                with torch.no_grad():
                    gens = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, num_return_sequences=group_size, pad_token_id=tokenizer.eos_token_id)
                cap_controller.remove()
                items = []
                for seq in gens:
                    text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    reward, info = base_rl.score_completion(text, row["target"])
                    all_rewards.append(float(reward))
                    items.append({"seq": seq, "reward": float(reward), "exact": info["exact"]})
                r = torch.tensor([x["reward"] for x in items], dtype=torch.float32, device=device)
                if float(r.std(unbiased=False).item()) < 1e-6:
                    continue
                adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
                for item, a in zip(items, adv):
                    item["adv"] = a.detach()
                groups.append({"items": items, "prompt_len": prompt_len})

            model.train()
            opt.zero_grad(set_to_none=True)
            losses = []
            kls = []
            for group in groups:
                for item in group["items"]:
                    ids = item["seq"].unsqueeze(0)
                    lp = base_rl.completion_logprobs(model, ids, group["prompt_len"])
                    with torch.no_grad():
                        lp_ref = base_rl.completion_logprobs(ref_model, ids, group["prompt_len"])
                    kl = (lp - lp_ref).abs().mean()
                    kls.append(float(kl.detach().item()))
                    losses.append(-item["adv"] * lp + beta * kl)
            loss_val = grad_norm = 0.0
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
                opt.step()
                loss_val = float(loss.detach().item())
            rec = {
                "arm": arm_name,
                "step": step,
                "cap_strength": round(cap_strength, 4),
                "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4),
                "active_groups": len(groups),
                "dead_groups": 2 - len(groups),
                "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0,
                "loss": round(loss_val, 5),
                "grad_norm": round(grad_norm, 5),
            }
            trace.append(rec)
            print(f"  [{arm_name}] step {step:02d}/{steps} | cap={cap_strength:.2f} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']}", flush=True)

    stats = {"trace": trace, "train_wall_sec": round(time.perf_counter() - t0, 2), "n_vector_params": sum(v.numel() for v in vectors.values())}
    model.eval()
    return model, stats


def evaluate_with_optional_cap(model, tokenizer, eval_rows, device, caps, cap_on: bool) -> dict:
    if not cap_on:
        return base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)
    with CapController(model, caps) as cap_controller:
        cap_controller.set(1.0)
        return base_rl.evaluate_heldout(model, tokenizer, eval_rows, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp54_activation_capped_scaffolding/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 54] Activation-Capped Rollout Scaffolding on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, dtype)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    selected = edit_info["selected_neurons"]
    ref_model = copy.deepcopy(edited).to(device).eval()
    vec_rl.freeze_weights(ref_model)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [validated edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)

    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[0] for x in ("call", "ref"))
    print("[Stage 1] Calibrating layer-28/29 false-refusal caps from benign-allow activations...", flush=True)
    caps = build_caps(edited, tokenizer, train_rows, device, action_ids)
    print("  cap thresholds=" + ", ".join(f"L{k}:{v['threshold']:.3f}" for k, v in caps.items()), flush=True)

    print("[Stage 2] Measuring cap-only inference (no RL)...", flush=True)
    cap_only_eval_on = evaluate_with_optional_cap(edited, tokenizer, eval_rows, device, caps, True)
    print(f"  [cap_only_inference cap-on] exact={cap_only_eval_on['exact_compliance_rate']:.1f}% | unsafe={cap_only_eval_on['unsafe_refusal_rate']:.1f}% | benign_fr={cap_only_eval_on['benign_false_refusal_rate']:.1f}%", flush=True)

    results = {
        "experiment": "exp54_activation_capped_scaffolding",
        "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 5",
        "paper_inspiration": ["arXiv:2610.29129", "arXiv:2604.01476"],
        "config": vars(args),
        "device": str(device),
        "selected_neurons": selected,
        "validated_edit_operator": edit_info,
        "caps": {str(k): {kk: vv for kk, vv in cap.items() if kk != "vec"} for k, cap in caps.items()},
        "baseline_sft": sft_eval,
        "anchored_edit_only_cap_off": edit_eval,
        "cap_only_inference_cap_on": cap_only_eval_on,
        "arms": {},
    }

    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm_name in ["rl_only_control", "cap_annealed_rl", "permanent_cap_rl"]:
            print(f"[Run] {arm_name}", flush=True)
            trained, stats = train_capped_arm(arm_name, edited, ref_model, tokenizer, prompt_stream, device, caps, args.steps, args.group_size, args.lr, args.beta, args.seed)
            for rec in stats["trace"]:
                log.write(json.dumps(rec) + "\n")
            log.flush()
            eval_off = evaluate_with_optional_cap(trained, tokenizer, eval_rows, device, caps, False)
            eval_on = evaluate_with_optional_cap(trained, tokenizer, eval_rows, device, caps, True)
            print(f"  [{arm_name} cap-off] exact={eval_off['exact_compliance_rate']:.1f}% | unsafe={eval_off['unsafe_refusal_rate']:.1f}% | benign_fr={eval_off['benign_false_refusal_rate']:.1f}%", flush=True)
            print(f"  [{arm_name} cap-on ] exact={eval_on['exact_compliance_rate']:.1f}% | unsafe={eval_on['unsafe_refusal_rate']:.1f}% | benign_fr={eval_on['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm_name] = {"train_stats": stats, "eval_cap_off": eval_off, "eval_cap_on": eval_on}
            del trained
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "activation_capped_scaffolding_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 54 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
