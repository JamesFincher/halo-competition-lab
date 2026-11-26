#!/usr/bin/env python3
"""Experiment 56: Hidden-State-Conditioned Boundary-Token Temperature.

Implements SUBAGENT1 rank-7: selective exploration for ultra-short structured
JSON actions. Schema punctuation stays near-deterministic while action/tool/
policy-code decision positions can use high temperature. A tiny hidden-state
controller chooses low vs high temperature and is coordinate-ascent trained
with the same exact-compliance advantage as disjoint steering vectors.

Research anchor: arXiv:2602.13035 (IntroLLM-style selective exploration).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_random_control_proof as rnd_mod  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402

PROBE_LAYER = 29
LOW_TEMP = 0.20
HIGH_TEMP = 1.20
FIXED_TEMP = 0.85
DECISION_SUFFIXES = ('"action": "', '"tool": "', '"policy_code": "')


class TempController(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.float())


def is_decision_position(generated_text: str) -> bool:
    return any(generated_text.endswith(suffix) for suffix in DECISION_SUFFIXES)


def capture_layer29_hook(model, captured):
    def hook(_module, _inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out
    return model.model.layers[PROBE_LAYER].register_forward_hook(hook)


@torch.no_grad()
def manual_sample(model, tokenizer, prompt: str, device, mode: str, controller: TempController | None, max_new_tokens: int = 64) -> tuple[torch.Tensor, list[tuple[int, int]], dict]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=320).to(device)
    prompt_ids = enc["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])
    generated: list[int] = []
    decisions: list[tuple[int, int]] = []
    temps = []
    past = None
    cur = enc["input_ids"]
    entropies = []

    for _step in range(max_new_tokens):
        captured = {}
        handle = capture_layer29_hook(model, captured)
        try:
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
        finally:
            handle.remove()
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        hidden = captured["h"][0, -1, :].detach().float()
        generated_text = tokenizer.decode(generated, skip_special_tokens=False) if generated else ""
        decision = is_decision_position(generated_text)

        if mode == "fixed":
            temp = FIXED_TEMP
            controller_action = -1
        elif mode == "selective_fixed":
            temp = HIGH_TEMP if decision else LOW_TEMP
            controller_action = -1
        elif mode == "controller":
            if decision and controller is not None:
                c_logits = controller(hidden)
                c_probs = F.softmax(c_logits, dim=-1)
                controller_action = int(torch.multinomial(c_probs, 1).item())
                temp = HIGH_TEMP if controller_action == 1 else LOW_TEMP
            else:
                temp = LOW_TEMP
                controller_action = -1
        else:
            raise ValueError(f"unknown mode {mode}")

        probs = F.softmax(logits / max(temp, 1e-4), dim=-1)
        token = int(torch.multinomial(probs, 1).item())
        entropies.append(float((-(probs * torch.log(probs + 1e-12)).sum()).item()))
        temps.append(float(temp))
        abs_pos = prompt_len + len(generated)
        if decision and controller_action >= 0:
            decisions.append((abs_pos, controller_action))
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[token]], device=device, dtype=torch.long)

    seq = torch.cat([prompt_ids.detach().cpu(), torch.tensor(generated, dtype=torch.long)], dim=0).to(device)
    meta = {"mean_temp": sum(temps) / max(len(temps), 1), "mean_entropy": sum(entropies) / max(len(entropies), 1), "n_decision_controller_actions": len(decisions)}
    return seq, decisions, meta


def controller_logprobs(model, controller: TempController, ids: torch.Tensor, decisions: list[tuple[int, int]]) -> list[torch.Tensor]:
    if not decisions:
        return []
    captured = {}
    handle = capture_layer29_hook(model, captured)
    try:
        model(input_ids=ids.unsqueeze(0))
    finally:
        handle.remove()
    hidden = captured["h"][0]
    out = []
    for pos, action in decisions:
        if pos <= 0 or pos - 1 >= hidden.shape[0]:
            continue
        logits = controller(hidden[pos - 1])
        out.append(F.log_softmax(logits, dim=-1)[action])
    return out


def train_temperature_arm(arm_name, mode, init_model, ref_model, tokenizer, prompt_stream, device, steps, group_size, lr, beta, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    vec_rl.freeze_weights(model)
    vectors = vec_rl.attach_vectors(model, vec_rl.DISJOINT_LAYERS, device)
    controller = TempController(model.config.hidden_size).to(device) if mode == "controller" else None
    vector_opt = torch.optim.AdamW(list(vectors.values()), lr=lr, weight_decay=0.0)
    controller_opt = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=0.0) if controller is not None else None
    trace = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        rows = [prompt_stream[(2 * (step - 1)) % len(prompt_stream)], prompt_stream[(2 * (step - 1) + 1) % len(prompt_stream)]]
        model.eval()
        if controller is not None:
            controller.eval()
        groups = []
        all_rewards = []
        rollout_temps = []
        rollout_entropies = []
        for row in rows:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=320).to(device)
            prompt_len = int(enc["input_ids"].shape[1])
            items = []
            for _ in range(group_size):
                seq, decisions, meta = manual_sample(model, tokenizer, prompt, device, mode, controller)
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                reward, info = base_rl.score_completion(text, row["target"])
                all_rewards.append(float(reward))
                rollout_temps.append(meta["mean_temp"])
                rollout_entropies.append(meta["mean_entropy"])
                items.append({"seq": seq, "decisions": decisions, "reward": float(reward), "exact": info["exact"], "meta": meta})
            r = torch.tensor([x["reward"] for x in items], dtype=torch.float32, device=device)
            if float(r.std(unbiased=False).item()) < 1e-6:
                continue
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            for item, a in zip(items, adv):
                item["adv"] = a.detach()
            groups.append({"items": items, "prompt_len": prompt_len})

        model.train()
        if controller is not None:
            controller.train()
        vector_opt.zero_grad(set_to_none=True)
        if controller_opt is not None:
            controller_opt.zero_grad(set_to_none=True)
        losses = []
        kls = []
        ctrl_loss_count = 0
        for group in groups:
            for item in group["items"]:
                ids = item["seq"].unsqueeze(0)
                lp = base_rl.completion_logprobs(model, ids, group["prompt_len"])
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref_model, ids, group["prompt_len"])
                kl = (lp - lp_ref).abs().mean()
                kls.append(float(kl.detach().item()))
                loss = -item["adv"] * lp + beta * kl
                if controller is not None:
                    cps = controller_logprobs(model, controller, item["seq"], item["decisions"])
                    if cps:
                        ctrl_loss_count += len(cps)
                        loss = loss - 0.5 * item["adv"] * torch.stack(cps).mean()
                losses.append(loss)
        loss_val = grad_norm = 0.0
        stepped = "none"
        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            if controller is None or step % 2 == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(list(vectors.values()), 1.0).item())
                vector_opt.step()
                stepped = "vectors"
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(list(controller.parameters()), 1.0).item())
                controller_opt.step()
                stepped = "controller"
            loss_val = float(loss.detach().item())

        rec = {
            "arm": arm_name,
            "step": step,
            "mean_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4),
            "active_groups": len(groups),
            "dead_groups": 2 - len(groups),
            "mean_rollout_temp": round(sum(rollout_temps) / max(len(rollout_temps), 1), 4),
            "mean_token_entropy": round(sum(rollout_entropies) / max(len(rollout_entropies), 1), 4),
            "controller_decisions": ctrl_loss_count,
            "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0,
            "loss": round(loss_val, 5),
            "grad_norm": round(grad_norm, 5),
            "coordinate_step": stepped,
        }
        trace.append(rec)
        print(f"  [{arm_name}] step {step:02d}/{steps} | reward={rec['mean_reward']:.3f} | active={len(groups)}/2 dead={rec['dead_groups']} | temp={rec['mean_rollout_temp']:.2f} | update={stepped}", flush=True)

    stats = {"trace": trace, "train_wall_sec": round(time.perf_counter() - t0, 2), "n_vector_params": sum(v.numel() for v in vectors.values()), "n_controller_params": sum(p.numel() for p in controller.parameters()) if controller is not None else 0}
    model.eval()
    if controller is not None:
        controller.eval()
    return model, controller, stats


@torch.no_grad()
def pass_at_k(model, tokenizer, eval_rows, device, mode, controller, k: int = 4) -> dict:
    pass_exact = pass_json = 0
    mean_reward = 0.0
    for row in eval_rows:
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        best = 0.0
        any_exact = any_json = False
        for _ in range(k):
            seq, _decisions, _meta = manual_sample(model, tokenizer, prompt, device, mode, controller)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=320)
            prompt_len = int(enc["input_ids"].shape[1])
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            reward, info = base_rl.score_completion(text, row["target"])
            best = max(best, float(reward))
            any_exact = any_exact or bool(info["exact"])
            any_json = any_json or bool(info["json_valid"])
        pass_exact += int(any_exact)
        pass_json += int(any_json)
        mean_reward += best
    n = len(eval_rows)
    return {"pass_at_4_exact_rate": round(100 * pass_exact / n, 1), "pass_at_4_json_rate": round(100 * pass_json / n, 1), "mean_best_of_4_reward": round(mean_reward / n, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.steps = min(args.steps, 10)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    out_dir = ROOT / "experiments/exp56_boundary_temperature_controller/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 56] Boundary-Token Temperature Controller on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, dtype)
    scores, u = rnd_mod.compute_dla_scores(sft, tokenizer, train_rows, device)
    selected = {li: torch.topk(scores[li], k=64).indices.tolist() for li in vec_rl.EDIT_LAYERS}
    edited = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(edited, selected, u, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval()
    vec_rl.freeze_weights(ref_model)

    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [DLA] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}%", flush=True)

    results = {
        "experiment": "exp56_boundary_temperature_controller",
        "research_source": "research/SUBAGENT1_FRONTIER_RL_INTERNALS.md rank 7",
        "paper_inspiration": ["arXiv:2602.13035"],
        "config": vars(args),
        "device": str(device),
        "selected_neurons": selected,
        "temperatures": {"low_schema": LOW_TEMP, "high_decision": HIGH_TEMP, "fixed_control": FIXED_TEMP},
        "baseline_sft": sft_eval,
        "anchored_edit_only": edit_eval,
        "arms": {},
    }

    arms = [
        ("fixed_temperature_control", "fixed"),
        ("schema_frozen_selective_fixed", "selective_fixed"),
        ("hidden_controller_selective", "controller"),
    ]
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm_name, mode in arms:
            print(f"[Run] {arm_name}", flush=True)
            trained, controller, stats = train_temperature_arm(arm_name, mode, edited, ref_model, tokenizer, prompt_stream, device, args.steps, args.group_size, args.lr, args.beta, args.seed)
            for rec in stats["trace"]:
                log.write(json.dumps(rec) + "\n")
            log.flush()
            ev = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            sampling_eval = pass_at_k(trained, tokenizer, eval_rows, device, mode, controller, k=4)
            print(f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benign_fr={ev['benign_false_refusal_rate']:.1f}% | pass@4={sampling_eval['pass_at_4_exact_rate']:.1f}%", flush=True)
            results["arms"][arm_name] = {"train_stats": stats, "eval_greedy": ev, "eval_sampling": sampling_eval}
            del trained, controller
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "boundary_temperature_controller_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 56 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
