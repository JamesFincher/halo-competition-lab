#!/usr/bin/env python3
"""Experiment 21: Flagship Upgrade — DLA Surgical Edit -> Re-Anchored Disjoint RL.

Combines Experiment 17's zero-grad Direct Logit Lens (DLA) neuron-level edit with
Experiment 11 Fork V3's re-anchored Disjoint RL (edited layers [29, 28] frozen;
complementary layers [0, 1, 24, 25, 26, 27] trained gently with KL anchored to the edited model).
"""

from __future__ import annotations

import copy
import json
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


def run_disjoint_rl_stage(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    train_layers: tuple[int, ...] = (0, 1, 24, 25, 26, 27),
    steps: int = 8,
    lr: float = 6e-5,
    kl_beta: float = 0.08,
) -> list[dict]:
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for li in train_layers:
        for name, p in model.model.layers[li].named_parameters():
            if "mlp" in name or "o_proj" in name:
                p.requires_grad_(True)
                trainable.append(p)

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    trace = []

    for step in range(1, steps + 1):
        batch = [ben[(step * 2 + i) % len(ben)] for i in range(2)] + [uns[(step + i) % len(uns)] for i in range(2)]
        model.eval()
        rollouts = []
        for r in batch:
            prompt_txt = base_rl.build_prompt(tokenizer, r["messages"])
            p_ids = tokenizer(prompt_txt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
            p_len = p_ids.shape[1]
            with torch.no_grad():
                gens = model.generate(
                    p_ids,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            rews = []
            seqs = []
            for g in gens:
                comp = tokenizer.decode(g[p_len:], skip_special_tokens=True)
                rw, _info = base_rl.score_completion(comp, r["target"])
                rews.append(float(rw))
                seqs.append(g)
            mean_r = sum(rews) / len(rews)
            for seq, rw in zip(seqs, rews):
                rollouts.append((seq, p_len, rw - mean_r, rw))

        model.train()
        opt.zero_grad(set_to_none=True)
        loss_terms = []
        kls = []
        for seq, p_len, adv, _rw in rollouts:
            if seq.shape[0] <= p_len:
                continue
            full_ids = seq.unsqueeze(0)
            lp_cur = base_rl.completion_logprobs(model, full_ids, p_len)
            with torch.no_grad():
                lp_ref = base_rl.completion_logprobs(ref_model, full_ids, p_len)
            kl = (lp_cur - lp_ref).abs().mean()
            kls.append(float(kl.item()))
            loss_terms.append(-adv * lp_cur.mean() + kl_beta * kl)

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            opt.step()
            step_kl = round(sum(kls) / max(len(kls), 1), 5)
            step_rew = round(sum(x[3] for x in rollouts) / max(len(rollouts), 1), 3)
            trace.append({"step": step, "mean_reward": step_rew, "mean_kl": step_kl, "grad_norm": round(grad_norm, 4)})
            print(f"    [RL step {step}/{steps}] rew={step_rew:.3f} | kl={step_kl:.5f} | gnorm={grad_norm:.3f}", flush=True)
    del ref_model
    return trace


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = ROOT / "experiments/ig5_dla_closed_loop_rl/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 21] Running DLA Surgical Edit -> Re-Anchored Disjoint RL on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()

    dla_scores, u = rnd_mod.compute_dla_scores(sft, tokenizer, train_rows, device)
    dla_sel = {li: torch.topk(dla_scores[li], k=64).indices.tolist() for li in (29, 28)}

    runs = []
    for strength in (0.65, 0.80):
        m = copy.deepcopy(sft)
        rnd_mod.apply_neuron_subset_edit(m, dla_sel, u, strength=strength)
        ev_edit = base_rl.evaluate_heldout(m, tokenizer, eval_rows, device)
        print(f"  [DLA st={strength} edit-only] exact={ev_edit['exact_compliance_rate']:.1f}% | pol={ev_edit['policy_accuracy']:.1f}% | unsafe={ev_edit['unsafe_refusal_rate']:.1f}% | benign_fr={ev_edit['benign_false_refusal_rate']:.1f}%", flush=True)

        trace = run_disjoint_rl_stage(m, tokenizer, train_rows, device, steps=8, lr=6e-5, kl_beta=0.08)
        ev_rl = base_rl.evaluate_heldout(m, tokenizer, eval_rows, device)
        print(f"  [DLA st={strength} + DisjRL ] exact={ev_rl['exact_compliance_rate']:.1f}% | pol={ev_rl['policy_accuracy']:.1f}% | unsafe={ev_rl['unsafe_refusal_rate']:.1f}% | benign_fr={ev_rl['benign_false_refusal_rate']:.1f}%", flush=True)
        runs.append({"strength": strength, "selected_neurons": dla_sel, "after_edit": ev_edit, "after_disjoint_rl": ev_rl, "rl_trace": trace})
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {"device": str(device), "wall_sec": round(time.time() - t0, 2), "runs": runs}
    (out_dir / "dla_disjoint_rl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 21 DONE] Saved dla_disjoint_rl_summary.json in {summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
