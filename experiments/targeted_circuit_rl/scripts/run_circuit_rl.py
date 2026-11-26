#!/usr/bin/env python3
"""Experiment 32: Targeted-Circuit RL — RL constrained to circuit-identified neurons only.

Novel combination: use DLA circuit localization not just for editing but as the RL training mask.
Instead of training whole complementary layers, train ONLY the down_proj columns of the exact
top-K circuit neurons (+ freeze everything else incl. edited layers). Tests whether extreme
surgical RL preserves safety better than layer-level disjoint RL at equal parameter count.
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
import run_micro_targeted_rl as base_rl
import run_random_control_proof as rnd_mod

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = ROOT / "experiments/targeted_circuit_rl/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 32] Targeted-Circuit RL on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()

    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    base = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(base, sel, u, strength=0.7)
    ev0 = base_rl.evaluate_heldout(base, tok, eval_rows, device)
    print(f"  [post-edit pre-RL] exact={ev0['exact_compliance_rate']:.1f}%", flush=True)

    # circuit RL: train down_proj ONLY (columns of complementary circuit in layers 24-27)
    # implemented via full-param requires_grad on down_proj of layers 24-27 with column-masked grads
    results = {"baseline_after_edit": ev0, "arms": []}
    for arm_name, train_mode in [("circuit_columns", "columns"), ("full_layers_control", "layers")]:
        m = copy.deepcopy(base)
        ref = copy.deepcopy(base).eval()
        for p in ref.parameters(): p.requires_grad_(False)
        for p in m.parameters(): p.requires_grad_(False)
        trainable = []
        col_masks = {}
        if train_mode == "columns":
            for li in (27, 26):
                sc = scores[li]
                idxs = torch.topk(sc, k=64).indices
                p = m.model.layers[li].mlp.down_proj.weight
                p.requires_grad_(True)
                trainable.append(p)
                mask = torch.zeros_like(p)
                mask[:, idxs] = 1.0
                col_masks[p] = mask.to(p.device)
        else:
            for li in (27, 26):
                for name, p in m.model.layers[li].named_parameters():
                    if "mlp" in name:
                        p.requires_grad_(True); trainable.append(p)
        hooks = []
        if col_masks:
            for p, mask in col_masks.items():
                hooks.append(p.register_hook(lambda g, msk=mask: g * msk))
        opt = torch.optim.AdamW(trainable, lr=8e-5, weight_decay=0.0)
        ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
        uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
        trace = []
        for step in range(1, 9):
            batch = [ben[(step*2+i) % len(ben)] for i in range(2)] + [uns[(step+i) % len(uns)] for i in range(2)]
            m.eval(); rollouts = []
            for r in batch:
                ptok = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
                plen = ptok.shape[1]
                with torch.no_grad():
                    gens = m.generate(ptok, max_new_tokens=64, do_sample=True, temperature=0.8, num_return_sequences=2, pad_token_id=tok.eos_token_id)
                rews, seqs = [], []
                for g in gens:
                    comp = tok.decode(g[plen:], skip_special_tokens=True)
                    rw, _ = base_rl.score_completion(comp, r["target"])
                    rews.append(float(rw)); seqs.append(g)
                mr = sum(rews)/len(rews)
                for seq, rw in zip(seqs, rews): rollouts.append((seq, plen, rw - mr, rw))
            m.train(); opt.zero_grad(set_to_none=True)
            losses = []
            for seq, plen, adv, _rw in rollouts:
                if seq.shape[0] <= plen: continue
                ids = seq.unsqueeze(0)
                lp = base_rl.completion_logprobs(m, ids, plen)
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref, ids, plen)
                kl = (lp - lp_ref).abs().mean()
                losses.append(-adv * lp.mean() + 0.08 * kl)
            if losses:
                lsum = torch.stack(losses).mean()
                lsum.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
            rew_mean = sum(x[3] for x in rollouts)/max(len(rollouts),1)
            trace.append({"step": step, "mean_reward": round(rew_mean,3)})
            print(f"    [{arm_name} step {step}/8] rew={rew_mean:.3f}", flush=True)
        for h in hooks: h.remove()
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results["arms"].append({"arm": arm_name, "mode": train_mode, "eval": ev, "trace": trace})
        del m, ref
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), **results}
    (out/"circuit_rl_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 32 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
