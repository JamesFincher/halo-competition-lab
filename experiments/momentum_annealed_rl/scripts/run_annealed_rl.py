#!/usr/bin/env python3
"""Experiment 29: Momentum-Annealed Surgery+RL.
Start with strong surgical edit, anneal edit strength toward 0 as RL steps progress — hand-off from manual edit to learned behavior."""
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

def load_sft(device, dtype):
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()
    return sft, tok

def load_data():
    return (base_rl.load_rows(ROOT / "data/policy_train.jsonl"), base_rl.load_rows(ROOT / "data/policy_eval.jsonl"))


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = ROOT / "experiments/momentum_annealed_rl/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 29] Momentum-Annealed Surgery+RL on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    wdown_orig = {}
    for li in (29, 28):
        wdown_orig[li] = sft.model.layers[li].mlp.down_proj.weight.detach().clone()
    results = {"arms": []}
    for arm_name, anneal in [("fixed_edit", False), ("annealed_edit", True)]:
        m = copy.deepcopy(sft)
        rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=0.9)
        ref = copy.deepcopy(m).eval()
        for p in ref.parameters(): p.requires_grad_(False)
        for p in m.parameters(): p.requires_grad_(False)
        trainable = []
        for li in (0, 1, 24, 25, 26, 27):
            for name, p in m.model.layers[li].named_parameters():
                if "mlp" in name or "o_proj" in name:
                    p.requires_grad_(True); trainable.append(p)
        opt = torch.optim.AdamW(trainable, lr=8e-5, weight_decay=0.0)
        ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
        uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
        trace = []
        for step in range(1, 9):
            if anneal:
                frac = max(0.0, 1.0 - step / 8.0)
                with torch.no_grad():
                    for li in (29, 28):
                        idxs = sel[li]
                        w = m.model.layers[li].mlp.down_proj.weight
                        orig_cols = wdown_orig[li][:, idxs].to(w.device)
                        uu = u.to(w.device, w.dtype)
                        proj = torch.outer(uu, uu @ orig_cols)
                        w[:, idxs] = orig_cols - 0.9 * frac * proj
                edit_now = 0.9 * frac
            else:
                edit_now = 0.9
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
            trace.append({"step": step, "edit_strength": round(edit_now,3), "mean_reward": round(rew_mean,3)})
            print(f"    [{arm_name} step {step}/8] edit={edit_now:.2f} rew={rew_mean:.3f}", flush=True)
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results["arms"].append({"arm": arm_name, "annealed": anneal, "eval": ev, "trace": trace})
        del m, ref
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), **results}
    (out/"annealed_rl_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 29 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
