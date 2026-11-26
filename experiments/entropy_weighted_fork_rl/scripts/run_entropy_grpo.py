#!/usr/bin/env python3
"""Experiment 26: Entropy-Weighted Decision-Fork GRPO.
Weight RL advantage by decision-token entropy — focus gradient where the model is uncertain."""
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


def completion_logprobs_entropy(model, ids, prompt_len):
    out = model(ids)
    logits = out.logits[0, prompt_len-1:-1, :].float()
    lp = torch.log_softmax(logits, dim=-1)
    ent = -(lp.exp() * lp).sum(-1)
    tok_lp = lp.gather(-1, ids[0, prompt_len:].unsqueeze(-1)).squeeze(-1)
    return tok_lp, ent

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = ROOT / "experiments/entropy_weighted_fork_rl/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 26] Entropy-Weighted Fork GRPO on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    base = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(base, sel, u, strength=0.7)
    ev0 = base_rl.evaluate_heldout(base, tok, eval_rows, device)
    print(f"  [post-edit pre-RL] exact={ev0['exact_compliance_rate']:.1f}%", flush=True)
    results = {"baseline_after_edit": ev0, "arms": []}
    for arm_name, ent_weight in [("uniform_adv", 0.0), ("entropy_weighted", 1.0)]:
        m = copy.deepcopy(base)
        ref = copy.deepcopy(base).eval()
        for p in ref.parameters(): p.requires_grad_(False)
        for p in m.parameters(): p.requires_grad_(False)
        trainable = []
        for li in (0, 1, 24, 25, 26, 27):
            for name, p in m.model.layers[li].named_parameters():
                if "mlp" in name or "o_proj" in name:
                    p.requires_grad_(True); trainable.append(p)
        opt = torch.optim.AdamW(trainable, lr=6e-5, weight_decay=0.0)
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
                for seq, rw in zip(seqs, rews): rollouts.append((seq, plen, rw - mr))
            m.train(); opt.zero_grad(set_to_none=True)
            losses = []
            for seq, plen, adv in rollouts:
                if seq.shape[0] <= plen: continue
                ids = seq.unsqueeze(0)
                tok_lp, ent = completion_logprobs_entropy(m, ids, plen)
                w = ent / (ent.sum() + 1e-8) * ent.numel() if ent_weight else torch.ones_like(ent) / ent.numel()
                loss = -(adv * (tok_lp * w.detach()).sum())
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref, ids, plen)
                kl = (tok_lp.mean() - lp_ref.mean()).abs()
                losses.append(loss + 0.08 * kl)
            if losses:
                lsum = torch.stack(losses).mean()
                lsum.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                trace.append({"step": step, "loss": round(float(lsum.item()), 4)})
                print(f"    [{arm_name} step {step}/8] loss={float(lsum.item()):.4f}", flush=True)
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{arm_name}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}%", flush=True)
        results["arms"].append({"arm": arm_name, "entropy_weight": ent_weight, "eval": ev, "trace": trace})
        del m, ref
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), **results}
    (out/"entropy_grpo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 26 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
