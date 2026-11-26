#!/usr/bin/env python3
"""Experiment 23: Dual-Polarity Push-Pull Circuit Surgery.
Suppress refusal direction AND amplify compliance direction in the same edit."""
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
    out = ROOT / "experiments/dual_polarity_circuit_surgery/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 23] Dual-Polarity Push-Pull on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    scores, u_ref = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    tid_ref = tok.encode("ref", add_special_tokens=False)[0]
    tid_call = tok.encode("call", add_special_tokens=False)[0]
    lm_w = sft.lm_head.weight.detach().float()
    u_call = lm_w[tid_call] - lm_w[tid_ref]
    u_call = u_call / (u_call.norm() + 1e-8)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    results = []
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)
    for name, sup, amp in [("suppress_only", 0.70, 0.0), ("pushpull_0.3", 0.70, 0.30), ("pushpull_0.5", 0.70, 0.50), ("pushpull_1.0", 0.70, 1.00)]:
        m = copy.deepcopy(sft)
        with torch.no_grad():
            for li, idxs in sel.items():
                w = m.model.layers[li].mlp.down_proj.weight
                cols = w[:, idxs]
                ur = u_ref.to(w.device, w.dtype); uc = u_call.to(w.device, w.dtype)
                proj_r = torch.outer(ur, ur @ cols)
                proj_c = torch.outer(uc, uc @ cols)
                w[:, idxs] = cols - sup * proj_r + amp * proj_c
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{name}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results.append({"arm": name, "suppress": sup, "amplify": amp, **ev})
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "baseline": base_ev, "arms": results}
    (out/"pushpull_summary.json").write_text(json.dumps(summary, indent=2))
    best = max(results, key=lambda r: r["exact_compliance_rate"])
    print(f"[Exp 23 DONE] best={best['arm']} exact={best['exact_compliance_rate']:.1f}% in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
