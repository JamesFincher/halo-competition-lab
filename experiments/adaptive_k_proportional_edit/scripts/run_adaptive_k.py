#!/usr/bin/env python3
"""Experiment 24: Adaptive-K Proportional Editing.
Per-neuron continuous strength scaled by DLA salience instead of binary top-K uniform edit."""
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
    dtype = torch.float32  # fp32 required: bf16 coarsens rank-1 surgical edits (wave-1 finding)
    out = ROOT / "experiments/adaptive_k_proportional_edit/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 24] Adaptive-K Proportional Edit on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)
    results = []
    for mode in ("uniform", "linear", "sqrt", "square"):
        for k in (64, 128):
            m = copy.deepcopy(sft)
            with torch.no_grad():
                for li in (29, 28):
                    sc = scores[li]
                    topv, topi = torch.topk(sc, k=k)
                    w = m.model.layers[li].mlp.down_proj.weight
                    uu = u.to(w.device, w.dtype)
                    cols = w[:, topi]
                    proj = torch.outer(uu, uu @ cols)
                    if mode == "uniform":
                        scale = torch.full_like(topv, 0.7)
                    else:
                        norm = topv / (topv.max() + 1e-8)
                        if mode == "linear": scale = 0.7 * norm
                        elif mode == "sqrt": scale = 0.7 * norm.sqrt()
                        else: scale = 0.7 * norm * norm
                    w[:, topi] = cols - scale.unsqueeze(0).to(cols.device, cols.dtype) * proj
            ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
            print(f"  [{mode} k={k}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
            results.append({"mode": mode, "k": k, **ev})
            del m
            if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "baseline": base_ev, "arms": results}
    (out/"adaptive_k_summary.json").write_text(json.dumps(summary, indent=2))
    best = max(results, key=lambda r: r["exact_compliance_rate"])
    print(f"[Exp 24 DONE] best={best['mode']} k={best['k']} exact={best['exact_compliance_rate']:.1f}% in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
