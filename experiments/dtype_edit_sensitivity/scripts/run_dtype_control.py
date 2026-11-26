#!/usr/bin/env python3
"""Experiment 33: Dtype Sensitivity of Surgical Edits (bf16 vs fp32 vs fp64).

Wave-1 observation: identical DLA edits gave 80% exact / 90% unsafe-refusal in CPU fp32
but safety collapse in CUDA bf16. Directly measures edit precision vs dtype — critical
for competition transfer since Halo production trains in bf16.
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

def run_for_dtype(dtype, device, train_rows, eval_rows, tok):
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    m = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=0.7)
    edit_ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
    # measure edit precision: weight delta magnitude actually applied vs intended
    with torch.no_grad():
        w_orig = sft.model.layers[29].mlp.down_proj.weight.detach().float()
        w_edit = m.model.layers[29].mlp.down_proj.weight.detach().float()
        delta = (w_edit - w_orig).abs().max().item()
    del sft, m
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"baseline": base_ev, "edited": edit_ev, "max_weight_delta": round(delta, 6)}

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "experiments/dtype_edit_sensitivity/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 33] Dtype Sensitivity Control on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    results = {}
    for name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        r = run_for_dtype(dt, device, train_rows, eval_rows, tok)
        results[name] = r
        print(f"  [{name}] baseline exact={r['baseline']['exact_compliance_rate']:.1f}% -> edited exact={r['edited']['exact_compliance_rate']:.1f}% unsafe={r['edited']['unsafe_refusal_rate']:.1f}% benignfr={r['edited']['benign_false_refusal_rate']:.1f}% maxwdelta={r['max_weight_delta']:.5f}", flush=True)
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "results": results,
               "implication": "if bf16 degrades edits, production Halo pipelines need fp32 edit application then bf16 cast, or stochastic rounding"}
    (out/"dtype_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 33 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
