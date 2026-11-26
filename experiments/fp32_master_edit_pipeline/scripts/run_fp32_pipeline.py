#!/usr/bin/env python3
"""Experiment 34: fp32-Master Edit -> bf16 Cast Production Pipeline.

Wave-1 finding: editing directly in bf16 destroys the circuit (DLA 15% vs random 47.5% —
a full inversion of fp32 results). Production Halo trains in bf16, so the comp-critical
question: apply the surgical edit in fp32 master weights, THEN cast to bf16 — does the
edit survive the cast? Arms: (a) fp32 edit eval-fp32 (reference), (b) fp32 edit -> bf16
deterministic cast, (c) fp32 edit -> bf16 stochastic-rounding cast, (d) direct bf16 edit (control).
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

def stochastic_round_bf16(w: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding to bf16 via fp32 bit manipulation."""
    g = torch.Generator(device=w.device).manual_seed(0)
    wf = w.float()
    lo = wf.to(torch.bfloat16).float()
    hi = torch.where(lo < wf, lo + (2 ** -8) * lo.abs().clamp(min=2**-126), lo)  # approx ulp step
    prob = torch.rand(wf.shape, generator=g, device=wf.device)
    use_hi = prob < ((wf - lo) / (hi - lo + 1e-20)).clamp(0, 1)
    out = torch.where(use_hi, hi, lo)
    return out.to(torch.bfloat16)

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "experiments/fp32_master_edit_pipeline/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 34] fp32-Master Edit Pipeline on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # fp32 master: load, score, select, edit — all fp32
    raw32 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft32 = PeftModel.from_pretrained(raw32, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft32.eval()
    scores, u = rnd_mod.compute_dla_scores(sft32, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}

    results = {}
    # arm a: fp32 edit, eval fp32 (reference — should reproduce CPU-grade result)
    m = copy.deepcopy(sft32)
    rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=0.7)
    ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
    results["a_fp32_edit_fp32_eval"] = ev
    print(f"  [a fp32/fp32      ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
    del m
    # arm b: fp32 edit -> deterministic bf16 cast, eval bf16
    m = copy.deepcopy(sft32)
    rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=0.7)
    m = m.to(torch.bfloat16)
    ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
    results["b_fp32edit_bf16cast"] = ev
    print(f"  [b fp32edit->bf16 ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
    del m
    # arm c: fp32 edit -> stochastic-rounding bf16 cast
    m = copy.deepcopy(sft32)
    rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=0.7)
    with torch.no_grad():
        for li in (29, 28):
            p = m.model.layers[li].mlp.down_proj.weight
            p.copy_(stochastic_round_bf16(p))
    m = m.to(torch.bfloat16)
    ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
    results["c_fp32edit_stochastic_bf16"] = ev
    print(f"  [c stochastic bf16] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
    del m
    # arm d: direct bf16 edit (wave-1 control)
    rawb = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.bfloat16).to(device)
    sftb = PeftModel.from_pretrained(rawb, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sftb.eval()
    scoresb, ub = rnd_mod.compute_dla_scores(sftb, tok, train_rows, device)
    selb = {li: torch.topk(scoresb[li], k=64).indices.tolist() for li in (29, 28)}
    rnd_mod.apply_neuron_subset_edit(sftb, selb, ub, strength=0.7)
    ev = base_rl.evaluate_heldout(sftb, tok, eval_rows, device)
    results["d_direct_bf16_edit"] = ev
    print(f"  [d direct bf16    ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)

    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "results": results,
               "question": "does fp32-edit-then-cast preserve surgical precision in a bf16 production model?"}
    (out/"fp32_pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 34 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
