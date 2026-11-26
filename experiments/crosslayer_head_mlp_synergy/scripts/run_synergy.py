#!/usr/bin/env python3
"""Experiment 28: Cross-Layer Head+MLP Synergy Edit.
Jointly edit top false attention head (L29H1) plus top MLP neurons in coordinated fashion."""
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
    out = ROOT / "experiments/crosslayer_head_mlp_synergy/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 28] Head+MLP Synergy Edit on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel64 = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    sel32 = {li: torch.topk(scores[li], k=32).indices.tolist() for li in (29, 28)}
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)
    hd = sft.config.head_dim if hasattr(sft.config, "head_dim") else 64
    results = []

    def head_suppress(m, layer, head, strength):
        with torch.no_grad():
            w = m.model.layers[layer].self_attn.o_proj.weight
            uu = u.to(w.device, w.dtype)
            s = head * hd; e = s + hd
            rows = w[:, s:e]
            proj = torch.outer(uu, uu @ rows)
            w[:, s:e] = rows - strength * proj

    arms = [
        ("mlp64_only", dict(mlp=sel64, mlp_str=0.7, head=False)),
        ("head_only", dict(mlp=None, mlp_str=0.0, head=True, head_str=0.7)),
        ("synergy_half_half", dict(mlp=sel32, mlp_str=0.35, head=True, head_str=0.35)),
        ("synergy_full", dict(mlp=sel64, mlp_str=0.7, head=True, head_str=0.7)),
    ]
    for name, cfg in arms:
        m = copy.deepcopy(sft)
        if cfg.get("mlp"):
            rnd_mod.apply_neuron_subset_edit(m, cfg["mlp"], u, strength=cfg["mlp_str"])
        if cfg.get("head"):
            head_suppress(m, 29, 1, cfg.get("head_str", 0.7))
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{name}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        rec = {"arm": name, "mlp_str": cfg.get("mlp_str"), "head": cfg.get("head"), "head_str": cfg.get("head_str", 0.0)}
        rec.update(ev)
        results.append(rec)
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "baseline": base_ev, "arms": results}
    (out/"synergy_summary.json").write_text(json.dumps(summary, indent=2))
    best = max(results, key=lambda r: r["exact_compliance_rate"])
    print(f"[Exp 28 DONE] best={best['arm']} exact={best['exact_compliance_rate']:.1f}% in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
