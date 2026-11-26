#!/usr/bin/env python3
"""Experiment 41: Original-Operator GPU Replication — THE validation test.

Exp39 proved: unembed directions are anti-safety at every strength (cosine 0.004 vs behavioral
directions), crude variance directions fail too, and the dev-gate correctly rejects destructive
edits. Remaining question: does the ORIGINAL pipeline (contrastive activation sniff ->
per-layer resid false-dirs -> gated strength) reproduce its 80%+ exact compliance on GPU?
This runs the original exp12/17 operator verbatim (imported, not reimplemented) on H100.
If it replicates: validated base operator for all future novel methods. If not: the original
CPU-era results themselves are suspect — equally important to know.
"""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/head_and_node_sniff_rl/scripts"))
import run_micro_targeted_rl as base_rl
import run_head_and_node_sniff as hn_sniff

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "experiments/original_operator_replication/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 41] Original-operator replication on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = ben[-5:] + uns[-5:]
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}% unsafe={base_ev['unsafe_refusal_rate']:.1f}% benignfr={base_ev['benign_false_refusal_rate']:.1f}%", flush=True)

    # original sniff pipeline verbatim: contrastive activation measurement
    print("  sniffing (contrastive activation measurement, original pipeline)...", flush=True)
    prof = hn_sniff.sniff_heads_and_false_nodes(sft, tok, train_rows, device, n_pairs=8, safety_penalty=0.45)
    layer_rank = [r["layer"] for r in prof["layer_rank"]]
    # original bake-off configs from exp17: top2 solo vs complementary-6
    arms = {}
    arms["top2_gated"] = {"layers": sorted(layer_rank[:2]), "top_nodes": 64}
    arms["comp6_gated"] = {"layers": sorted(layer_rank[2:8]), "top_nodes": 64}
    results = []
    for name, cfg in arms.items():
        fn_by_layer = {li: list(prof["false_nodes_by_layer"][li]) for li in cfg["layers"] if li in prof["false_nodes_by_layer"]}
        if not fn_by_layer:
            print(f"  [{name}] no false nodes found in layers {cfg['layers']} — skipping", flush=True)
            continue
        use_layers = sorted(fn_by_layer.keys())
        def builder(cand, st, uls=use_layers, fnl=fn_by_layer):
            info = hn_sniff.apply_false_node_mlp_edit(cand, uls, fnl, prof["resid_false_dirs"], cfg["top_nodes"], st)
            return {"strength": st, **info}
        best_m, meta = hn_sniff.double_check_gate(sft, tok, dev_rows, device, builder, name, strengths=(0.45, 0.70, 0.90))
        ev = base_rl.evaluate_heldout(best_m, tok, eval_rows, device)
        print(f"  [{name:12s}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}% accepted={meta.get('accepted')} st={meta.get('strength')}", flush=True)
        results.append({"arm": name, "layers": use_layers, "accepted": meta.get("accepted"), "gate_strength": meta.get("strength"), **ev})
        del best_m

    summary = {"device": str(device), "wall_sec": round(time.time()-t0, 2), "baseline": base_ev,
               "arms": results, "layer_rank_top8": layer_rank[:8],
               "verdict_question": "does the original contrastive-sniff + gated operator replicate 80%+ exact on GPU?"}
    (out/"replication_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 41 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
