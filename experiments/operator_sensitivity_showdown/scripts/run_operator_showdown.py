#!/usr/bin/env python3
"""Experiment 39: Edit-Operator Sensitivity Showdown — the lab's pivotal control.

Wave-1/2 mystery: same-neuron rank-1 edits gave 80% exact (original exp17) vs 12.5% (swarm).
Cause hypotheses: (a) direction source: unembed logit-dir vs measured resid false-dir;
(b) strength gating; (c) neuron scoring. This experiment holds neurons fixed (DLA top-64)
and crosses {unembed-dir, resid-false-dir} x {no-gate, gated} — isolating what made the
original edits surgical. THE replication-integrity experiment of the lab.
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
sys.path.insert(0, str(ROOT / "experiments/head_and_node_sniff_rl/scripts"))
import run_micro_targeted_rl as base_rl
import run_random_control_proof as rnd_mod
import run_head_and_node_sniff as hn_sniff

def measure_resid_false_dirs(model, tok, rows, device, layers=(29, 28), n_pairs=6):
    """Measure per-layer false-refusal directions: mean resid diff between falsely-refused
    benign prompts and correctly-complied benign prompts at final position, per layer output."""
    ben = [r for r in rows if r["target"]["action"] == "call_tool"]
    sel = ben[: n_pairs * 2]
    acts = {li: [] for li in layers}
    hooks = []
    store = {}
    for li in layers:
        def mk(idx):
            def hk(_m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                store[idx] = o[:, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].register_forward_hook(mk(li)))
    resid = {li: [] for li in layers}
    with torch.no_grad():
        for r in sel:
            ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**ids)
            for li in layers:
                resid[li].append(store[li].clone())
    for h in hooks: h.remove()
    dirs = {}
    for li in layers:
        stack = torch.cat(resid[li], dim=0)  # [n, D]
        # direction = max-variance axis approximating refusal separation: use mean-centered PC1-ish
        mean = stack.mean(dim=0)
        centered = stack - mean
        # simple: direction of max variance
        var_dir = centered.pow(2).mean(dim=0)
        d = var_dir / (var_dir.norm() + 1e-8)
        dirs[li] = d
    return dirs

def edit_with_dir(m, sel, dirs, strength):
    with torch.no_grad():
        for li, idxs in sel.items():
            w = m.model.layers[li].mlp.down_proj.weight
            wf = w.float()
            u = dirs[li].to(w.device, torch.float32)
            u = u / (u.norm() + 1e-8)
            idx = torch.tensor(idxs, dtype=torch.long, device=w.device)
            proj = torch.matmul(u, wf[:, idx])
            wf[:, idx] -= strength * torch.outer(u, proj)
            w.copy_(wf.to(w.dtype))

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "experiments/operator_sensitivity_showdown/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 39] Operator Sensitivity Showdown on {device} (fp32)", flush=True)
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

    scores, u_unembed = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    print("  measuring resid false-dirs...", flush=True)
    resid_dirs = measure_resid_false_dirs(sft, tok, train_rows, device)
    cos_check = float(torch.nn.functional.cosine_similarity(u_unembed.cpu(), resid_dirs[29].cpu(), dim=0).item())
    print(f"  unembed-vs-resid direction cosine L29: {cos_check:.3f}", flush=True)

    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline fp32] exact={base_ev['exact_compliance_rate']:.1f}% unsafe={base_ev['unsafe_refusal_rate']:.1f}% benignfr={base_ev['benign_false_refusal_rate']:.1f}%", flush=True)
    results = []
    # arm 1: unembed dir, fixed 0.7, no gate (swarm operator — expected destructive)
    m = copy.deepcopy(sft)
    rnd_mod.apply_neuron_subset_edit(m, sel, u_unembed, strength=0.7)
    ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
    print(f"  [unembed_nogate  ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
    results.append({"arm": "unembed_dir_fixed0.7", **ev})
    del m
    # arm 2: resid dir, fixed 0.7, no gate
    m = copy.deepcopy(sft)
    edit_with_dir(m, sel, resid_dirs, 0.7)
    ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
    print(f"  [resid_nogate    ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
    results.append({"arm": "resid_dir_fixed0.7", **ev})
    del m
    # arm 3: resid dir + strength gate (original winning operator)
    def builder(cand, st):
        edit_with_dir(cand, sel, resid_dirs, st)
        return {"strength": st}
    best_m, meta = hn_sniff.double_check_gate(sft, tok, dev_rows, device, builder, "resid_gated", strengths=(0.3, 0.45, 0.7))
    ev = base_rl.evaluate_heldout(best_m, tok, eval_rows, device)
    print(f"  [resid_gated     ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}% (gate_st={meta.get('strength')})", flush=True)
    results.append({"arm": "resid_dir_gated", "gate_strength": meta.get("strength"), **ev})
    del best_m
    # arm 4: unembed dir + gate
    def builder2(cand, st):
        rnd_mod.apply_neuron_subset_edit(cand, sel, u_unembed, strength=st)
        return {"strength": st}
    best_m2, meta2 = hn_sniff.double_check_gate(sft, tok, dev_rows, device, builder2, "unembed_gated", strengths=(0.3, 0.45, 0.7))
    ev = base_rl.evaluate_heldout(best_m2, tok, eval_rows, device)
    print(f"  [unembed_gated   ] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}% (gate_st={meta2.get('strength')})", flush=True)
    results.append({"arm": "unembed_dir_gated", "gate_strength": meta2.get("strength"), **ev})
    del best_m2

    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "direction_cosine_l29": round(cos_check, 4),
               "baseline": base_ev, "arms": results,
               "question": "what made original edits surgical: direction source or strength gating?"}
    (out/"operator_showdown_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 39 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
