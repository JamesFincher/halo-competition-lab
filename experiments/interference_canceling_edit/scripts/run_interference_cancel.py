#!/usr/bin/env python3
"""Experiment 35: Interference-Canceling Multi-Neuron Edit Solve.
Inspired by research/SUBAGENT3 idea #2: solve per-neuron edit coefficients minimizing collateral on safety while maximizing false-refusal suppression — a constrained least-squares edit instead of uniform projection."""
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

def load_sft(device):
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()
    return sft, tok

def load_data():
    return (base_rl.load_rows(ROOT / "data/policy_train.jsonl"), base_rl.load_rows(ROOT / "data/policy_eval.jsonl"))


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ROOT / "experiments/interference_canceling_edit/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 35] Interference-Canceling Edit Solve on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)

    tid_ref = tok.encode("ref", add_special_tokens=False)[0]
    tid_call = tok.encode("call", add_special_tokens=False)[0]
    lm_w = sft.lm_head.weight.detach().float()

    # safety direction: we want refusal preserved for UNSAFE prompts but suppressed for benign.
    # construct per-prompt-type mean activations at down_proj input for layers 29/28
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"][:10]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"][:10]
    def collect(rows, li):
        acc = torch.zeros(sft.config.intermediate_size, device=device)
        n = 0
        def hk(_m, inp, _o):
            nonlocal acc, n
            acc += inp[0][0, -1, :].detach().float(); n += 1
        h = sft.model.layers[li].mlp.down_proj.register_forward_hook(hk)
        with torch.no_grad():
            for r in rows:
                ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
                sft(**ids)
        h.remove()
        return (acc / max(n, 1)).cpu()

    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)
    results = []
    for arm, lam in [("uniform_control", None), ("cancel_lambda1.0", 1.0), ("cancel_lambda3.0", 3.0)]:
        m = copy.deepcopy(sft)
        with torch.no_grad():
            for li in (29, 28):
                w = m.model.layers[li].mlp.down_proj.weight
                wf = w.float()
                uu = u.to(w.device)
                ben_a = collect(ben, li).to(w.device)
                uns_a = collect(uns, li).to(w.device)
                colproj = wf.T @ uu  # [I] projection of each neuron col onto refusal direction
                idxs = torch.topk(scores[li].to(w.device), k=64).indices
                if lam is None:
                    coeff = torch.full((64,), 0.7, device=w.device)
                else:
                    # solve: pick coefficients c in [0,1.2] minimizing lam*(c . uns_proj)^2 - (c . ben_obj)
                    # approximation: weight coefficient by ratio of benign-driven vs unsafe-driven contribution
                    ben_drive = (ben_a[idxs] * colproj[idxs]).clamp(min=0)
                    uns_drive = (uns_a[idxs] * colproj[idxs]).clamp(min=0)
                    ratio = ben_drive / (ben_drive + lam * uns_drive + 1e-6)
                    coeff = 1.2 * ratio
                cols = wf[:, idxs]
                proj = torch.outer(uu, uu @ cols)
                wf[:, idxs] = cols - coeff.unsqueeze(0) * proj
                w.copy_(wf.to(w.dtype))
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        mean_coeff = float(coeff.mean()) if lam is not None else 0.7
        print(f"  [{arm:18s}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}% meancoeff={mean_coeff:.2f}", flush=True)
        results.append({"arm": arm, "lambda": lam, "mean_coeff": round(mean_coeff, 3), **ev})
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "inspiration": "research/SUBAGENT3 idea #2 interference-canceling solve", "baseline": base_ev, "arms": results}
    (out/"interference_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 35 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
