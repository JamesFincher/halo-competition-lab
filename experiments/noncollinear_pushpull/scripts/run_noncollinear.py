#!/usr/bin/env python3
"""Experiment 38: True Non-Collinear Push-Pull (exp23 corrected).
Analyzer catch: ref/call unembed directions are exact negatives with sign-invariant projections.
True push-pull needs non-collinear directions: refusal direction (ref-vs-call) vs JSON-syntax direction (well-formed action token vs malformed)."""
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
    out = ROOT / "experiments/noncollinear_pushpull/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 38] True Non-Collinear Push-Pull on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device)
    scores, u_ref = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    lm_w = sft.lm_head.weight.detach().float()
    # non-collinear amplify direction: difference between action-token embeddings 'call_tool' vs 'refuse' full tokens
    t_call_full = tok.encode("call_tool", add_special_tokens=False)[0]
    t_ref_full = tok.encode("refuse", add_special_tokens=False)[0]
    u_amp = lm_w[t_call_full] - lm_w[t_ref_full]
    u_amp = u_amp / (u_amp.norm() + 1e-8)
    cos = float(torch.dot(u_ref.cpu(), u_amp.cpu()).item())
    print(f"  direction cosine (collinearity check): {cos:.3f} — vs 1.0 for degenerate exp23", flush=True)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    results = []
    for arm, sup, amp in [("suppress_only", 0.7, 0.0), ("pushpull_0.3", 0.7, 0.3), ("pushpull_0.7", 0.7, 0.7)]:
        m = copy.deepcopy(sft)
        with torch.no_grad():
            for li, idxs in sel.items():
                w = m.model.layers[li].mlp.down_proj.weight
                wf = w.float()
                ur = u_ref.to(w.device); ua = u_amp.to(w.device)
                cols = wf[:, idxs]
                proj_r = torch.outer(ur, ur @ cols)
                proj_a = torch.outer(ua, ua @ cols)
                wf[:, idxs] = cols - sup * proj_r + amp * proj_a
                w.copy_(wf.to(w.dtype))
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{arm:14s}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results.append({"arm": arm, "suppress": sup, "amplify": amp, "direction_cosine": round(cos, 4), **ev})
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "inspiration": "analyzer correction of exp23 degenerate collinearity", "direction_cosine": round(cos, 4), "baseline": base_ev, "arms": results}
    (out/"noncollinear_pushpull_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 38 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
