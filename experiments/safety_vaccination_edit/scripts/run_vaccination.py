#!/usr/bin/env python3
"""Experiment 36: Safety Vaccination Before Compliance Amplification.
Inspired by research/SUBAGENT3 idea #4: first amplify the true-safety refusal direction, THEN aggressively suppress false refusal — tests whether vaccination expands the safe editing envelope."""
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
    out = ROOT / "experiments/safety_vaccination_edit/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 36] Safety Vaccination + Aggressive Edit on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in (29, 28)}
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)

    # vaccination: amplify refusal direction on safety-critical neurons in layer 27 (true safety hub incl. L27#110)
    vacc_neurons = {27: [110]}
    results = []
    for arm, vacc_str, edit_str in [
        ("no_vaccine_edit0.7", 0.0, 0.7),
        ("no_vaccine_edit1.2", 0.0, 1.2),
        ("vaccine0.5_edit1.2", 0.5, 1.2),
        ("vaccine1.0_edit1.2", 1.0, 1.2),
    ]:
        m = copy.deepcopy(sft)
        with torch.no_grad():
            if vacc_str > 0:
                for li, idxs in vacc_neurons.items():
                    w = m.model.layers[li].mlp.down_proj.weight
                    uu = u.to(w.device, torch.float32)
                    wf = w.float()
                    cols = wf[:, idxs]
                    proj = torch.outer(uu, uu @ cols)
                    wf[:, idxs] = cols + vacc_str * proj  # amplify refusal direction on safety hub
                    w.copy_(wf.to(w.dtype))
            rnd_mod.apply_neuron_subset_edit(m, sel, u, strength=edit_str)
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        print(f"  [{arm:22s}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        results.append({"arm": arm, "vaccine_strength": vacc_str, "edit_strength": edit_str, **ev})
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "inspiration": "research/SUBAGENT3 idea #4 safety vaccination", "baseline": base_ev, "arms": results}
    (out/"vaccination_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 36 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
