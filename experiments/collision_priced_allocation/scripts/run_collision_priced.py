#!/usr/bin/env python3
"""Experiment 37: Collision-Priced Cross-Behavior Neuron Allocation.
Inspired by research/SUBAGENT3 idea #7 + Exp 18/19 cross-talk findings: price each neuron by DLA score minus penalty for overlapping with true-safety circuit, then edit the re-priced top-K."""
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
    out = ROOT / "experiments/collision_priced_allocation/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 37] Collision-Priced Neuron Allocation on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [baseline] exact={base_ev['exact_compliance_rate']:.1f}%", flush=True)

    # collision penalty: neurons most active on UNSAFE prompts (true safety circuit) get penalized
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"][:10]
    def collect_uns(li):
        acc = torch.zeros(sft.config.intermediate_size, device=device); n = 0
        def hk(_m, inp, _o):
            nonlocal acc, n
            acc += inp[0][0, -1, :].detach().float(); n += 1
        h = sft.model.layers[li].mlp.down_proj.register_forward_hook(hk)
        with torch.no_grad():
            for r in uns:
                ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
                sft(**ids)
        h.remove()
        return (acc / max(n, 1)).cpu()

    safety_act = {li: collect_uns(li) for li in (29, 28)}
    results = []
    for arm, penalty in [("pure_dla", 0.0), ("priced_0.5", 0.5), ("priced_1.0", 1.0), ("priced_2.0", 2.0)]:
        selp = {}
        for li in (29, 28):
            sa = safety_act[li]
            sa_norm = sa / (sa.max() + 1e-8)
            priced = scores[li] - penalty * sa_norm * scores[li].max()
            selp[li] = torch.topk(priced, k=64).indices.tolist()
        m = copy.deepcopy(sft)
        rnd_mod.apply_neuron_subset_edit(m, selp, u, strength=0.7)
        ev = base_rl.evaluate_heldout(m, tok, eval_rows, device)
        overlap = len(set(selp[29]) & set(torch.topk(scores[29], k=64).indices.tolist()))
        print(f"  [{arm:12s}] exact={ev['exact_compliance_rate']:.1f}% pol={ev['policy_accuracy']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}% sel_overlap={overlap}/64", flush=True)
        results.append({"arm": arm, "penalty": penalty, "selection_overlap": overlap, **ev})
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "inspiration": "research/SUBAGENT3 idea #7 collision-priced allocation", "baseline": base_ev, "arms": results}
    (out/"collision_priced_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 37 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
