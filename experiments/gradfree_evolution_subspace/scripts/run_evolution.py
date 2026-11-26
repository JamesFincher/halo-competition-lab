#!/usr/bin/env python3
"""Experiment 25: Gradient-Free Evolution in 8D Surgical Subspace.
CMA-ES-style search over (k_l29, k_l28, strength_l29, strength_l28, suppress/amplify mix, layer mix) using dev-split fitness."""
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

import random

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = ROOT / "experiments/gradfree_evolution_subspace/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 25] Gradient-Free Evolution 8D on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    scores, u = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    dev_rows = eval_rows[:20]
    test_rows = eval_rows[20:]

    def decode(x):
        k29 = int(round(min(max(x[0], 8), 256)))
        k28 = int(round(min(max(x[1], 8), 256)))
        s29 = min(max(x[2], 0.1), 1.5)
        s28 = min(max(x[3], 0.1), 1.5)
        amp = min(max(x[4], 0.0), 1.0)
        mix = min(max(x[5], 0.0), 1.0)
        l3 = int(round(min(max(x[6], 20), 29)))
        s3 = min(max(x[7], 0.0), 1.0)
        return dict(k29=k29, k28=k28, s29=s29, s28=s28, amp=amp, mix=mix, l3=l3, s3=s3)

    def apply_params(m, p):
        with torch.no_grad():
            for li, k, s in ((29, p["k29"], p["s29"]), (28, p["k28"], p["s28"])):
                idxs = torch.topk(scores[li], k=k).indices.tolist()
                rnd_mod.apply_neuron_subset_edit(m, {li: idxs}, u, strength=s)
            if p["s3"] > 0.05:
                idxs = torch.topk(scores[p["l3"]], k=32).indices.tolist()
                rnd_mod.apply_neuron_subset_edit(m, {p["l3"]: idxs}, u, strength=p["s3"])

    def fitness(m):
        ev = base_rl.evaluate_heldout(m, tok, dev_rows, device, max_new_tokens=48)
        return ev["exact_compliance_rate"] + 0.5 * ev["unsafe_refusal_rate"] - 0.5 * ev["benign_false_refusal_rate"]

    rng = random.Random(7)
    center = [64, 64, 0.7, 0.7, 0.0, 0.5, 27, 0.0]
    sigma = [32, 32, 0.2, 0.2, 0.2, 0.2, 3, 0.2]
    pop_size, gens = 6, 4
    hist = []
    for gen in range(gens):
        pop = [center] + [[c + rng.gauss(0, s) for c, s in zip(center, sigma)] for _ in range(pop_size - 1)]
        scored = []
        for x in pop:
            p = decode(x)
            m = copy.deepcopy(sft)
            apply_params(m, p)
            f = fitness(m)
            scored.append((f, x, p))
            print(f"  [gen{gen}] fit={f:.1f} k29={p['k29']} s29={p['s29']:.2f} k28={p['k28']} s28={p['s28']:.2f} l3={p['l3']} s3={p['s3']:.2f}", flush=True)
            del m
            if device.type == "cuda": torch.cuda.empty_cache()
        scored.sort(key=lambda t: -t[0])
        elites = scored[:2]
        center = [sum(e[1][i] for e in elites) / len(elites) for i in range(8)]
        hist.append({"gen": gen, "best_fit": scored[0][0], "best_params": scored[0][2]})
    best_p = scored[0][2]
    m = copy.deepcopy(sft)
    apply_params(m, best_p)
    final_dev = base_rl.evaluate_heldout(m, tok, dev_rows, device)
    final_test = base_rl.evaluate_heldout(m, tok, test_rows, device)
    print(f"  [evolved] dev exact={final_dev['exact_compliance_rate']:.1f}% | heldout-test exact={final_test['exact_compliance_rate']:.1f}% unsafe={final_test['unsafe_refusal_rate']:.1f}% benignfr={final_test['benign_false_refusal_rate']:.1f}%", flush=True)
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "best_params": best_p, "dev": final_dev, "test": final_test, "history": hist}
    (out/"evolution_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 25 DONE] test exact={final_test['exact_compliance_rate']:.1f}% in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
