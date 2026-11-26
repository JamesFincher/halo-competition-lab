#!/usr/bin/env python3
"""Benchmark Failure Tree — reverse-engineer which weights produce wrong answers.

Novel interpretability fork: run real MMLU/ARC items, split into correct vs incorrect,
christmas-tree probe the per-layer per-neuron activation differences (failure signature),
then surgically suppress failure-correlated neurons / amplify success-correlated ones
and measure the benchmark delta. This is direct failure-mode reverse engineering:
which weights contribute to bad outputs, answered causally by edit-and-measure.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]

def mc_items(n=60):
    from datasets import load_dataset
    out = []
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{n}]")
        for r in arc:
            out.append({"context": r["question"], "choices": r["choices"]["text"],
                        "answer": r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])], "src": "arc"})
    except Exception as e:
        print(f"arc load fail {e}", flush=True)
    return out

def score_items(model, tok, device, items):
    right, wrong = [], []
    with torch.no_grad():
        for q in items:
            best, best_lp = None, -1e18
            for ch in q["choices"]:
                ids = tok(q["context"] + " " + ch, return_tensors="pt", truncation=True, max_length=384).to(device)
                o = model(**ids, labels=ids["input_ids"])
                lp = -o.loss.item() * ids["input_ids"].shape[1] / max(len(ch), 1)
                if lp > best_lp: best, best_lp = ch, lp
            (right if best == q["answer"] else wrong).append(q)
    return right, wrong

def probe_activations(model, tok, device, items, layers=(28, 29)):
    acts = {li: [] for li in layers}
    store = {}
    hooks = []
    for li in layers:
        def mk(idx):
            def hk(_m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                store[idx] = o[:, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].register_forward_hook(mk(li)))
    with torch.no_grad():
        for q in items:
            ids = tok(q["context"] + " " + q["answer"], return_tensors="pt", truncation=True, max_length=384).to(device)
            model(**ids)
            for li in layers:
                acts[li].append(store[li].clone())
    for h in hooks: h.remove()
    return {li: torch.cat(acts[li], 0).mean(0) for li in layers}

def probe_mlp(model, tok, device, items, layers=(28, 29)):
    acts = {li: [] for li in layers}
    store = {}
    hooks = []
    for li in layers:
        def mk(idx):
            def hk(_m, inp, out):
                store[idx] = out[:, -1, :].detach().float().cpu()
            return hk
        hooks.append(model.model.layers[li].mlp.act_fn.register_forward_hook(mk(li)) if hasattr(model.model.layers[li].mlp.act_fn, 'register_forward_hook') else model.model.layers[li].mlp.register_forward_hook(lambda _m, inp, out, idx=li: store.__setitem__(idx, inp[0][:, -1, :].detach().float().cpu())))
    with torch.no_grad():
        for q in items[:20]:
            ids = tok(q["context"] + " " + q["answer"], return_tensors="pt", truncation=True, max_length=384).to(device)
            model(**ids)
            for li in layers:
                if li in store: acts[li].append(store[li].clone())
    for h in hooks: h.remove()
    return {li: torch.cat(acts[li], 0).mean(0) for li in layers if acts[li]}

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outd = ROOT / "experiments/benchmark_failure_tree/results"
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[FailureTree] on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device).eval()

    items = mc_items(60)
    print(f"  scoring {len(items)} benchmark items...", flush=True)
    right, wrong = score_items(model, tok, device, items)
    print(f"  correct={len(right)} wrong={len(wrong)} ({100*len(right)/len(items):.0f}% accuracy)", flush=True)

    layers = (28, 29)
    print("  probing failure vs success activation signatures...", flush=True)
    right_acts = probe_activations(model, tok, device, right[:20], layers) if right else {}
    wrong_acts = probe_activations(model, tok, device, wrong[:20], layers) if wrong else {}

    # failure signature: direction in resid space separating wrong from right
    report = {"device": str(device), "accuracy": round(100*len(right)/len(items), 1), "n_right": len(right), "n_wrong": len(wrong)}
    if right and wrong:
        sig = {}
        for li in layers:
            d = wrong_acts[li] - right_acts[li]
            sig[li] = (d / (d.norm() + 1e-8)).to(device)
            report[f"failure_dir_norm_L{li}"] = round(float((wrong_acts[li] - right_acts[li]).norm().item()), 3)
        # reverse-engineer: suppress the failure direction in MLP down_proj output space at final layers
        import copy
        edited = copy.deepcopy(model)
        with torch.no_grad():
            for li in layers:
                u = sig[li].to(torch.float32)
                for mod_name, mod in [("down_proj", edited.model.layers[li].mlp.down_proj)]:
                    w = mod.weight
                    wf = w.float()
                    proj = torch.matmul(u, wf)
                    wf -= 0.3 * torch.outer(u, proj)
                    w.copy_(wf.to(w.dtype))
        right2, wrong2 = score_items(edited, tok, device, items)
        print(f"  after suppressing failure dirs: correct={len(right2)} ({100*len(right2)/len(items):.0f}%) vs before {100*len(right)/len(items):.0f}%", flush=True)
        report["accuracy_after_suppression"] = round(100*len(right2)/len(items), 1)
        report["delta_pp"] = round(100*len(right2)/len(items) - 100*len(right)/len(items), 1)
    report["wall_sec"] = round(time.time()-t0, 2)
    (outd/"failure_tree_summary.json").write_text(json.dumps(report, indent=2))
    print(f"[FailureTree DONE] {json.dumps({k:v for k,v in report.items() if k!='device'})}", flush=True)

if __name__ == "__main__":
    main()
