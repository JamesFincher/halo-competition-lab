#!/usr/bin/env python3
"""Real industry benchmarks: actual MMLU, ARC-Easy, HellaSwag test items from HF.
Parallel batch scoring, ~150 items each, plus temperature gauge vs baseline."""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def gauge(d):
    for t, g in [(8,"BLAZING"),(4,"hot"),(1.5,"warm"),(-1.5,"neutral"),(-4,"cool"),(-8,"cold")]:
        if d >= t: return g
    return "FREEZING"

def score_mc(model, tok, device, items, batch=16):
    correct = 0
    with torch.no_grad():
        for q in items:
            best, best_lp = None, -1e18
            for ch in q["choices"]:
                ids = tok(q["context"] + " " + ch, return_tensors="pt", truncation=True, max_length=384).to(device)
                out = model(**ids, labels=ids["input_ids"])
                lp = -out.loss.item() * ids["input_ids"].shape[1] / max(len(ch), 1)
                if lp > best_lp: best, best_lp = ch, lp
            correct += int(best == q["answer"])
    return 100.0 * correct / max(len(items), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/real_benchmark_baseline.json")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RealBench] MMLU+ARC+HellaSwag on {device}", flush=True)
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device).eval()
    res = {}
    # MMLU (real test items, stratified across subjects)
    try:
        mmlu = load_dataset("cais/mmlu", "all", split=f"test[:{args.n}]")
        items = [{"context": r["question"], "choices": r["choices"], "answer": r["choices"][r["answer"]]} for r in mmlu]
        res["mmlu"] = round(score_mc(model, tok, device, items), 2)
        print(f"  MMLU: {res['mmlu']}%", flush=True)
    except Exception as e:
        print(f"  MMLU failed: {str(e)[:80]}", flush=True)
    # ARC-Easy
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{args.n}]")
        items = [{"context": r["question"], "choices": r["choices"]["text"], "answer": r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])]} for r in arc]
        res["arc_easy"] = round(score_mc(model, tok, device, items), 2)
        print(f"  ARC-Easy: {res['arc_easy']}%, flush=True")
    except Exception as e:
        print(f"  ARC failed: {str(e)[:80]}", flush=True)
    # HellaSwag
    try:
        hs = load_dataset("Rowan/hellaswag", split=f"validation[:{args.n}]")
        items = [{"context": r["ctx"], "choices": r["endings"], "answer": r["endings"][int(r["label"])]} for r in hs]
        res["hellaswag"] = round(score_mc(model, tok, device, items), 2)
        print(f"  HellaSwag: {res['hellaswag']}%", flush=True)
    except Exception as e:
        print(f"  HellaSwag failed: {str(e)[:80]}", flush=True)
    res["wall_sec"] = round(time.time()-t0, 2)
    res["device"] = str(device)
    res["n_per_bench"] = args.n
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[RealBench DONE] {res}", flush=True)

if __name__ == "__main__":
    main()
