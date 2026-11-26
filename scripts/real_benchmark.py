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

def run_benchmark(model, tok, device, n: int = 100) -> dict:
    """Score an already-loaded causal LM on real MMLU/ARC-Easy/HellaSwag items."""
    import os
    if os.environ.get("HALO_SMOKE_TEST") == "1":
        n = min(n, 2)
    from datasets import load_dataset
    res = {}
    try:
        mmlu = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
        items = [{"context": r["question"], "choices": r["choices"], "answer": r["choices"][r["answer"]]} for r in mmlu]
        res["mmlu"] = round(score_mc(model, tok, device, items), 2)
        print(f"  MMLU: {res['mmlu']}%", flush=True)
    except Exception as e:
        print(f"  MMLU failed: {str(e)[:80]}", flush=True)
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{n}]")
        items = [{"context": r["question"], "choices": r["choices"]["text"], "answer": r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])]} for r in arc]
        res["arc_easy"] = round(score_mc(model, tok, device, items), 2)
        print(f"  ARC-Easy: {res['arc_easy']}%", flush=True)
    except Exception as e:
        print(f"  ARC failed: {str(e)[:80]}", flush=True)
    try:
        hs = load_dataset("Rowan/hellaswag", split=f"validation[:{n}]")
        items = [{"context": r["ctx"], "choices": r["endings"], "answer": r["endings"][int(r["label"])]} for r in hs]
        res["hellaswag"] = round(score_mc(model, tok, device, items), 2)
        print(f"  HellaSwag: {res['hellaswag']}%", flush=True)
    except Exception as e:
        print(f"  HellaSwag failed: {str(e)[:80]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/real_benchmark_baseline.json")
    ap.add_argument("--edit", action="store_true")
    import sys
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RealBench] MMLU+ARC+HellaSwag on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device).eval()
    if args.edit:
        sys.path.insert(0, "experiments/common")
        sys.path.insert(0, "experiments/edit_then_target_rl/scripts")
        import importlib, glob
        # apply validated residual-direction edit inline: behavioral contrastive dirs + gated strength
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, "checkpoints/smollm2-135m-policy-lora/checkpoint-36").merge_and_unload().to(device).eval()
        print("  applying validated surgical edit before scoring...", flush=True)
        import run_micro_targeted_rl as base_rl
        train_rows = base_rl.load_rows(Path("data/policy_train.jsonl"))
        ben = [r for r in train_rows if r["target"]["action"] == "call_tool"][:8]
        uns = [r for r in train_rows if r["target"]["action"] == "refuse"][:8]
        store, hooks = {}, []
        for li in (29, 28):
            def mk(idx):
                def hk(_m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    store[idx] = o[:, -1, :].detach().float()
                return hk
            hooks.append(model.model.layers[li].register_forward_hook(mk(li)))
        acts = {li: {"ben": [], "uns": []} for li in (29, 28)}
        with torch.no_grad():
            for r in ben + uns:
                ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
                model(**ids)
                key = "ben" if r in ben else "uns"
                for li in (29, 28): acts[li][key].append(store[li].cpu().clone())
        for h in hooks: h.remove()
        sys.path.insert(0, "experiments/random_node_control_proof/scripts")
        import run_random_control_proof as rnd_mod
        scores, _ = rnd_mod.compute_dla_scores(model, tok, train_rows, device)
        with torch.no_grad():
            for li in (29, 28):
                mb = torch.cat(acts[li]["ben"], 0).mean(0)
                mu = torch.cat(acts[li]["uns"], 0).mean(0)
                d = (mu - mb); u = (d / (d.norm() + 1e-8)).to(device, torch.float32)
                idxs = torch.topk(scores[li], k=64).indices.tolist()
                w = model.model.layers[li].mlp.down_proj.weight
                wf = w.float()
                idx = torch.tensor(idxs, dtype=torch.long, device=w.device)
                proj = torch.matmul(u, wf[:, idx])
                wf[:, idx] -= 0.7 * torch.outer(u, proj)
                w.copy_(wf.to(w.dtype))
        print("  edit applied", flush=True)
    res = run_benchmark(model, tok, device, args.n)
    res["wall_sec"] = round(time.time()-t0, 2)
    res["device"] = str(device)
    res["n_per_bench"] = args.n
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[RealBench DONE] {res}", flush=True)


if __name__ == "__main__":
    main()
