#!/usr/bin/env python3
"""Experiment 31: Request-Type -> Layer/Neuron Causal Circuit Mapper.

For each request TYPE in the policy data (benign-tool-call vs unsafe-refusal, and semantic
subcategories), maps exactly which layers and neurons causally carry the decision, using:
  1. per-type DLA scores (activation x unembed-direction attribution)
  2. cross-type neuron exclusivity (which neurons are type-specific vs shared)
Output: a targeting atlas JSON consumable by targeted RL methods — for each request type,
the top layers/neurons to edit or freeze.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out = ROOT / "experiments/request_circuit_mapper/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 31] Request-Type Circuit Mapper on {device}", flush=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()

    n_layers = sft.config.num_hidden_layers
    inter = sft.config.intermediate_size
    tid_ref = tok.encode("ref", add_special_tokens=False)[0]
    tid_call = tok.encode("call", add_special_tokens=False)[0]
    lm_w = sft.lm_head.weight.detach().float()
    u = (lm_w[tid_ref] - lm_w[tid_call]); u = u / (u.norm() + 1e-8)

    # classify rows into semantic request types by keyword matching on user content
    def classify(row):
        txt = " ".join(m.get("content", "") for m in row["messages"]).lower()
        action = row["target"]["action"]
        if action == "refuse":
            if any(w in txt for w in ("bomb", "weapon", "explosive", "poison")): return "violence_weapons"
            if any(w in txt for w in ("hack", "password", "malware", "breach", "exploit")): return "cyber_intrusion"
            if any(w in txt for w in ("drug", "meth", "cocaine", "narcotic")): return "drugs"
            return "other_unsafe"
        else:
            if any(w in txt for w in ("weather", "temperature", "forecast")): return "weather_query"
            if any(w in txt for w in ("email", "send", "message", "inbox")): return "comms_task"
            if any(w in txt for w in ("calculate", "math", "sum", "total", "price")): return "math_task"
            if any(w in txt for w in ("search", "find", "look up", "research")): return "search_task"
            return "other_benign"

    typed = {}
    for r in train_rows:
        c = classify(r)
        typed.setdefault(c, []).append(r)
    print(f"  request types: { {k: len(v) for k, v in typed.items()} }", flush=True)

    # per-type DLA: mean activation at down_proj input x unembed refusal-direction projection
    def collect(rows):
        sums = [torch.zeros(inter, device=device) for _ in range(n_layers)]
        cnt = 0
        hooks = []
        acc = [torch.zeros(inter, device=device) for _ in range(n_layers)]
        def mk(idx):
            def hk(_m, inp, _o):
                acc[idx] += inp[0][0, -1, :].detach().float()
            return hk
        for li in range(n_layers):
            hooks.append(sft.model.layers[li].mlp.down_proj.register_forward_hook(mk(li)))
        with torch.no_grad():
            for r in rows[:8]:
                ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
                sft(**ids); cnt += 1
        for h in hooks: h.remove()
        return [a / max(cnt, 1) for a in acc]

    atlas = {}
    all_scores = {}
    for ctype, rows in sorted(typed.items()):
        acts = collect(rows)
        sc = torch.zeros(n_layers, inter)
        for li in range(n_layers):
            w_down = sft.model.layers[li].mlp.down_proj.weight.detach().float()
            colproj = u.to(w_down.device) @ w_down
            sc[li] = (acts[li].cpu() * colproj.cpu()).clamp(min=0.0)
        all_scores[ctype] = sc
        top = []
        flat = sc.flatten()
        topv, topi = torch.topk(flat, k=10)
        for v, i in zip(topv.tolist(), topi.tolist()):
            top.append({"layer": i // inter, "neuron": i % inter, "score": round(v, 3)})
        layer_mass = sc.sum(dim=1)
        com = float((layer_mass * torch.arange(n_layers)).sum() / (layer_mass.sum() + 1e-8))
        atlas[ctype] = {"n_rows": len(rows), "depth_com": round(com, 2), "top10_neurons": top}
        print(f"  [{ctype:18s}] depth_com={com:.1f} top={top[0]['layer']}/{top[0]['neuron']}", flush=True)

    # cross-type exclusivity: for each neuron, max-type score / sum-type score — high = type-specific
    types = sorted(all_scores)
    stack = torch.stack([all_scores[c] for c in types])  # [T, L, I]
    total = stack.sum(dim=0) + 1e-8
    exclusivity = stack.max(dim=0).values / total  # [L, I]
    ex_flat = exclusivity.flatten()
    exv, exi = torch.topk(ex_flat, k=15)
    excl = []
    for v, i in zip(exv.tolist(), exi.tolist()):
        li, ni = i // inter, i % inter
        owner = types[int(stack[:, li, ni].argmax().item())]
        excl.append({"layer": li, "neuron": ni, "exclusivity": round(v, 3), "owner_type": owner})
    print(f"  top exclusive neuron: L{excl[0]['layer']}#{excl[0]['neuron']} owner={excl[0]['owner_type']} excl={excl[0]['exclusivity']}", flush=True)

    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "atlas": atlas, "exclusive_neurons": excl,
               "usage": "targeted RL: edit/freeze per-type top neurons; high-exclusivity neurons are safe single-type targets"}
    (out/"request_circuit_atlas.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 31 DONE] atlas for {len(atlas)} request types in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
