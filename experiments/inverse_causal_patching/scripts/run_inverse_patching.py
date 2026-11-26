#!/usr/bin/env python3
"""Experiment 27: Inverse Causal Patching Localization.
Patch activations from compliant runs into falsely-refusing runs layer-by-layer to find minimal causal layers, then edit only those."""
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


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # fp32 required: bf16 coarsens rank-1 surgical edits (wave-1 finding)
    out = ROOT / "experiments/inverse_causal_patching/results"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 27] Inverse Causal Patching on {device}", flush=True)
    train_rows, eval_rows = load_data()
    sft, tok = load_sft(device, dtype)
    n_layers = sft.config.num_hidden_layers
    tid_ref = tok.encode("ref", add_special_tokens=False)[0]
    tid_call = tok.encode("call", add_special_tokens=False)[0]
    lm_w = sft.lm_head.weight.detach().float()
    u = (lm_w[tid_ref] - lm_w[tid_call]); u = u / (u.norm() + 1e-8)

    def margin_at_pos(m, prompt):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = m(**ids).logits[0, -1, :].float()
        return float((logits[tid_ref] - logits[tid_call]).item()), ids

    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    pairs = []
    for r in benign[:6]:
        prompt = base_rl.build_prompt(tok, r["messages"])
        marg, ids = margin_at_pos(sft, prompt)
        pairs.append({"prompt": prompt, "margin": marg, "ids": ids})
    pos_rows = sorted(pairs, key=lambda p: p["margin"])
    neg_rows = sorted(pairs, key=lambda p: -p["margin"])
    print(f"  margins across 6 benign prompts: {[round(p['margin'],2) for p in pairs]}", flush=True)

    def capture_resid(m, ids):
        acts = {}
        hooks = []
        for li in range(n_layers):
            def mk(idx):
                def hook(_mod, inp, out):
                    acts[idx] = (out[0] if isinstance(out, tuple) else out)[:, -1, :].detach().clone()
                return hook
            hooks.append(m.model.layers[li].register_forward_hook(mk(li)))
        with torch.no_grad(): m(**ids)
        for h in hooks: h.remove()
        return acts

    results = []
    src = pos_rows[0]
    tgt = pos_rows[-1]
    acts_src = capture_resid(sft, src["ids"])
    base_marg_tgt, tgt_ids = margin_at_pos(sft, tgt["prompt"])
    print(f"  patch pair: src margin={src['margin']:.2f} -> tgt margin={base_marg_tgt:.2f}", flush=True)
    for li in range(n_layers):
        def mkpatch(layer_idx):
            def hook(_mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                o2 = o.clone()
                len_diff = o2.shape[1] - acts_src[layer_idx].shape[0] if acts_src[layer_idx].dim() == 2 else 0
                patch_vec = acts_src[layer_idx]
                if patch_vec.dim() == 2: patch_vec = patch_vec[0]
                o2[:, -1, :] = patch_vec.to(o2.device, o2.dtype).unsqueeze(0) if patch_vec.dim() == 1 else patch_vec.to(o2.device, o2.dtype)
                return (o2,) + tuple(out[1:]) if isinstance(out, tuple) else o2
            return hook
        h = sft.model.layers[li].register_forward_hook(mkpatch(li))
        marg, _ = margin_at_pos(sft, tgt["prompt"])
        h.remove()
        results.append({"layer": li, "margin_after_patch": round(marg, 3), "delta": round(marg - base_marg_tgt, 3)})
    results.sort(key=lambda r: -r["delta"])
    print("  top causal layers by patch delta:", [(r["layer"], r["delta"]) for r in results[:5]], flush=True)
    summary = {"device": str(device), "wall_sec": round(time.time()-t0,2), "base_target_margin": base_marg_tgt, "layer_deltas": results}
    (out/"causal_patching_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 27 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
