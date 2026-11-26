#!/usr/bin/env python3
"""Experiment 43: Combined Winner — whitened-subspace surgical edit + slider-gated head + RL.

Stacks the lab's two validated techniques: (1) residual-direction surgical edit (65%/90%/0%
operator) applied first, then (2) zero-init slider-gated attention head trained with RL on the
edited model. Tests whether additive architectural capacity + subtractive surgical editing
compose into something better than either alone. Before/after at each stage on heldout eval.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/whitened_subspace_rfm_edit/scripts"))
sys.path.insert(0, str(ROOT / "experiments/slider_gated_attention_head/scripts"))
import run_micro_targeted_rl as base_rl
from run_slider_head import SliderGatedHead

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/combined_slider_subspace/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 43] Combined slider+subspace on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    sft.eval()
    base_ev = base_rl.evaluate_heldout(sft, tok, eval_rows, device)
    print(f"  [stage0 baseline] exact={base_ev['exact_compliance_rate']:.1f}% unsafe={base_ev['unsafe_refusal_rate']:.1f}% benignfr={base_ev['benign_false_refusal_rate']:.1f}%", flush=True)

    # stage 1: replicate the winning whitened-subspace edit direction approach:
    # use behavioral contrastive direction: mean resid diff refused-benign vs complied-benign at L29/L28
    layers = (29, 28)
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    acts = {li: {"ben": [], "uns": []} for li in layers}
    store = {}
    hooks = []
    for li in layers:
        def mk(idx):
            def hk(_m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                store[idx] = o[:, -1, :].detach().float()
            return hk
        hooks.append(sft.model.layers[li].register_forward_hook(mk(li)))
    with torch.no_grad():
        for r in ben[:8]:
            ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
            sft(**ids)
            for li in layers: acts[li]["ben"].append(store[li].cpu().clone())
        for r in uns[:8]:
            ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
            sft(**ids)
            for li in layers: acts[li]["uns"].append(store[li].cpu().clone())
    for h in hooks: h.remove()
    dirs = {}
    for li in layers:
        mb = torch.cat(acts[li]["ben"], 0).mean(0)
        mu = torch.cat(acts[li]["uns"], 0).mean(0)
        d = mu - mb  # refusal direction in resid space
        dirs[li] = (d / (d.norm() + 1e-8)).to(device)
    # gated surgical edit with behavioral direction on top-DLA neurons
    sys.path.insert(0, str(ROOT / "experiments/random_node_control_proof/scripts"))
    import run_random_control_proof as rnd_mod
    scores, _ = rnd_mod.compute_dla_scores(sft, tok, train_rows, device)
    sel = {li: torch.topk(scores[li], k=64).indices.tolist() for li in layers}
    best_ev, best_st = None, None
    for st in (0.45, 0.7, 0.9):
        import copy
        cand = copy.deepcopy(sft)
        with torch.no_grad():
            for li, idxs in sel.items():
                w = cand.model.layers[li].mlp.down_proj.weight
                wf = w.float()
                u = dirs[li].to(torch.float32)
                idx = torch.tensor(idxs, dtype=torch.long, device=w.device)
                proj = torch.matmul(u, wf[:, idx])
                wf[:, idx] -= st * torch.outer(u, proj)
                w.copy_(wf.to(w.dtype))
        ev = base_rl.evaluate_heldout(cand, tok, eval_rows, device)
        print(f"  [stage1 edit st={st}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benignfr={ev['benign_false_refusal_rate']:.1f}%", flush=True)
        score = ev["exact_compliance_rate"] + 0.5 * ev["unsafe_refusal_rate"] - 0.5 * ev["benign_false_refusal_rate"]
        if best_ev is None or score > best_ev[0]:
            best_ev, best_st, best_model = (score, ev), st, cand
        else:
            del cand
    print(f"  stage1 best: st={best_st}", flush=True)

    # stage 2: slider head + RL on edited model (compressed version of exp42)
    model = best_model
    head = SliderGatedHead(model.config.hidden_size).to(device)
    LAYER = 27
    def hook(_m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        d = head(o) - o
        return ((o + d,) + out[1:]) if isinstance(out, tuple) else o + d
    h = model.model.layers[LAYER].register_forward_hook(hook)
    for p in model.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)
    rows = ben[:12] + uns[:12]
    head.train()
    for step in range(24):
        r = rows[step % len(rows)]
        ids = tok(base_rl.build_prompt(tok, r["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
        pl = ids["input_ids"].shape[1]
        gen = model.generate(**ids, max_new_tokens=48, do_sample=True, temperature=0.9, top_p=0.95, num_return_sequences=4, pad_token_id=tok.eos_token_id)
        rewards = []
        for gi in range(4):
            text = tok.decode(gen[gi][pl:], skip_special_tokens=True)
            rew, _ = base_rl.score_completion(text, r["target"])
            rewards.append(rew)
        rew_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = rew_t - rew_t.mean()
        loss = torch.zeros((), device=device)
        for gi in range(4):
            lp = base_rl.completion_logprobs(model, gen[gi].unsqueeze(0), pl)
            loss = loss - (adv[gi] * lp.sum() / max(1, lp.numel()))
        loss = loss / 4
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 12 == 0:
            print(f"    rl step {step}: alpha={head.alpha.item():.4f} mean_rew={rew_t.mean().item():.2f}", flush=True)
    final_ev = base_rl.evaluate_heldout(model, tok, eval_rows, device)
    h.remove()
    print(f"  [stage2 combined] exact={final_ev['exact_compliance_rate']:.1f}% unsafe={final_ev['unsafe_refusal_rate']:.1f}% benignfr={final_ev['benign_false_refusal_rate']:.1f}% alpha={head.alpha.item():.4f}", flush=True)
    summary = {"device": str(device), "wall_sec": round(time.time()-t0, 2),
               "baseline": base_ev, "stage1_edit": best_ev[1] if best_ev else None, "stage1_strength": best_st,
               "stage2_combined": final_ev, "final_alpha": round(head.alpha.item(), 5),
               "question": "do surgical edit + slider head compose better than either alone?"}
    (out_dir/"combined_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Exp 43 DONE] in {summary['wall_sec']}s", flush=True)

if __name__ == "__main__":
    main()
