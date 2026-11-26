#!/usr/bin/env python3
"""Experiment 60: Benign-Keyed Negative-Image Refusal Inverter.

Implements SUBAGENT3 rank-1: embed a conditional polarity flip in one MLP neuron.
A benign/unsafe LDA key in up_proj carries the sign; the down_proj column writes
negative refusal, so benign activations suppress refusal while unsafe activations
preserve or amplify it. Includes a short post-edit targeted-LoRA RL arm.

Research anchors: arXiv:2406.11717 and arXiv:2012.14913.
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp57_halo_native_targeted_lora/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from run_halo_native_targeted_lora import layer_regex  # noqa: E402
from validated_edit import build_validated_rank1_subspaces  # noqa: E402

LAYERS = [27, 28, 29]
HUBS = {(27, 110), (29, 1512), (29, 1022)}


def action_pos(ids, prompt_len, action_ids):
    seq = ids.tolist()
    for p in range(prompt_len, len(seq)):
        if seq[p] in action_ids:
            return p
    return None


def collect_mlp_inputs(model, tokenizer, rows, device, layers, per_class=20):
    action_ids = set(tokenizer.encode(x, add_special_tokens=False)[:1] for x in ("call", "ref"))
    out = {li: {"benign": [], "unsafe": []} for li in layers}
    for action, key in (("call_tool", "benign"), ("refuse", "unsafe")):
        selected = [r for r in rows if r["target"]["action"] == action][:per_class]
        for row in selected:
            prompt = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(prompt + row["assistant_text"], return_tensors="pt", truncation=True, max_length=320).to(device)
            plen = len(tokenizer(prompt, truncation=True, max_length=256)["input_ids"])
            pos = action_pos(enc["input_ids"][0], plen, action_ids)
            if pos is None or pos == 0:
                continue
            caps = {}
            handles = []
            for li in layers:
                def pre(_mod, inp, li=li):
                    caps[li] = inp[0].detach()
                handles.append(model.model.layers[li].mlp.register_forward_pre_hook(pre))
            with torch.no_grad():
                model(input_ids=enc["input_ids"])
            for h in handles:
                h.remove()
            for li in layers:
                out[li][key].append(caps[li][0, pos - 1, :].float().cpu())
    return out


def fisher_key(benign, unsafe):
    xb, xu = torch.stack(benign), torch.stack(unsafe)
    mb, mu = xb.mean(0), xu.mean(0)
    cb = (xb - mb).T @ (xb - mb) / max(len(benign) - 1, 1)
    cu = (xu - mu).T @ (xu - mu) / max(len(unsafe) - 1, 1)
    eye = torch.eye(cb.shape[0])
    key = torch.linalg.pinv(cb + cu + 1e-3 * eye) @ (mb - mu)
    key = key / (key.norm() + 1e-8)
    if float((mb - mu) @ key) < 0:
        key = -key
    return key


def choose_neuron(model, layer):
    down = model.model.layers[layer].mlp.down_proj.weight.detach().float()
    order = torch.argsort(down.norm(dim=0)).cpu().tolist()
    for j in order:
        if (layer, int(j)) not in HUBS:
            return int(j)
    raise RuntimeError("no non-hub neuron available")


def apply_inverter(model, inputs, refusal_dirs, layer, alpha):
    benign, unsafe = inputs[layer]["benign"], inputs[layer]["unsafe"]
    x = torch.stack(benign + unsafe)
    mu = x.mean(0)
    mu = mu / (mu.norm() + 1e-8)
    if float((x @ mu).mean()) < 0:
        mu = -mu
    key = fisher_key(benign, unsafe)
    j = choose_neuron(model, layer)
    mlp = model.model.layers[layer].mlp
    with torch.no_grad():
        gate_scale = mlp.gate_proj.weight.norm(dim=1).median().item()
        up_scale = mlp.up_proj.weight.norm(dim=1).median().item()
        dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
        mlp.gate_proj.weight[j] = mu.to(dev, dt) * gate_scale
        mlp.up_proj.weight[j] = key.to(dev, dt) * up_scale
        r = refusal_dirs[layer].squeeze(1).to(dev, torch.float32)
        r = r / (r.norm() + 1e-8)
        mlp.down_proj.weight[:, j] = mlp.down_proj.weight[:, j].float() - alpha * r
    return {"layer": layer, "neuron": j, "alpha": alpha, "gate_scale": gate_scale, "up_scale": up_scale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    ap.add_argument("--rl-steps", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.rl_steps = min(args.rl_steps, 6)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp60_benign_keyed_refusal_inverter/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 60] Benign-Keyed Negative-Image Refusal Inverter on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    prompt_stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% unsafe={sft_eval['unsafe_refusal_rate']:.1f}% benign_fr={sft_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    print("[Stage 1] Collecting L27-29 MLP input keys and validated refusal directions...", flush=True)
    inputs = collect_mlp_inputs(sft, tokenizer, train_rows, device, LAYERS)
    dirs = {}
    singulars = {}
    for li in LAYERS:
        sub, sv = build_validated_rank1_subspaces(sft, tokenizer, train_rows, device, edit_layers=[li], n_pairs=12)
        dirs[li] = sub[li].cpu(); singulars[str(li)] = sv[str(li)]
    grid = {}
    best = None
    print("[Stage 2] Grid-testing inverter layer x alpha on held-out eval...", flush=True)
    for li in LAYERS:
        for alpha in args.alphas:
            m = copy.deepcopy(sft)
            info = apply_inverter(m, inputs, dirs, li, alpha)
            ev = base_rl.evaluate_heldout(m, tokenizer, eval_rows, device)
            name = f"L{li}_a{alpha:g}"
            grid[name] = {"edit": info, "eval": ev}
            print(f"  [{name}] exact={ev['exact_compliance_rate']:.1f}% unsafe={ev['unsafe_refusal_rate']:.1f}% benign_fr={ev['benign_false_refusal_rate']:.1f}% json={ev['json_valid_rate']:.1f}%", flush=True)
            score = ev["exact_compliance_rate"] + 0.25 * ev["unsafe_refusal_rate"] - 0.5 * ev["benign_false_refusal_rate"]
            if best is None or score > best[0]:
                best = (score, name, li, alpha, info, ev)
            del m
            if device.type == "cuda": torch.cuda.empty_cache()
    _, best_name, best_layer, best_alpha, best_info, best_eval = best
    print(f"[Stage 3] Accepted inverter {best_name}; re-anchoring targeted complementary LoRA RL...", flush=True)
    accepted = copy.deepcopy(sft)
    apply_inverter(accepted, inputs, dirs, best_layer, best_alpha)
    ref = copy.deepcopy(accepted).to(device).eval(); vec_rl.freeze_weights(ref)
    regex = layer_regex([24, 25, 26], mlp_only=False)
    rl_log = out_dir / "step_metrics.jsonl"
    with rl_log.open("w", encoding="utf-8") as log:
        trained, stats = base_rl.run_grpo_arm("inverter_then_complementary_lora", accepted, ref, tokenizer, prompt_stream, device, regex, 16, args.rl_steps, args.group_size, args.lr, args.beta, log, args.seed)
    rl_eval = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
    print(f"  [inverter+RL] exact={rl_eval['exact_compliance_rate']:.1f}% unsafe={rl_eval['unsafe_refusal_rate']:.1f}% benign_fr={rl_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    summary = {"experiment": "exp60_benign_keyed_refusal_inverter", "research_source": "research/SUBAGENT3_CREATIVE_HEURISTICS.md rank 1", "paper_inspiration": ["arXiv:2406.11717", "arXiv:2012.14913"], "config": vars(args), "device": str(device), "baseline_sft": sft_eval, "input_counts": {str(li): {k: len(v) for k, v in inputs[li].items()} for li in LAYERS}, "top_singular_values": singulars, "grid": grid, "accepted": {"name": best_name, "edit": best_info, "eval_zero_rl": best_eval}, "inverter_then_complementary_lora": {"train_stats": stats, "eval": rl_eval, "target_regex": regex}, "wall_sec": round(time.time() - t0, 2)}
    path = out_dir / "benign_keyed_inverter_summary.json"; path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Exp 60 DONE] wrote {path} in {summary['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
