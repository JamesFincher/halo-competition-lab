#!/usr/bin/env python3
"""Experiment 101: Merge-Aware Served-Weight Action-Token Fine-Tuning.

Implements HYPOTHESES H10. Custom low-rank adapters train either in ordinary
unrounded fp32 or through a straight-through BF16-rounded merged-weight forward.
Loss is factorially applied to all completion tokens or only action/tool/policy
values. Every result is evaluated both in-process and after merge+BF16 export.

Research anchors: research/HYPOTHESES.md H10 and arXiv:2611.32064.
"""
from __future__ import annotations

import argparse, copy, json, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/exp50_surgically_anchored_vector_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_surgically_anchored_vector_rl as vec_rl  # noqa: E402
from validated_edit import apply_validated_rank1_edit  # noqa: E402

COMPLEMENT_LAYERS = [24, 25, 26, 27]
DECISION_SUFFIXES = ('"action": "', '"tool": "', '"policy_code": "')


class MergeAwareLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, served_rounding: bool):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.rank = rank; self.alpha = alpha; self.served_rounding = served_rounding
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device, dtype=torch.float32) / max(rank, 1) ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))

    def effective_weight(self):
        delta = (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        return self.base.weight.float() + delta

    def served_weight(self):
        eff = self.effective_weight()
        if not self.served_rounding: return eff
        rounded = eff.to(torch.bfloat16).float()
        return eff + (rounded - eff).detach()

    def forward(self, x):
        return F.linear(x, self.served_weight().to(dtype=x.dtype), self.base.bias)

    def rounded_export_weight(self):
        return self.effective_weight().detach().to(torch.bfloat16).float()


def install_lora_wrappers(model, layers, rank, alpha, served_rounding):
    registry = []
    for li in layers:
        layer = model.model.layers[li]
        for parent, attrs in ((layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (layer.mlp, ["gate_proj", "up_proj", "down_proj"])):
            for attr in attrs:
                base = getattr(parent, attr)
                wrapped = MergeAwareLoRALinear(base, rank, alpha, served_rounding)
                setattr(parent, attr, wrapped)
                registry.append({"layer": li, "attr": attr, "wrapper": wrapped})
    return registry


def target_linear(model, layer_id, attr):
    layer = model.model.layers[layer_id]
    parent = layer.self_attn if attr in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp
    return getattr(parent, attr)


def export_merged_rounded(edited_model, trained_model, registry):
    exported = copy.deepcopy(edited_model)
    max_err = 0.0
    with torch.no_grad():
        for rec in registry:
            target = target_linear(exported, rec["layer"], rec["attr"])
            w = rec["wrapper"].rounded_export_weight().to(device=target.weight.device, dtype=target.weight.dtype)
            target.weight.copy_(w)
            max_err = max(max_err, float((target.weight.float() - rec["wrapper"].rounded_export_weight().to(target.weight.device)).abs().max().item()))
    return exported, max_err


def decision_positions(tokenizer, completion, mode):
    comp_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if mode == "all_tokens": return list(range(len(comp_ids))), comp_ids, {"fallback": False, "n_decision_tokens": len(comp_ids)}
    positions = set()
    for suffix in DECISION_SUFFIXES:
        start = 0
        while True:
            idx = completion.find(suffix, start)
            if idx < 0: break
            value_start = idx + len(suffix)
            value_end = completion.find('"', value_start)
            if value_end < 0: break
            pre_count = len(tokenizer(completion[:value_start], add_special_tokens=False)["input_ids"])
            end_count = len(tokenizer(completion[:value_end], add_special_tokens=False)["input_ids"])
            positions.update(range(pre_count, max(end_count, pre_count + 1)))
            start = value_end
    fallback = not positions
    if fallback: positions = set(range(len(comp_ids)))
    return sorted(p for p in positions if p < len(comp_ids)), comp_ids, {"fallback": fallback, "n_decision_tokens": len(positions)}


def supervised_sequence_loss(model, ref_model, tokenizer, row, device, mask_mode, beta):
    prompt = base_rl.build_prompt(tokenizer, row["messages"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    positions, comp_ids, mask_info = decision_positions(tokenizer, row["assistant_text"], mask_mode)
    comp_ids = comp_ids.unsqueeze(0).to(device)
    ids = torch.cat([prompt_ids, comp_ids], dim=1)
    plen = prompt_ids.shape[1]
    out = model(ids)
    token_losses = F.cross_entropy(out.logits[:, :-1, :].float().transpose(1, 2), ids[:, 1:], reduction="none")
    selected = []
    for comp_pos in positions:
        pred_index = plen + comp_pos - 1
        if 0 <= pred_index < token_losses.shape[1]: selected.append(token_losses[0, pred_index])
    if not selected: selected = [token_losses[0, plen - 1:].mean()]
    ce = torch.stack(selected).mean()
    lp = base_rl.completion_logprobs(model, ids, plen)
    with torch.no_grad(): lp_ref = base_rl.completion_logprobs(ref_model, ids, plen)
    loss = ce + beta * (lp - lp_ref).pow(2)
    return loss, {"ce": float(ce.detach().item()), "kl2": float((lp - lp_ref).pow(2).detach().item()), **mask_info}


def train_lora_arm(arm_name, edited, ref_model, tokenizer, stream, device, mask_mode, served_rounding, steps, batch_size, lr, beta, seed, rank):
    torch.manual_seed(seed); random.seed(seed)
    model = copy.deepcopy(edited).to(device)
    vec_rl.freeze_weights(model)
    registry = install_lora_wrappers(model, COMPLEMENT_LAYERS, rank, alpha=rank * 2, served_rounding=served_rounding)
    params = [p for rec in registry for p in (rec["wrapper"].lora_A, rec["wrapper"].lora_B)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    trace = []; t0 = time.perf_counter(); cursor = 0
    for step in range(1, steps + 1):
        rows = []
        for _ in range(batch_size): rows.append(stream[cursor % len(stream)]); cursor += 1
        opt.zero_grad(set_to_none=True); losses = []; infos = []
        for row in rows:
            loss, info = supervised_sequence_loss(model, ref_model, tokenizer, row, device, mask_mode, beta)
            losses.append(loss / len(rows)); infos.append(info)
        total = torch.stack(losses).sum(); total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item()); opt.step()
        rec = {"arm": arm_name, "step": step, "loss": round(float(total.detach().item()), 5), "mean_ce": round(sum(x["ce"] for x in infos) / len(infos), 5), "mean_kl2": round(sum(x["kl2"] for x in infos) / len(infos), 6), "decision_token_fallbacks": sum(int(x["fallback"]) for x in infos), "mean_decision_tokens": round(sum(x["n_decision_tokens"] for x in infos) / len(infos), 2), "grad_norm": round(grad_norm, 5)}
        trace.append(rec)
        print(f"  [{arm_name}] step {step:02d}/{steps} | ce={rec['mean_ce']:.4f} | kl2={rec['mean_kl2']:.5f} | fallback={rec['decision_token_fallbacks']} | grad={rec['grad_norm']:.3f}", flush=True)
    stats = {"mask_mode": mask_mode, "served_rounding": served_rounding, "lora_rank": rank, "trainable_params": sum(p.numel() for p in params), "complement_layers": COMPLEMENT_LAYERS, "train_wall_sec": round(time.perf_counter() - t0, 2), "trace": trace}
    model.eval(); return model, registry, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); args.steps = min(args.steps, 10)
    t0 = time.time(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "experiments/exp101_merge_aware_served_weight_finetune/results"; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Exp 101] Merge-Aware Served-Weight Fine-Tuning on {device}", flush=True)
    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl"); eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    stream = vec_rl.make_prompt_stream(train_rows)
    sft, tokenizer = vec_rl.load_sft(device, torch.float32)
    edited = copy.deepcopy(sft)
    edit_info = apply_validated_rank1_edit(edited, tokenizer, train_rows, device, edit_layers=vec_rl.EDIT_LAYERS, top_columns=64, strength=0.90)
    ref_model = copy.deepcopy(edited).to(device).eval(); vec_rl.freeze_weights(ref_model)
    sft_eval = base_rl.evaluate_heldout(sft, tokenizer, eval_rows, device); edit_eval = base_rl.evaluate_heldout(edited, tokenizer, eval_rows, device)
    print(f"  [SFT] exact={sft_eval['exact_compliance_rate']:.1f}% | [post-edit] exact={edit_eval['exact_compliance_rate']:.1f}% unsafe={edit_eval['unsafe_refusal_rate']:.1f}% benign_fr={edit_eval['benign_false_refusal_rate']:.1f}%", flush=True)
    results = {"experiment": "exp101_merge_aware_served_weight_finetune", "research_source": "research/HYPOTHESES.md H10", "paper_inspiration": ["arXiv:2611.32064"], "config": vars(args), "device": str(device), "validated_edit_operator": edit_info, "baseline_sft": sft_eval, "post_edit_anchor": edit_eval, "arms": {}}
    arm_specs = [
        ("standard_unrounded_all_tokens", False, "all_tokens"),
        ("served_weight_all_tokens", True, "all_tokens"),
        ("standard_unrounded_action_only", False, "action_only"),
        ("served_weight_action_only", True, "action_only"),
    ]
    with (out_dir / "step_metrics.jsonl").open("w", encoding="utf-8") as log:
        for arm, served, mask_mode in arm_specs:
            print(f"[Run] {arm}", flush=True)
            trained, registry, stats = train_lora_arm(arm, edited, ref_model, tokenizer, stream, device, mask_mode, served, args.steps, args.batch_size, args.lr, args.beta, args.seed, args.rank)
            for rec in stats["trace"]: log.write(json.dumps(rec) + "\n")
            log.flush()
            hook_eval = base_rl.evaluate_heldout(trained, tokenizer, eval_rows, device)
            exported, export_max_err = export_merged_rounded(edited, trained, registry)
            export_eval = base_rl.evaluate_heldout(exported, tokenizer, eval_rows, device)
            gap = round(hook_eval["exact_compliance_rate"] - export_eval["exact_compliance_rate"], 3)
            print(f"  [{arm}] hook exact={hook_eval['exact_compliance_rate']:.1f}% -> export exact={export_eval['exact_compliance_rate']:.1f}% gap={gap:.1f}pp | export unsafe={export_eval['unsafe_refusal_rate']:.1f}% benign_fr={export_eval['benign_false_refusal_rate']:.1f}%", flush=True)
            results["arms"][arm] = {"train_stats": stats, "hook_eval": hook_eval, "export_eval": export_eval, "hook_minus_export_exact_pp": gap, "export_tensor_max_abs_err": export_max_err}
            del trained, exported
            if device.type == "cuda": torch.cuda.empty_cache()
    results["wall_sec"] = round(time.time() - t0, 2)
    path = out_dir / "merge_aware_finetune_summary.json"; path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Exp 101 DONE] wrote {path} in {results['wall_sec']}s", flush=True)


if __name__ == "__main__":
    main()
