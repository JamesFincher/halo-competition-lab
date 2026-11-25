#!/usr/bin/env python3
"""Multi-Domain Tuned Seek -> Token-Aligned Sniff -> Gated Edit -> Disjoint RL Benchmark.

Tests our Tuned Loop v2 across 3 completely distinct competition use cases on SmolLM2-135M-Instruct:
  1. `policy_router`: Agent Tool-Call Policy & Secret-Safety (starting from Halo SFT checkpoint-36)
  2. `json_extraction`: Strict Raw-JSON Support Ticket Extraction (starting from BASE SmolLM2-135M)
  3. `math_verifiable`: Concise Verifiable Arithmetic / Word Problems (`#### <ans>`, BASE SmolLM2-135M)

For each domain, benchmarks 4 arms on unseen held-out tasks (0 overlap with train/sniff):
  - Baseline (Starting model)
  - Full-Model LoRA GRPO (all 30 layers, r=8)
  - Tuned Pure Sniff-Edit Only (0 gradient steps, multi-strength dev gate)
  - Tuned Closed-Loop Seek -> Sniff -> Gated Edit -> Disjoint Re-Anchored RL (hard-prompt focused)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402


# ---------------- Domain 2: Strict Raw-JSON Support Ticket Extraction ----------------
def build_json_extraction_dataset(seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    sys_p = (
        "Extract ticket fields. Output ONLY a raw JSON object with NO markdown, NO code fences, and NO extra text: "
        '{"intent": "billing"|"bug"|"access", "priority": "low"|"high", "id": <int>}'
    )
    templates = [
        ("billing", "high", "Customer reports duplicate charge of ${amt} on invoice #{id} and needs urgent refund."),
        ("billing", "low", "User asks for a PDF copy of receipt #{id} for accounting records."),
        ("bug", "high", "Production checkout crashes with HTTP 500 on order #{id} for all users."),
        ("bug", "low", "Minor tooltip alignment glitch on settings page for ticket #{id}."),
        ("access", "high", "Admin locked out of production SSO workspace on request #{id}, blocking deploy."),
        ("access", "low", "New intern requests read-only viewer access to staging docs on ticket #{id}."),
    ]
    rows = []
    for i in range(60):
        intent, prio, tmpl = templates[i % len(templates)]
        tid = 1000 + i * 7 + rng.randint(1, 5)
        usr = tmpl.format(amt=rng.randint(49, 499), id=tid)
        target = {"intent": intent, "priority": prio, "id": tid}
        asst = json.dumps(target)
        rows.append(
            {
                "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr}],
                "target": target,
                "assistant_text": asst,
                "canonical_fail": f"Sure! Here is the extracted JSON for your ticket:\n```json\n{asst}\n```\nLet me know if you need anything else!",
            }
        )
    rng.shuffle(rows)
    return rows[:24], rows[24:32], rows[32:56]  # train=24, dev=8, unseen_eval=24


def score_json_extraction(pred_text: str, target: dict) -> tuple[float, dict]:
    raw = pred_text.strip()
    is_pure_raw_json = raw.startswith("{") and raw.endswith("}") and "```" not in raw
    obj = base_rl.parse_json_object(raw)
    if obj is None:
        return 0.0, {"json_valid": False, "pure_raw": False, "exact": False}
    intent_ok = obj.get("intent") == target["intent"]
    prio_ok = obj.get("priority") == target["priority"]
    id_ok = str(obj.get("id")) == str(target["id"])
    rew = (0.35 if is_pure_raw_json else 0.10) + (0.25 if intent_ok else 0.0) + (0.20 if prio_ok else 0.0) + (0.20 if id_ok else 0.0)
    exact = is_pure_raw_json and intent_ok and prio_ok and id_ok
    return rew, {"json_valid": True, "pure_raw": is_pure_raw_json, "exact": exact}


# ---------------- Domain 3: Concise Verifiable Arithmetic Reasoning ----------------
def build_math_dataset(seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    sys_p = (
        "Solve the arithmetic word problem concisely in 1 short sentence, then end with the exact final integer formatted as:\n"
        "#### <integer>"
    )
    rows = []
    for i in range(60):
        a = rng.randint(4, 25)
        b = rng.randint(3, 19)
        c = rng.randint(2, 12)
        if i % 3 == 0:
            ans = a + b * c
            usr = f"A warehouse has {a} boxes and receives {b} pallets with {c} boxes each. How many boxes are there in total?"
            asst = f"{a} + ({b} * {c}) = {ans}.\n#### {ans}"
        elif i % 3 == 1:
            ans = a * b - c
            usr = f"A shop packs {a} trays with {b} muffins each and sells {c} muffins. How many muffins remain?"
            asst = f"({a} * {b}) - {c} = {ans}.\n#### {ans}"
        else:
            ans = (a + b) * c
            usr = f"Each kit has {a} red bolts and {b} blue bolts. How many bolts are in {c} kits?"
            asst = f"({a} + {b}) * {c} = {ans}.\n#### {ans}"
        rows.append(
            {
                "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr}],
                "target": {"answer": ans},
                "assistant_text": asst,
                "canonical_fail": f"Let's think step by step about this problem carefully. First we read the numbers {a}, {b}, and {c}. The answer is probably around {ans + 3}.",
            }
        )
    rng.shuffle(rows)
    return rows[:24], rows[24:32], rows[32:56]  # train=24, dev=8, unseen_eval=24


def score_math(pred_text: str, target: dict) -> tuple[float, dict]:
    txt = pred_text.strip()
    m = re.search(r"####\s*(-?\d+)", txt)
    has_fmt = m is not None
    concise = len(txt.split()) <= 35
    ans_ok = False
    if m:
        try:
            ans_ok = int(m.group(1)) == int(target["answer"])
        except Exception:
            ans_ok = False
    else:
        nums = re.findall(r"-?\d+", txt)
        if nums and int(nums[-1]) == int(target["answer"]):
            ans_ok = True
    rew = (0.30 if has_fmt else 0.0) + (0.15 if concise else 0.0) + (0.55 if ans_ok else 0.0)
    exact = has_fmt and ans_ok
    return rew, {"has_fmt": has_fmt, "ans_ok": ans_ok, "exact": exact}


# ---------------- Generic Tuned Seek -> Sniff -> Gated Edit -> Disjoint RL ----------------
@torch.no_grad()
def eval_generic(model, tokenizer, rows, score_fn, device, max_new_tokens=64) -> dict:
    model.eval()
    tot_rew = 0.0
    exact_cnt = 0
    aux_counts: dict[str, int] = {}
    for r in rows:
        p = base_rl.build_prompt(tokenizer, r["messages"])
        enc = tokenizer(p, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        rew, info = score_fn(gen, r["target"])
        tot_rew += rew
        exact_cnt += int(info["exact"])
        for k, v in info.items():
            if isinstance(v, bool):
                aux_counts[k] = aux_counts.get(k, 0) + int(v)
    n = max(len(rows), 1)
    res = {"n_eval": len(rows), "mean_reward": round(tot_rew / n, 4), "exact_rate": round(100.0 * exact_cnt / n, 1)}
    for k, v in aux_counts.items():
        res[f"{k}_rate"] = round(100.0 * v / n, 1)
    return res


@torch.no_grad()
def token_aligned_sniff(model, tokenizer, train_rows, score_fn, device, n_pairs=8) -> dict:
    """SEEK live failing prompts, then SNIFF afflicted layers using token-aligned canonical contrast pairs."""
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)

    hard_rows = []
    pass_texts = []
    fail_texts = []
    for r in train_rows[: max(n_pairs * 2, 12)]:
        p = base_rl.build_prompt(tokenizer, r["messages"])
        enc = tokenizer(p, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        rew, info = score_fn(gen, r["target"])
        if not info["exact"]:
            hard_rows.append(r)
        if len(pass_texts) < n_pairs:
            pass_texts.append(p + r["assistant_text"])
            fail_texts.append(p + r["canonical_fail"])

    def _mean_acts(texts: list[str]) -> list[torch.Tensor]:
        acc = [None] * n_layers
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
            cap = {}
            hooks = []
            for idx, lyr in enumerate(layers):
                def _mk(i: int):
                    def _hk(_m, _inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        cap[i] = h[0, -24:, :].detach().float().mean(dim=0).cpu()
                    return _hk
                hooks.append(lyr.register_forward_hook(_mk(idx)))
            try:
                _ = model(**enc)
            finally:
                for h in hooks:
                    h.remove()
            for idx in range(n_layers):
                acc[idx] = cap[idx] if acc[idx] is None else acc[idx] + cap[idx]
        return [a / max(len(texts), 1) for a in acc]

    mu_p = _mean_acts(pass_texts)
    mu_f = _mean_acts(fail_texts)
    dirs = {i: (mu_f[i] - mu_p[i]) for i in range(n_layers)}
    ranked = sorted([{"layer": i, "l2": round(float(dirs[i].norm().item()), 4)} for i in range(n_layers)], key=lambda x: x["l2"], reverse=True)
    return {"hard_rows": hard_rows if hard_rows else train_rows[:8], "dirs": dirs, "ranked": ranked}


def gated_edit_generic(model, tokenizer, dev_rows, dirs, edit_layers, score_fn, device, strengths=(0.35, 0.55, 0.75), top_columns=64):
    base_ev = eval_generic(model, tokenizer, dev_rows, score_fn, device)
    best_m, best_ev = model, base_ev
    meta = {"accepted": False, "base_dev_reward": base_ev["mean_reward"], "edit_layers": edit_layers}
    for st in strengths:
        cand = copy.deepcopy(model).to(device)
        info = base_rl.apply_surgical_pre_edit(cand, dirs, edit_layers, top_columns=top_columns, strength=st)
        ev = eval_generic(cand, tokenizer, dev_rows, score_fn, device)
        print(f"    [Gate] layers={edit_layers} st={st} | dev_rew={base_ev['mean_reward']:.3f}->{ev['mean_reward']:.3f} | exact={ev['exact_rate']}%", flush=True)
        if ev["mean_reward"] > best_ev["mean_reward"] + 1e-4:
            best_m, best_ev = cand, ev
            meta = {"accepted": True, "strength": st, "dev_before": base_ev["mean_reward"], "dev_after": ev["mean_reward"], **info}
    return best_m, meta


def run_grpo_generic(
    arm_name, init_model, ref_model, tokenizer, rows, score_fn, device,
    target_modules, lora_r, steps, group_size, lr, beta, log_fh, seed,
):
    torch.manual_seed(seed)
    random.seed(seed)
    model = copy.deepcopy(init_model).to(device)
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.0, target_modules=target_modules, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    t0 = time.perf_counter()
    kl_curve = []
    for step in range(1, steps + 1):
        st0 = time.perf_counter()
        batch = [rows[(2 * (step - 1)) % len(rows)], rows[(2 * (step - 1) + 1) % len(rows)]]
        opt.zero_grad(set_to_none=True)
        step_rews, step_kls = [], []
        loss = torch.zeros((), device=device)
        active = 0
        for row in batch:
            p = base_rl.build_prompt(tokenizer, row["messages"])
            enc = tokenizer(p, return_tensors="pt").to(device)
            p_len = enc["input_ids"].shape[1]
            model.eval()
            roll_ids, rews = [], []
            with torch.no_grad():
                for _ in range(group_size):
                    out = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
                    gen = tokenizer.decode(out[0][p_len:], skip_special_tokens=True)
                    rv, _ = score_fn(gen, row["target"])
                    roll_ids.append(out)
                    rews.append(rv)
            step_rews.extend(rews)
            rt = torch.tensor(rews, dtype=torch.float32, device=device)
            rs = rt.std(unbiased=False)
            if rs.item() < 1e-6:
                continue
            adv = (rt - rt.mean()) / (rs + 1e-6)
            active += 1
            model.train()
            for seq_ids, ai in zip(roll_ids, adv):
                lpc = base_rl.completion_logprobs(model, seq_ids, p_len)
                with torch.no_grad():
                    lpr = base_rl.completion_logprobs(ref_model, seq_ids, p_len)
                step_kls.append((lpc - lpr).detach().abs().item())
                loss = loss + (-(ai.detach() * lpc) + beta * (lpc - lpr) ** 2) / (group_size * len(batch))
        gn = 0.0
        if active > 0 and loss.requires_grad:
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            opt.step()
        mkl = sum(step_kls) / max(len(step_kls), 1) if step_kls else 0.0
        kl_curve.append(round(mkl, 5))
        rec = {"arm": arm_name, "step": step, "mean_reward": round(sum(step_rews) / max(len(step_rews), 1), 4), "active": active, "kl": round(mkl, 5), "grad": round(gn, 4), "sec": round(time.perf_counter() - st0, 1)}
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()
        print(f"  [{arm_name}] step {step:02d}/{steps} | rew={rec['mean_reward']:.3f} | act={active}/2 | kl={rec['kl']:.4f} | {rec['sec']:.1f}s", flush=True)
    return model, {"trainable_params": trainable, "trainable_pct": round(100.0 * trainable / max(total, 1), 3), "wall_sec": round(time.perf_counter() - t0, 1), "final_kl": kl_curve[-1] if kl_curve else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["policy_router", "json_extraction", "math_verifiable"])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    out_dir = ROOT / "experiments/seek_sniff_edit_rl/results/multi_domain"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)

    if args.domain == "policy_router":
        start_model = PeftModel.from_pretrained(base_model, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload()
        tr_all = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
        for r in tr_all:
            r["canonical_fail"] = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'
        ben = [r for r in tr_all if r["target"]["action"] == "call_tool"]
        uns = [r for r in tr_all if r["target"]["action"] == "refuse"]
        dev_rows = ben[-6:] + uns[-6:]
        train_rows = [x for pair in zip(ben[:-6], uns[:-6]) for x in pair]
        eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
        score_fn = base_rl.score_completion
    elif args.domain == "json_extraction":
        start_model = base_model
        train_rows, dev_rows, eval_rows = build_json_extraction_dataset(args.seed)
        score_fn = score_json_extraction
    else:
        start_model = base_model
        train_rows, dev_rows, eval_rows = build_math_dataset(args.seed)
        score_fn = score_math

    start_model.eval()
    ref_start = copy.deepcopy(start_model).to(device)
    ref_start.eval()

    print(f"\n=== DOMAIN: {args.domain} ===", flush=True)
    ev_base = eval_generic(start_model, tokenizer, eval_rows, score_fn, device)
    print(f"  [Baseline] eval: {ev_base}", flush=True)

    sniff = token_aligned_sniff(start_model, tokenizer, train_rows, score_fn, device, n_pairs=8)
    top2 = [r["layer"] for r in sniff["ranked"][:2]]
    comp6 = sorted([r["layer"] for r in sniff["ranked"][2:8]])
    print(f"  [Sniff] hard_rows={len(sniff['hard_rows'])} | top2={top2} | comp6={comp6}", flush=True)

    edited_m, edit_meta = gated_edit_generic(start_model, tokenizer, dev_rows, sniff["dirs"], top2, score_fn, device)
    ev_edit = eval_generic(edited_m, tokenizer, eval_rows, score_fn, device)
    print(f"  [Sniff-Edit Only] eval: {ev_edit} (accepted={edit_meta['accepted']})", flush=True)

    all_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    comp_regex = f".*\\.layers\\.({'|'.join(map(str, comp6))})\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"

    log_path = out_dir / f"step_metrics_{args.domain}.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        # Arm A: Full-Model LoRA GRPO
        full_m, full_stats = run_grpo_generic(
            f"{args.domain}_full_rl", start_model, ref_start, tokenizer, train_rows, score_fn, device,
            all_targets, 8, args.steps, 4, 2e-4, 0.02, fh, args.seed,
        )
        ev_full = eval_generic(full_m, tokenizer, eval_rows, score_fn, device)
        print(f"  [Full RL] eval: {ev_full}", flush=True)

        # Arm B: Tuned Closed-Loop (Sniff-Edit + Re-Anchored Disjoint RL on hard-focused stream)
        ref_ed = copy.deepcopy(edited_m).to(device)
        ref_ed.eval()
        # Hard-prompt-focused stream interleaved with general train_rows
        rl_stream = sniff["hard_rows"] + train_rows
        loop_m, loop_stats = run_grpo_generic(
            f"{args.domain}_tuned_loop", edited_m, ref_ed, tokenizer, rl_stream, score_fn, device,
            comp_regex, 16, args.steps, 4, 6e-5, 0.03, fh, args.seed,
        )
        ev_loop = eval_generic(loop_m, tokenizer, eval_rows, score_fn, device)
        print(f"  [Tuned Loop] eval: {ev_loop}", flush=True)

    summary = {
        "domain": args.domain,
        "top2_edit_layers": top2,
        "comp6_rl_layers": comp6,
        "n_hard_rows_sought": len(sniff["hard_rows"]),
        "edit_meta": edit_meta,
        "arms": {
            "baseline": {"eval": ev_base},
            "full_rl": {"stats": full_stats, "eval": ev_full},
            "sniff_edit_only": {"eval": ev_edit},
            "tuned_seek_sniff_edit_rl": {"stats": loop_stats, "eval": ev_loop},
        },
    }
    out_file = out_dir / f"summary_{args.domain}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
