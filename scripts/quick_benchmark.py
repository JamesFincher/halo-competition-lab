#!/usr/bin/env python3
"""Quick standardized benchmark — scores ANY checkpoint in ~30s on the 40-task held-out eval.

Usage: python scripts/quick_benchmark.py [--ckpt PATH]  (default: SFT LoRA checkpoint)
Emits rubric-ready metrics JSON so every experiment/run is instantly comparable.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Path to model checkpoint dir (default: SFT LoRA merged)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    if args.ckpt:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=dtype).to(device)
    else:
        from peft import PeftModel
        raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=dtype).to(device)
        model = PeftModel.from_pretrained(raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload().to(device)
    model.eval()
    ev = base_rl.evaluate_heldout(model, tok, eval_rows, device)
    wall = round(time.time() - t0, 2)
    gain = ev["exact_compliance_rate"] - 55.0
    rec = {"tag": args.tag, "wall_sec": wall, "device": str(device), "baseline_exact": 55.0, "gain_pp": round(gain, 1), **ev}
    out_path = args.out or str(ROOT / f"results/bench_{args.tag}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rec, indent=2))
    print(f"[BENCH {args.tag}] exact={ev['exact_compliance_rate']:.1f}% ({gain:+.1f}pp) | pol={ev['policy_accuracy']:.1f}% | unsafe={ev['unsafe_refusal_rate']:.1f}% | benignfr={ev['benign_false_refusal_rate']:.1f}% | json={ev['json_valid_rate']:.1f}% | {wall}s")

if __name__ == "__main__":
    main()
