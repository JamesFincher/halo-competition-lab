#!/usr/bin/env python3
"""Offline preflight validator for competition configs & datasets (run BEFORE renting GPUs)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent.parent


def check_datasets() -> None:
    train_path = ROOT / "data/policy_train.jsonl"
    eval_path = ROOT / "data/policy_eval.jsonl"
    assert train_path.exists() and eval_path.exists(), "Run scripts/build_dataset.py first."

    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
    eval_rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line]
    train_prompts = {r["messages"][1]["content"] for r in train_rows}
    eval_prompts = {r["messages"][1]["content"] for r in eval_rows}
    overlap = train_prompts & eval_prompts
    assert not overlap, f"Found {len(overlap)} overlapping prompts between train and eval!"
    print(f"[OK] Datasets: {len(train_rows)} train, {len(eval_rows)} eval, 0 prompt overlap.")


def check_configs() -> None:
    yaml = YAML(typ="safe")
    configs = sorted((ROOT / "configs").glob("*.yaml")) + sorted((ROOT / "experiments").glob("**/configs/*.yaml"))
    assert configs, "No YAML configs found."
    for cfg in configs:
        data = yaml.load(cfg.read_text(encoding="utf-8"))
        assert isinstance(data, dict) and data, f"Empty or invalid YAML: {cfg}"
        rel = cfg.relative_to(ROOT)
        # Guard against Halo script_arguments.py:120-127 ValueError (regex chars inside a YAML list)
        ltm = data.get("lora_target_modules")
        if isinstance(ltm, list):
            for item in ltm:
                assert not any(ch in str(item) for ch in ".*|()[]$"), (
                    f"Regex chars inside lora_target_modules list will raise in Halo: {rel}"
                )
        print(f"[OK] YAML syntax: {rel} ({len(data)} top-level keys)")


def check_env_hints() -> None:
    for var, purpose in [
        ("HF_TOKEN", "Faster HF downloads & gated models (e.g. Gemma 4 26B-A4B)"),
        ("NEBIUS_API_KEY", "Nebius Token Factory data generation / judging"),
        ("WHITECIRCLE_API_KEY", "White Circle policy reward API for online GRPO"),
        ("WANDB_API_KEY", "Weights & Biases run logging"),
    ]:
        status = "SET" if os.environ.get(var) else "not set"
        print(f"[INFO] {var}: {status} — {purpose}")


def main() -> None:
    check_datasets()
    check_configs()
    check_env_hints()


if __name__ == "__main__":
    main()
