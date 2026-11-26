#!/usr/bin/env python3
"""Zero-cost CPU Pre-Flight Gate for 1x H100 (80 GB, halo:hopper) Competition Quick-Boot.

Catches YAML, schema, weight_decay, dataset, manifest, and H100-VRAM issues on CPU ($0)
BEFORE any paid Nebius H100 VM boots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"Missing manifest: {manifest_path}"]
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return [f"Manifest is not a YAML mapping: {manifest_path}"]

    target = data.get("target_hardware", {})
    if target.get("gpu_type") != "H100-80GB":
        errors.append(f"Expected gpu_type='H100-80GB', got {target.get('gpu_type')!r}")
    if int(target.get("num_gpus", 0)) != 1:
        errors.append(f"Expected num_gpus=1 for affordable H100 quick-boot, got {target.get('num_gpus')!r}")

    max_min = int(data.get("budget_guard", {}).get("hard_timeout_minutes", 0))
    if max_min <= 0 or max_min > 180:
        errors.append(f"Invalid hard_timeout_minutes={max_min} (must be 1..180)")

    jobs = data.get("jobs", [])
    if not jobs:
        errors.append("Manifest has no jobs queued.")
    for idx, job in enumerate(jobs):
        name = job.get("name", f"job_{idx}")
        kind = job.get("kind")
        if kind not in {"halo_sft", "halo_grpo", "python_script"}:
            errors.append(f"[{name}] unsupported kind={kind!r}")
        cfg_rel = job.get("config")
        if cfg_rel:
            cfg_path = ROOT / cfg_rel
            if not cfg_path.exists():
                errors.append(f"[{name}] missing config: {cfg_rel}")
            else:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                if not isinstance(cfg, dict):
                    errors.append(f"[{name}] invalid YAML dict in {cfg_rel}")
                    continue
                # Guard: masked/targeted RL configs MUST use weight_decay: 0.0
                if "target" in name.lower() or "disjoint" in name.lower() or "random" in name.lower():
                    wd = cfg.get("weight_decay", 0.0)
                    if float(wd) != 0.0:
                        errors.append(f"[{name}] targeted config {cfg_rel} has weight_decay={wd} (must be 0.0)")
                # Guard: single-GPU H100 configs should not require EP/TP > 1
                for k in ("expert_parallel_size", "tensor_parallel_size", "pipeline_parallel_size"):
                    if int(cfg.get(k, 1)) > 1:
                        errors.append(f"[{name}] {cfg_rel} sets {k}={cfg[k]} > 1 (incompatible with 1x H100)")
        script_rel = job.get("script")
        if script_rel and not (ROOT / script_rel).exists():
            errors.append(f"[{name}] missing script: {script_rel}")
    return errors


def check_datasets() -> list[str]:
    errors: list[str] = []
    for rel in ("data/policy_train.jsonl", "data/policy_eval.jsonl"):
        p = ROOT / rel
        if not p.exists():
            errors.append(f"Missing dataset: {rel}")
            continue
        n = 0
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{rel}:{line_no} invalid JSON: {exc}")
                break
            if "messages" not in obj:
                errors.append(f"{rel}:{line_no} missing 'messages' key")
                break
            n += 1
        if n == 0:
            errors.append(f"{rel} has 0 rows")
    return errors


def check_all_yamls() -> tuple[int, list[str]]:
    errors: list[str] = []
    files = sorted(list(ROOT.glob("configs/*.yaml")) + list(ROOT.glob("experiments/**/configs/*.yaml")))
    for p in files:
        try:
            obj = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                errors.append(f"{p.relative_to(ROOT)}: top-level is not a mapping")
        except Exception as exc:
            errors.append(f"{p.relative_to(ROOT)}: YAML parse error: {exc}")
    return len(files), errors


def main() -> int:
    manifest_arg = sys.argv[1] if len(sys.argv) > 1 else "configs/h100_quickboot_manifest.yaml"
    manifest_path = ROOT / manifest_arg

    yaml_count, yaml_errs = check_all_yamls()
    ds_errs = check_datasets()
    man_errs = check_manifest(manifest_path)

    all_errs = yaml_errs + ds_errs + man_errs
    print(f"[Preflight] Checked {yaml_count} YAML configs, 2 policy datasets, and {manifest_arg}")
    if all_errs:
        print(f"[FAIL] {len(all_errs)} issue(s) caught BEFORE paid GPU boot:")
        for e in all_errs:
            print(f"  - {e}")
        return 1
    print("[PASS] 0 errors — safe to launch 1x H100 Quick-Boot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
