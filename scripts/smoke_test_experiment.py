#!/usr/bin/env python3
"""Fast pre-push smoke test for experiment scripts.

Catches runtime bugs that compile-check misses: tokenizer encoding slices inserted
into sets, post-compute `.item()` misuse on Python floats, import/argparse/model-path
failures, and invalid/missing result serialization. It executes the complete script
control flow end-to-end on CPU with tiny loop counts/dimensions, then validates every
result artifact before restoring the pre-smoke results tree.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BAD_PATTERNS = [
    (re.compile(r"set\([^\n]*tokenizer\.encode\([^\n]*\)\[:1\]"), "tokenizer.encode slice inserted into set (unhashable list)"),
]

TINY_VALUES = {
    "--steps": "1",
    "--gate-steps": "1",
    "--rl-steps": "1",
    "--group-size": "2",
    "--per-class": "1",
    "--n-per-class": "2",
    "--n-pairs": "2",
    "--bench-n": "2",
    "--benchmark-n": "2",
    "--cap-items-per-source": "2",
    "--continuation-n": "2",
    "--probe-rows": "4",
    "--population": "2",
    "--generations": "1",
    "--batch-size": "1",
    "--ems-rows": "2",
    "--candidate-top": "2",
    "--candidate-random": "1",
    "--observe-points": "2",
    "--max-new-tokens": "8",
    "--rank": "2",
    "--alphas": ["0.1"],
}


def float_item_hazards(path: Path) -> list[str]:
    """AST-exact detection of `.item()` called on the result of float(...)."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    problems = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "item"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "float"
        ):
            problems.append(f"line {node.lineno}: .item() called on float(...) result")
    return problems


def artifact_fingerprints(results_dir: Path) -> dict[str, tuple[int, int, str]]:
    if not results_dir.exists():
        return {}
    out = {}
    for path in sorted(list(results_dir.rglob("*.json")) + list(results_dir.rglob("*.jsonl"))):
        data = path.read_bytes()
        stat = path.stat()
        out[str(path.relative_to(results_dir))] = (stat.st_mtime_ns, len(data), hashlib.sha256(data).hexdigest())
    return out


def validate_result_artifacts(results_dir: Path, before: dict[str, tuple[int, int, str]]) -> list[str]:
    """Parse every result artifact produced during the end-to-end smoke run."""
    problems = []
    json_files = sorted(results_dir.rglob("*.json")) if results_dir.exists() else []
    jsonl_files = sorted(results_dir.rglob("*.jsonl")) if results_dir.exists() else []
    after = artifact_fingerprints(results_dir)
    produced_json = [name for name in after if name.endswith(".json") and (name not in before or after[name] != before[name])]
    if not produced_json:
        problems.append("no new/updated JSON summary artifact written; end-to-end serialization path was not proven")
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{path.name}: invalid JSON after write: {type(exc).__name__}: {exc}")
    for path in jsonl_files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, 1):
                if line.strip():
                    json.loads(line)
        except Exception as exc:
            problems.append(f"{path.name}: invalid JSONL: {type(exc).__name__}: {exc}")
    return problems


def scan_source(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems = []
    for pattern, message in BAD_PATTERNS:
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            problems.append(f"line {line}: {message}")
    return problems


def accepted_flags(script: Path, env: dict[str, str]) -> tuple[set[str], str]:
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        env=env,
    )
    help_text = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(f"--help failed with exit {proc.returncode}:\n{help_text[-4000:]}")
    return set(re.findall(r"(?<![\w-])--[a-z0-9][a-z0-9-]*", help_text)), help_text


def tiny_args(flags: set[str]) -> list[str]:
    args: list[str] = []
    for flag, value in TINY_VALUES.items():
        if flag not in flags:
            continue
        if isinstance(value, list):
            args.extend([flag, *value])
        else:
            args.extend([flag, value])
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()
    script = args.script.resolve()
    if not script.exists():
        raise SystemExit(f"missing script: {script}")

    print(f"[smoke] scanning {script}", flush=True)
    problems = scan_source(script) + float_item_hazards(script)
    if problems:
        raise SystemExit("static runtime hazards:\n" + "\n".join(problems))
    py_compile.compile(str(script), doraise=True)
    print("[smoke] static scan + compile passed", flush=True)

    env = os.environ.copy()
    env["HALO_SMOKE_TEST"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    flags, _help = accepted_flags(script, env)
    run_args = tiny_args(flags) + args.extra_arg
    cmd = [sys.executable, str(script), *run_args]
    print(f"[smoke] executing: {' '.join(cmd)}", flush=True)
    exp_dir = script.parent.parent if script.parent.name == "scripts" else script.parent
    results_dir = exp_dir / "results"
    backup_dir = Path(tempfile.mkdtemp(prefix="halo_smoke_results_"))
    had_results = results_dir.exists()
    if had_results:
        shutil.copytree(results_dir, backup_dir / "results", dirs_exist_ok=True)
    before_artifacts = artifact_fingerprints(results_dir)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            env=env,
        )
        artifact_problems = validate_result_artifacts(results_dir, before_artifacts)
    finally:
        # Smoke runs use truncated generations and must never create queue-satisfying
        # experiment results. Restore the exact pre-run results tree.
        if results_dir.exists():
            shutil.rmtree(results_dir)
        if had_results:
            shutil.copytree(backup_dir / "results", results_dir, dirs_exist_ok=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
    elapsed = time.time() - t0
    tail = proc.stdout[-6000:]
    if artifact_problems:
        print(tail, flush=True)
        raise SystemExit("smoke artifact validation failed:\n" + "\n".join(artifact_problems))
    if proc.returncode != 0:
        print(tail, flush=True)
        raise SystemExit(f"smoke execution failed with exit {proc.returncode}")
    print(tail, flush=True)
    print(f"[smoke PASS] {script} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
