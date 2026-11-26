#!/usr/bin/env python3
"""Fast pre-push smoke test for experiment scripts.

Catches the class of runtime bug that compile-check misses: tokenizer encoding slices
being inserted into sets, plus import/argparse/model-path/runtime failures in a tiny
CPU execution. HALO_SMOKE_TEST=1 makes the shared heldout evaluator use 2 rows and
8 generated tokens; this harness additionally clamps common loop/batch arguments.
"""
from __future__ import annotations

import argparse
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
    "--rl-steps": "1",
    "--group-size": "2",
    "--per-class": "1",
    "--n-per-class": "2",
    "--n-pairs": "2",
    "--alphas": ["0.1"],
}


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
    problems = scan_source(script)
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
    if proc.returncode != 0:
        print(tail, flush=True)
        raise SystemExit(f"smoke execution failed with exit {proc.returncode}")
    print(tail, flush=True)
    print(f"[smoke PASS] {script} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
