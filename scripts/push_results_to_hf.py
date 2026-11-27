#!/usr/bin/env python3
"""Push experiment result artifacts to the HuggingFace results dataset.

The feeder runs this every cycle, so uploads must be incremental and batched. A
content-hash state file in the repository root (kept outside the feeder rescue glob) prevents repeatedly recommitting
hundreds of unchanged artifacts; changed files are snapshotted in memory and
uploaded in multi-file commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "heartandkarma/halo-competition-lab-results"
STATE_NAME = ".hf_push_state.json"
BATCH_SIZE = 40


def artifact_paths(root: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for results_dir in sorted(root.glob("experiments/*/results")):
        exp = results_dir.parent.name
        for artifact in sorted(results_dir.rglob("*")):
            if not artifact.is_file():
                continue
            rel = artifact.relative_to(results_dir)
            if any(part.startswith("backup") or part in {"archive", "invalid"} for part in rel.parts):
                continue
            is_summary = artifact.suffix == ".json" and "summary" in artifact.name
            is_crash = "crashes" in rel.parts
            is_jsonl = artifact.suffix == ".jsonl"
            if not (is_summary or is_crash or is_jsonl):
                continue
            artifacts[f"{exp}/{rel.as_posix()}"] = artifact

    for doc in sorted((root / "research").glob("*.md")) + [root / "docs/EXPERIMENT_LEADERBOARD.md"]:
        if doc.exists():
            artifacts[f"research/{doc.name}"] = doc
    return artifacts


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore local upload state and re-upload every artifact")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    state_path = root / STATE_NAME
    state = {} if args.force else load_state(state_path)
    artifacts = artifact_paths(root)

    changed = []
    skipped = 0
    for dest, path in sorted(artifacts.items()):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest = sha256(data)
        meta = {"sha256": digest, "size": len(data)}
        if state.get(dest) == meta:
            skipped += 1
            continue
        changed.append((dest, data, meta))

    print(f"scanned {len(artifacts)} artifacts: {len(changed)} changed, {skipped} unchanged", flush=True)
    if args.dry_run:
        for dest, data, _meta in changed[:20]:
            print(f"would upload {dest} ({len(data)} bytes)", flush=True)
        return

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    commits = 0
    uploaded = 0
    for start in range(0, len(changed), BATCH_SIZE):
        chunk = changed[start:start + BATCH_SIZE]
        operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=data) for dest, data, _meta in chunk]
        if not operations:
            continue
        api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Update {len(operations)} halo result artifacts",
        )
        commits += 1
        uploaded += len(operations)
        for dest, _data, meta in chunk:
            state[dest] = meta
        save_state(state_path, state)

    save_state(state_path, state)
    print(f"pushed {uploaded} changed artifacts in {commits} commit(s) to hf.co/datasets/{REPO}", flush=True)


if __name__ == "__main__":
    main()
