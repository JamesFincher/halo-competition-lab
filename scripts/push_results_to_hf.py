#!/usr/bin/env python3
"""Push experiment result bundles (JSON summaries + logs) to HuggingFace Hub dataset repo."""
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

REPO = "heartandkarma/halo-competition-lab-results"
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
root = Path(__file__).resolve().parents[1]
count = 0
for summary in root.glob("experiments/*/results/*summary*.json"):
    dest = f"{summary.parent.parent.name}/{summary.name}"
    api.upload_file(path_or_fileobj=str(summary), path_in_repo=dest, repo_id=REPO, repo_type="dataset")
    count += 1
# Include crash artifacts and partial arm JSONL so failures/progress are visible even
# when no final summary exists. Keep these as small text artifacts; checkpoints stay local.
for results_dir in root.glob("experiments/*/results"):
    exp = results_dir.parent.name
    for artifact in results_dir.rglob("*"):
        if not artifact.is_file():
            continue
        rel = artifact.relative_to(results_dir)
        is_crash = "crashes" in rel.parts
        is_jsonl = artifact.suffix == ".jsonl"
        if not (is_crash or is_jsonl):
            continue
        dest = f"{exp}/{rel.as_posix()}"
        api.upload_file(path_or_fileobj=str(artifact), path_in_repo=dest, repo_id=REPO, repo_type="dataset")
        count += 1
for doc in list((root/"research").glob("*.md")) + [root/"docs/EXPERIMENT_LEADERBOARD.md"]:
    if doc.exists():
        api.upload_file(path_or_fileobj=str(doc), path_in_repo=f"research/{doc.name}", repo_id=REPO, repo_type="dataset")
        count += 1
print(f"pushed {count} artifacts to hf.co/datasets/{REPO}")
