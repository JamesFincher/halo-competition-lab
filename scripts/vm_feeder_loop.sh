#!/bin/bash
# Self-sustaining GPU feeder: pulls repo, runs any experiment with a script that lacks results.
# Runs FOREVER on the VM — zero downtime by construction.
cd ~/halo-competition-lab
while true; do
  git fetch -q origin main 2>/dev/null
  sudo chown -R ubuntu:ubuntu experiments/ scripts/ 2>/dev/null
  mkdir -p ~/results_rescue
  git status --porcelain | grep '^??' | awk '{print $2}' | grep -E 'results|jsonl' | grep -Ev '^results/swarm_logs/|/results/crashes/' | while read -r f; do
    mkdir -p "$HOME/results_rescue/$(dirname "$f")"; cp -a "$f" "$HOME/results_rescue/$f" 2>/dev/null
  done
  if ! git reset -q --hard origin/main 2>/dev/null; then
    echo "$(date): git reset failed; leaving repo untouched and skipping cycle"
    sleep 120
    continue
  fi
  RUNNING=$(sudo docker ps --format '{{.Names}}' | grep -c '^wave\|^mega\|^autowave' || true)
  if [ "$RUNNING" -eq 0 ]; then
    # build queue of all experiments missing results
    python3 - <<'PY' 2>/dev/null
import json, glob, os, re
exps = []
def has_completed_result(exp):
    root = f"experiments/{exp}/results"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not (d.startswith("backup") or d in {"archive", "invalid"})]
        if any(f.endswith(".json") for f in filenames):
            return True
    return False

for s in sorted(glob.glob("experiments/*/scripts/run_*.py")):
    exp = s.split("/")[1]
    # Persistent upstream blocklist: BLOCKED.md means a human/agent deliberately
    # de-queued the experiment, independent of result artifacts or stale waves.
    if os.path.exists(os.path.join("experiments", exp, "BLOCKED.md")):
        continue
    if not has_completed_result(exp):
        exps.append({"name": exp, "script": s})
# prioritize explicit expNN experiments numerically, newest first; unnumbered names keep lexical fallback
def priority_key(item):
    m = re.search(r"exp(\d+)", item["name"])
    return (int(m.group(1)) if m else -1, item["name"])
ordered = sorted(exps, key=priority_key, reverse=True)
def gpu_profile(item):
    profile_path = os.path.join("experiments", item["name"], "gpu_profile.json")
    try:
        return json.load(open(profile_path))
    except Exception:
        return {}

def exclusive_gpu(item):
    return bool(gpu_profile(item).get("exclusive_gpu"))

def exclusive_priority(item):
    # Higher priority first; experiment number is only the tie-breaker. This lets
    # follow-up controls queue behind the parent experiment they interpret.
    return int(gpu_profile(item).get("queue_priority", 0))

# Large-model jobs declare exclusive_gpu. The default remains solo execution, but
# measured low-memory jobs can opt into conservative pairwise packing. This honors
# the zero-idle/parallel-variant mandate without risking unmeasured 7B/fp32 jobs.
def allow_parallel(item):
    return bool(gpu_profile(item).get("allow_parallel"))

def estimated_peak_gb(item):
    try:
        return float(gpu_profile(item).get("estimated_peak_gb", 80.0))
    except Exception:
        return 80.0

def pack_exclusive(items, memory_budget_gb=64.0, max_jobs=2):
    if not items:
        return []
    chosen = [items[0]]  # never skip the highest-priority job
    if not allow_parallel(items[0]):
        return chosen
    used = estimated_peak_gb(items[0])
    for item in items[1:]:
        if len(chosen) >= max_jobs:
            break
        if not allow_parallel(item):
            # Preserve explicit priority order: never backfill behind a solo job.
            break
        est = estimated_peak_gb(item)
        if used + est > memory_budget_gb:
            break
        chosen.append(item)
        used += est
    return chosen

exclusive_unordered = [item for item in ordered if exclusive_gpu(item)]
exclusive = sorted(exclusive_unordered, key=lambda item: (exclusive_priority(item), priority_key(item)), reverse=True)
queue = pack_exclusive(exclusive) if exclusive else ordered[:8]
json.dump({"experiments": queue}, open("scripts/auto_queue.json", "w"), indent=1)
print(f"auto-queue: {len(queue)}")
PY
    if [ -s scripts/auto_queue.json ] && [ "$(python3 -c 'import json; print(len(json.load(open("scripts/auto_queue.json"))["experiments"]))' 2>/dev/null || echo 0)" -gt 0 ]; then
      sudo docker rm -f autowave >/dev/null 2>&1
      sudo docker run -d --name autowave --gpus all --ipc=host --shm-size=8g -v ~/halo-competition-lab:/lab -v /mnt/halo:/mnt/halo -e HF_HOME=/mnt/halo/hf -e TMPDIR=/mnt/halo/t -w /lab public.ecr.aws/whitecircle/halo:hopper python scripts/wave_runner.py scripts/auto_queue.json
      echo "$(date): launched autowave"
    fi
  fi
  # push result artifacts to HF dataset so harvesting does not depend on any live session
  if [ -n "$HF_TOKEN" ]; then
    sudo docker run --rm --ipc=host -v ~/halo-competition-lab:/lab -e HF_TOKEN="$HF_TOKEN" -e HF_HOME=/mnt/halo/hf -w /lab public.ecr.aws/whitecircle/halo:hopper python scripts/push_results_to_hf.py >/dev/null 2>&1
  fi
  sleep 120
done
