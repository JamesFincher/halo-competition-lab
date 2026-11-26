#!/bin/bash
# Self-sustaining GPU feeder: pulls repo, runs any experiment with a script that lacks results.
# Runs FOREVER on the VM — zero downtime by construction.
cd ~/halo-competition-lab
while true; do
  if ! git pull -q origin main 2>/dev/null; then
    echo "$(date): git pull failed; leaving repo untouched and skipping cycle"
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
    if not has_completed_result(exp):
        exps.append({"name": exp, "script": s})
# prioritize explicit expNN experiments numerically, newest first; unnumbered names keep lexical fallback
def priority_key(item):
    m = re.search(r"exp(\d+)", item["name"])
    return (int(m.group(1)) if m else -1, item["name"])
ordered = sorted(exps, key=priority_key, reverse=True)
def exclusive_gpu(item):
    profile_path = os.path.join("experiments", item["name"], "gpu_profile.json")
    try:
        return bool(json.load(open(profile_path)).get("exclusive_gpu"))
    except Exception:
        return False
# Large-model jobs declare exclusive_gpu. Queue the newest such job solo rather than
# packing it with seven 135M jobs that could collectively OOM the H100.
exclusive = [item for item in ordered if exclusive_gpu(item)]
queue = exclusive[:1] if exclusive else ordered[:8]
json.dump({"experiments": queue}, open("scripts/auto_queue.json", "w"), indent=1)
print(f"auto-queue: {len(queue)}")
PY
    if [ -s scripts/auto_queue.json ] && [ "$(python3 -c 'import json; print(len(json.load(open("scripts/auto_queue.json"))["experiments"]))' 2>/dev/null || echo 0)" -gt 0 ]; then
      sudo docker rm -f autowave >/dev/null 2>&1
      sudo docker run -d --name autowave --gpus all --ipc=host --shm-size=8g -v ~/halo-competition-lab:/lab -v /mnt/halo:/mnt/halo -e HF_HOME=/mnt/halo/hf -e TMPDIR=/mnt/halo/t -w /lab public.ecr.aws/whitecircle/halo:hopper python scripts/wave_runner.py scripts/auto_queue.json
      echo "$(date): launched autowave"
    fi
  fi
  # benchmark pass: if wave finished and no recent benchmark, run it
  if [ "$RUNNING" -eq 0 ] && [ ! -f results/real_benchmark_latest.json -o $(find results/real_benchmark_latest.json -mmin +30 2>/dev/null | wc -l) -gt 0 ]; then
    sudo docker rm -f benchloop >/dev/null 2>&1
    sudo docker run -d --name benchloop --gpus all --ipc=host --shm-size=8g -v ~/halo-competition-lab:/lab -v /mnt/halo:/mnt/halo -e HF_HOME=/mnt/halo/hf -e TMPDIR=/mnt/halo/t -w /lab public.ecr.aws/whitecircle/halo:hopper python scripts/real_benchmark.py --n 100 --out results/real_benchmark_latest.json >/dev/null 2>&1
  fi
  sleep 120
done
