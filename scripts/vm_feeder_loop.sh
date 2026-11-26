#!/bin/bash
# Self-sustaining GPU feeder: pulls repo, runs any experiment with a script that lacks results.
# Runs FOREVER on the VM — zero downtime by construction.
cd ~/halo-competition-lab
while true; do
  git pull -q origin main 2>/dev/null || git checkout -q -- . 2>/dev/null; git pull -q origin main 2>/dev/null
  RUNNING=$(sudo docker ps --format '{{.Names}}' | grep -c '^wave\|^mega\|^autowave' || true)
  if [ "$RUNNING" -eq 0 ]; then
    # build queue of all experiments missing results
    python3 - <<'PY' 2>/dev/null
import json, glob, os
exps = []
for s in sorted(glob.glob("experiments/*/scripts/run_*.py")):
    exp = s.split("/")[1]
    if not glob.glob(f"experiments/{exp}/results/*.json"):
        exps.append({"name": exp, "script": s})
# prioritize newest (higher exp numbers in name or path)
json.dump({"experiments": exps[-8:]}, open("scripts/auto_queue.json", "w"), indent=1)
print(f"auto-queue: {len(exps[-8:])}")
PY
    if [ -s scripts/auto_queue.json ] && [ "$(python3 -c 'import json; print(len(json.load(open("scripts/auto_queue.json"))["experiments"]))' 2>/dev/null || echo 0)" -gt 0 ]; then
      sudo docker rm -f autowave >/dev/null 2>&1
      sudo docker run -d --name autowave --gpus all --ipc=host --shm-size=8g -v ~/halo-competition-lab:/lab -v /mnt/halo:/mnt/halo -e HF_HOME=/mnt/halo/hf -e TMPDIR=/mnt/halo/t -w /lab public.ecr.aws/whitecircle/halo:hopper python scripts/wave_runner.py scripts/auto_queue.json
      echo "$(date): launched autowave"
    fi
  fi
  sleep 120
done
