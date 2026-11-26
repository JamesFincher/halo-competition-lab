#!/usr/bin/env bash
# Zero-Setup Quick-Boot Launcher for 1x NVIDIA H100 SXM (80 GB, eu-north1, halo:hopper).
# Follows official Halo Docker docs (agent-docs/infrastructure/docker.md):
#   - Uses public.ecr.aws/whitecircle/halo:hopper (Halo CLI, PyTorch 2.11+cu130, FA2/FA3, DeepEP baked in)
#   - Mounts THIS repo directly at /workspace (-w /workspace): ZERO cloning, ZERO pip install on GPU time
#   - Checks NVIDIA host driver >= R580 (CUDA 13.2 floor) + disk/RAM
#   - Uses persistent HF_HOME (auto-backs TRITON_CACHE_DIR) + short TMPDIR (<=108-byte socket limit)
#   - Runs queued manifest arms back-to-back under a hard wall-clock timeout + optional AUTO_SHUTDOWN
#
# Usage:
#   DRY_RUN=1 ./scripts/quickboot_h100.sh configs/h100_quickboot_manifest.yaml
#   AUTO_SHUTDOWN=1 HARD_TIMEOUT_MIN=45 ./scripts/quickboot_h100.sh configs/h100_quickboot_manifest.yaml
set -euo pipefail

MANIFEST="${1:-configs/h100_quickboot_manifest.yaml}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
HARD_TIMEOUT_MIN="${HARD_TIMEOUT_MIN:-45}"
HALO_IMAGE="${HALO_IMAGE:-public.ecr.aws/whitecircle/halo:hopper}"
DOCKER_RUNTIME_FLAG="${DOCKER_RUNTIME:+--runtime=${DOCKER_RUNTIME}}"
CACHE_ROOT="${CACHE_ROOT:-}"

cleanup() {
  local rc=$?
  echo "=== [Quick-Boot Exit rc=${rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)] ==="
  if [ "${AUTO_SHUTDOWN}" = "1" ] && [ "${DRY_RUN}" = "0" ]; then
    echo "AUTO_SHUTDOWN=1 enabled — powering off VM in 15s to protect Nebius credits..."
    sleep 15
    sudo shutdown -h now || poweroff || true
  fi
}
trap cleanup EXIT

echo "=== [1/4] Zero-Cost Schema Preflight ==="
PYTHON_BIN="${PYTHON_BIN:-$([ -x /home/ubuntu/halo-cpu-venv/bin/python ] && echo /home/ubuntu/halo-cpu-venv/bin/python || command -v python3)}"
if "${PYTHON_BIN}" -c "import yaml" >/dev/null 2>&1; then
  "${PYTHON_BIN}" scripts/quickboot_preflight.py "${MANIFEST}"
else
  echo "Host Python has no PyYAML (expected per Halo docker.md); preflight will run inside ${HALO_IMAGE}."
fi

echo "=== [2/4] Resolving Persistent Cache Volume ==="
if [ -z "${CACHE_ROOT}" ]; then
  if [ -d /mnt/halo ]; then
    CACHE_ROOT="/mnt/halo"
  else
    CACHE_ROOT=$(findmnt -rbno TARGET,AVAIL,FSTYPE | awk '$1!="/" && $3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $2+0>20e9{print $2,$1}' | sort -rn | head -1 | awk '{print $2}' || true)
    CACHE_ROOT="${CACHE_ROOT:-/tmp/h}"
  fi
fi
echo "CACHE_ROOT=${CACHE_ROOT} | HALO_IMAGE=${HALO_IMAGE} | TIMEOUT=${HARD_TIMEOUT_MIN}m | AUTO_SHUTDOWN=${AUTO_SHUTDOWN}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "=== [DRY_RUN=1] Resolved Direct-to-Repo Queue (Zero Setup) ==="
  "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
for i, j in enumerate(m.get("jobs", []), 1):
    print(f"  {i}. [{j['kind']}] {j['name']} -> config={j.get('config')} script={j.get('script')}")
PY
  echo "[DRY_RUN OK] Ready for direct-to-repo H100 boot."
  exit 0
fi

echo "=== [3/4] Live H100 Preflight (Driver >= R580, VRAM, Disk) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRIVER_MAJOR:-0}" -lt 580 ]; then
  echo "ERROR: Halo CUDA 13.2 requires NVIDIA driver >= R580 (found ${DRIVER_MAJOR})." >&2
  exit 1
fi
free -h
mkdir -p "${CACHE_ROOT}"/{hf/datasets,t,s,checkpoints}
df -h "${CACHE_ROOT}"

# Load pre-warmed image tarball if present on persistent volume (avoids 30 GB pull on GPU time)
TAR_PATH="${CACHE_ROOT}/docker/halo-hopper.tar"
if ! docker image inspect "${HALO_IMAGE}" >/dev/null 2>&1; then
  if [ -f "${TAR_PATH}" ]; then
    echo "Fast-loading pre-warmed Halo Hopper image from ${TAR_PATH}..."
    docker load -i "${TAR_PATH}"
  else
    echo "Pulling ${HALO_IMAGE}..."
    docker pull "${HALO_IMAGE}"
  fi
fi

ENV_ARGS=()
[ -f .env ] && ENV_ARGS+=(--env-file .env)

echo "=== [4/4] Launching Directly Into Repo (/workspace) — Hard Timeout ${HARD_TIMEOUT_MIN}m ==="
timeout --signal=SIGTERM --kill-after=30s "${HARD_TIMEOUT_MIN}m" \
  docker run --rm --gpus all ${DOCKER_RUNTIME_FLAG} \
    --network host \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=64g \
    --cap-add=SYS_PTRACE \
    "${ENV_ARGS[@]}" \
    -e PYTHONPATH=/workspace \
    -e HF_HOME="${CACHE_ROOT}/hf" \
    -e HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets" \
    -e TMPDIR="${CACHE_ROOT}/t" \
    -e HALO_DATA_ROOT="${CACHE_ROOT}/s" \
    -v "$(pwd)":/workspace \
    -v "${CACHE_ROOT}:${CACHE_ROOT}" \
    -w /workspace \
    "${HALO_IMAGE}" \
    python - "/workspace/${MANIFEST}" <<'PY'
import subprocess, sys, yaml
manifest = yaml.safe_load(open(sys.argv[1]))
jobs = manifest.get("jobs", [])
for idx, job in enumerate(jobs, 1):
    name, kind = job["name"], job["kind"]
    print(f"\n>>> [Job {idx}/{len(jobs)}] {name} ({kind})", flush=True)
    if kind == "halo_sft":
        cmd = ["halo", "launch", "sft", f"/workspace/{job['config']}", "-n", "1"]
    elif kind == "halo_grpo":
        cmd = ["halo", "launch", "grpo", f"/workspace/{job['config']}", "-n", "1"]
    elif kind == "python_script":
        cmd = ["python", f"/workspace/{job['script']}"] + job.get("args", [])
    else:
        raise ValueError(f"Unknown job kind: {kind}")
    subprocess.run(cmd, check=True)
print("\n>>> ALL QUEUED H100 JOBS COMPLETED.", flush=True)
PY
