#!/usr/bin/env bash
# Pre-warm a persistent Nebius disk (/mnt/halo) on a penny-cheap CPU VM in eu-north1
# so paid 1x H100 VMs boot directly into the repo with ZERO downloads or setup.
#
# Usage:
#   HF_TOKEN=... ./scripts/prewarm_nebius_volume.sh /mnt/halo "Qwen/Qwen3-4B"
set -euo pipefail

CACHE_ROOT="${1:-/mnt/halo}"
MODEL_ID="${2:-Qwen/Qwen3-4B}"
HALO_IMAGE="${HALO_IMAGE:-public.ecr.aws/whitecircle/halo:hopper}"

echo "=== [1/3] Creating short-path cache layout under ${CACHE_ROOT} ==="
mkdir -p "${CACHE_ROOT}"/{hf/datasets,t,s,checkpoints,docker}
df -h "${CACHE_ROOT}"

echo "=== [2/3] Pre-downloading ${MODEL_ID} into ${CACHE_ROOT}/hf ==="
export HF_HOME="${CACHE_ROOT}/hf"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets"
export HF_XET_HIGH_PERFORMANCE=1
python3 - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="${MODEL_ID}", token=os.environ.get("HF_TOKEN") or None)
print("Cached model at:", path)
PY

echo "=== [3/3] Pre-caching Halo Hopper Docker image (${HALO_IMAGE}, ~30 GB) ==="
TAR_PATH="${CACHE_ROOT}/docker/halo-hopper.tar"
if command -v docker >/dev/null 2>&1; then
  if [ ! -f "${TAR_PATH}" ]; then
    docker pull "${HALO_IMAGE}"
    docker save -o "${TAR_PATH}" "${HALO_IMAGE}"
    echo "Saved ${TAR_PATH}"
  else
    echo "Already cached: ${TAR_PATH}"
  fi
else
  echo "Docker not installed on CPU VM; install docker to pre-cache ${HALO_IMAGE}."
fi

echo "=== PRE-WARM COMPLETE ==="
du -sh "${CACHE_ROOT}"/* 2>/dev/null || true
