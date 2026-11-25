#!/usr/bin/env bash
# Zero-waste Halo Docker launcher for a rented NVIDIA H100/H200 (Hopper) or B200/B300 (Blackwell) node.
# Usage:
#   ./scripts/launch_nebius_gpu.sh configs/gptoss-20b-policy-sft-ep8.yaml [hopper|blackwell] [num_gpus]
set -euo pipefail

CONFIG_PATH="${1:?Pass a Halo YAML config path (e.g. configs/gptoss-20b-policy-sft-ep8.yaml)}"
ARCH="${2:-hopper}"
NUM_GPUS="${3:-$(nvidia-smi -L | wc -l)}"
HALO_IMAGE="public.ecr.aws/whitecircle/halo:${ARCH}-1.3.0rc1"
HALO_REF="3a21b4dbdb56cb840d42a59e98e40f967a1d2807"

echo "=== Preflight ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
SCRATCH=$(findmnt -rbno TARGET,AVAIL,FSTYPE | awk '$1!="/" && $3!~/tmpfs|overlay|squashfs|nfs|fuse|autofs/ && $2+0>20e9{print $2,$1}' | sort -rn | head -1 | awk '{print $2}')
SCRATCH="${SCRATCH:-/tmp/halo-scratch}"
mkdir -p "$SCRATCH"/{hf/datasets,tmp,halo_scratch,checkpoints}
echo "Scratch volume: $SCRATCH ($(df -h "$SCRATCH" | awk 'NR==2{print $4}') free)"

echo "=== Pull pinned Halo image ($HALO_IMAGE) ==="
docker pull "$HALO_IMAGE"

if [ ! -d "$SCRATCH/halo" ]; then
  git clone https://github.com/whitecircle/halo.git "$SCRATCH/halo"
  git -C "$SCRATCH/halo" checkout "$HALO_REF"
fi

ENV_ARGS=()
[ -f .env ] && ENV_ARGS+=(--env-file .env)

echo "=== Launching Halo SFT on $NUM_GPUS GPU(s) ==="
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=128g \
  --cap-add=SYS_PTRACE \
  "${ENV_ARGS[@]}" \
  -e HF_HOME="$SCRATCH/hf" \
  -e HF_DATASETS_CACHE="$SCRATCH/hf/datasets" \
  -e TMPDIR="$SCRATCH/tmp" \
  -e HALO_DATA_ROOT="$SCRATCH/halo_scratch" \
  -v "$(pwd)":/lab \
  -v "$SCRATCH/halo":/workspace \
  -v "$SCRATCH:$SCRATCH" \
  -w /workspace \
  "$HALO_IMAGE" \
  bash -lc "halo launch sft /lab/${CONFIG_PATH} -n ${NUM_GPUS}"
