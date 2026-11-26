# 1× H100 Competition Quick-Boot Guide (`eu-north1`, `halo:hopper`)

Verified against official Nebius Compute pricing (`docs.nebius.com/compute/resources/pricing`) and Halo's official Docker guide (`whitecircle/halo/agent-docs/infrastructure/docker.md`).

## 1. Current Nebius H100 & Storage Prices (`eu-north1`)
- **1× NVIDIA H100 NVLink (`gpu-h100-sxm`, `1gpu-16vcpu-200gb`, 80 GB VRAM):**
  - **On-Demand:** **`$4.50 / GPU-hr`** (`$27` = **6.0 full hours**)
  - **Preemptible:** **from `$0.79 / GPU-hr`** (up to **~34 hours** on `$27`, subject to reclamation)
- **Persistent Pre-Warm Disk (100 GiB in `eu-north1`):**
  - **Network SSD Non-replicated:** `$0.053 / GiB / month` (**`$5.30/mo`**, ~$0.17/day)
  - **Shared Filesystem SSD:** `$0.080 / GiB / month` (**`$8.00/mo`**, ~$0.26/day)

## 2. Why This Boots Directly Into the Repo (Zero Setup on GPU Time)
Per Halo's official `docker.md`:
- `public.ecr.aws/whitecircle/halo:hopper` (~30–32 GB) **already bakes in** PyTorch 2.11+cu130, FA2, FA3 (SM90), DeepEP, Transformers 5.20.2, TRL 1.14.2, PEFT, Liger, `HF_XET_HIGH_PERFORMANCE=1`, and the `halo` CLI.
- No `git clone whitecircle/halo`, no `uv`/`pip install`, and no venv setup run on paid H100 time — `quickboot_h100.sh` mounts `halo-competition-lab` directly as `/workspace` (`-w /workspace`) and runs `halo launch` immediately.
- Host requirements checked automatically before launch: **NVIDIA Driver >= R580** (CUDA 13.2 floor), NVIDIA Container Toolkit, and short `TMPDIR` (`/mnt/halo/t`, avoiding the 108-byte Unix-socket limit).
- Setting `HF_HOME=/mnt/halo/hf` automatically persists Halo's `TRITON_CACHE_DIR` across `--rm` containers (keeping runs bit-identical and skipping autotuner re-benchmarks).

## 3. Exact 3-Step Competition Workflow
1. **Before booting any GPU ($0 on CPU):**
   ```bash
   DRY_RUN=1 ./scripts/quickboot_h100.sh configs/h100_quickboot_manifest.yaml
   ```
2. **Pre-warm once on a penny-cheap CPU VM (`eu-north1`, disk mounted at `/mnt/halo`):**
   ```bash
   HF_TOKEN=... ./scripts/prewarm_nebius_volume.sh /mnt/halo "Qwen/Qwen3-4B"
   ```
3. **Launch on 1× H100 (`eu-north1`) with hard timeout + auto-shutdown:**
   ```bash
   chmod +x scripts/quickboot_h100.sh
   AUTO_SHUTDOWN=1 HARD_TIMEOUT_MIN=55 ./scripts/quickboot_h100.sh configs/h100_quickboot_manifest.yaml
   ```
