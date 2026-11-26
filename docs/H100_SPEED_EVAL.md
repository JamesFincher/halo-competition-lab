# Live 1× H100 SXM vs CPU Speed Evaluation (`halo:hopper`)

**Instance:** Nebius `eu-north1` Preemptible `gpu-h100-sxm` (`1gpu-16vcpu-200gb`, `NVIDIA H100 80GB HBM3`, driver `580.196.07`)  
**Container:** Stock `public.ecr.aws/whitecircle/halo:hopper` (`PyTorch 2.11.0+cu130`, `CUDA 13.2`, `FlashAttention 2.8.3.post1`) — zero custom installs  
**Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (`bfloat16`, peak VRAM `1.483 GB / 80 GB`)

## 1. Side-by-Side Speedup Table (Measured Live)

| Benchmark Workload | CPU Baseline (This VM) | Live 1× H100 (`halo:hopper`) | Measured Speedup |
|---|---:|---:|---:|
| **Exp 18 — 7-Behavior DLA Fingerprint** (`48 prompts × 30 layers`) | `7.03s` | **`0.98s`** | **`7.2× faster`** |
| **Exp 17 — 5-Step Integrated Gradients (`IG-5`)** (`20 fwd+bwd`) | `4.72s` | **`0.40s`** | **`11.7× faster`** |
| **6-Step Micro-RL Loop** (`6 steps × 4 rollouts × 48 tok + fwd/bwd`) | `123.30s` | **`7.71s`** | **`16.0× faster`** |
| **40-Task Unseen Eval Generation** (`batch=20`, `64 new tok`) | `~38.00s` | **`7.86s`** | **`4.8× faster`** |
| *Model Load + `to(cuda)`* | — | `6.42s` | — |
| *Cold `docker pull public.ecr.aws/whitecircle/halo:hopper`* | — | **`280s` (4m 40s)** | Saved via persistent pre-warm disk |

## 2. Critical Launch Bug Caught Live (Fixed in `scripts/quickboot_h100.sh`)
- Mounting our lab repo directly over `/workspace` shadowed Halo's editable-installed package inside `halo:hopper` (`halo --help` exited `1`).
- **Fix:** Mount `halo-competition-lab` at `/lab` (`-v "$(pwd)":/lab -w /lab`) and leave `/workspace` untouched so `halo` CLI works out of the box with zero cloning or setup.
