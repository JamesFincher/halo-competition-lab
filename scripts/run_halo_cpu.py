#!/usr/bin/env python3
"""Run Halo's real SFT entrypoint on a CPU-only host without modifying upstream halo/.

Upstream halo/model_loading/parallel/loader.py:197 passes ``device_map={"": get_local_rank()}``
(int 0), which transformers >=5 resolves as accelerator device 0 even on CPU-only hosts.
When CUDA is unavailable, this shim substitutes ``device_map="cpu"`` before invoking
Halo's real ``scripts/training/sft.py`` main() so the full Halo config parser, dataset
pipeline, collator, LoRA setup, and DistributedSFTTrainer execute unmodified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: run_halo_cpu.py <path/to/halo-repo> <config.yaml> [extra halo args...]")

    halo_repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(halo_repo))
    sys.argv = [str(halo_repo / "scripts/training/sft.py"), *sys.argv[2:]]

    if not torch.cuda.is_available():
        import halo.model_loading.parallel.loader as loader

        orig_load_whole = loader.load_whole_model

        def _cpu_load_whole_model(*args, **kwargs):
            if "device_map" in kwargs:
                kwargs["device_map"] = "cpu"
            return orig_load_whole(*args, **kwargs)

        loader.load_whole_model = _cpu_load_whole_model

    import scripts.training.sft as halo_sft

    halo_sft.with_distributed_teardown(halo_sft.main)()


if __name__ == "__main__":
    main()
