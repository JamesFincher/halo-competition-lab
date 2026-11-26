# exp152 — Qwen3-4B behavioral scale pivot

Tests H24/H23 on `Qwen/Qwen3-4B-Instruct-2507`: zero-shot, fixed-update plain LoRA on the original 160-row corpus, fixed-update plain LoRA on the 1,600-row counterfactual corpus, capability-replay-protected LoRA, and architecture-mapped H22-style sparse-edit + complementary-MLP-LoRA composition.

The experiment declares `exclusive_gpu` because it is a 4B training job. It checkpoints adapters outside `results/`; `results/arm_results.jsonl` is incremental and the queue-completing summary JSON is written only after all arms finish.
