# Large-model feasibility for the pivot tracks

Date: 2026-11-26. Hardware target: one preemptible H100 SXM (80 GB) in the White Circle `halo:hopper` container. These are pre-launch estimates, not measured benchmarks; stage-0 scripts must record actual memory and wall time.

## Model facts

| Model | Parameters | Native dtype | Architecture facts |
|---|---:|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4,022,468,096 | bf16 | 36 dense layers, hidden 2560, 32 query heads, 8 KV heads, head dim 128, context 262k |
| `allenai/OLMoE-1B-7B-0125-Instruct` | 6,919,161,856 total | bf16 | 16 layers, hidden 2048, 64 experts/layer, top-8/token, about 1.3B active parameters/token |

## Memory estimates

| Workload | Fixed model memory | Trainable/optimizer memory | Activation/workspace estimate | Expected peak | H100 verdict |
|---|---:|---:|---:|---:|---|
| Qwen3-4B inference/eval | ~8.0 GB | none | 1-4 GB | ~9-12 GB | comfortable |
| Qwen3-4B LoRA, all attention+MLP projections, r16 | ~8.0 GB frozen | ~33.0M trainable; <0.5 GB including grads/Adam | ~8-18 GB at batch 4-8 x seq 384 with gradient checkpointing | ~17-26 GB | comfortable |
| Qwen3-4B full fine-tune reference (not recommended) | ~8.0 GB | ~48 GB grads+Adam before activations | 8-18 GB | ~64-74 GB | risky/preemptible-hostile; unnecessary |
| OLMoE inference/eval | ~13.8 GB | none | 2-5 GB | ~16-19 GB | comfortable |
| OLMoE expert low-rank adapters | ~13.8 GB frozen | adapter-dependent, normally <0.5 GB | ~8-18 GB at batch 4-8 x seq 384 with checkpointing | ~22-32 GB | comfortable |

Important OLMoE note: only about 1.3B parameters are active per token, but all 6.9B parameters must remain resident unless expert offloading is implemented. The memory floor is therefore total-model bf16 size, not active-parameter size.

## Wall-clock estimates

These assume one H100, bf16, gradient checkpointing, batch 8, max sequence length 384 unless stated otherwise. First-run HF download is additional: roughly 8 GB for Qwen3-4B and 14 GB for OLMoE; on the current VM this should be cached under `/mnt/halo/hf` and reused by later containers.

| Job | Estimate basis | Expected wall clock |
|---|---|---:|
| Qwen3 behavior LoRA epoch | 1,600 rows / batch 8 = 200 optimizer steps; estimated 0.6-1.5 sec/step | 2-5 min/epoch |
| Qwen3 behavior LoRA 3 epochs | 600 optimizer steps | 6-15 min training |
| Qwen3 one full experiment arm | training + 120-row scaled eval + 40-row legacy eval + continuation probes/checkpointing | 15-30 min |
| Qwen3 four-arm comparison, sequential inside one experiment | baseline/zero-shot + original-data LoRA + scaled-data LoRA + H22-style composition | 60-120 min |
| OLMoE stage 0 | behavior eval + continuation benchmark + router profiling; no training | 10-20 min after model download |
| OLMoE expert-adapter training arm | 300-600 low-rank adapter steps plus eval | 15-35 min/arm |

Throughput uncertainty is large because the hopper image's attention kernels, host networking, and other containers affect performance. Stage-0 scripts must record `torch.cuda.max_memory_allocated`, per-phase wall time, tokens/sec, and eval/sec before any larger sweep is launched.

## Preemption-resilience requirements

1. Do not launch large-model experiments as ordinary members of an eight-way 135M wave. They need an `exclusive_gpu` profile so the feeder queues them solo or at most alongside jobs that declare tiny memory use.
2. Save LoRA/adapter checkpoints every 100 optimizer steps and after every arm under the experiment directory; the repo directory is VM-mounted, so these survive container loss.
3. Write per-arm partial JSON immediately after each arm rather than one final post-compute write. Treat final summarization as a compaction step only.
4. On restart, detect valid existing checkpoints/results and resume at the first incomplete arm. Never re-download or retrain completed arms.
5. Keep stage 0 measurement-only and cheap. Use it to validate tokenizer/chat-template behavior, model load, cached generation, expert routing capture, memory, and wall clock before any adapter training.
6. Use `--max-steps`, `--batch-size`, `--eval-limit`, and `--bench-n` smoke flags so CPU/container smoke tests exercise the full serialization path without pretending to produce scientific metrics.

## Track-specific implementation notes

### Qwen3-4B behavioral pivot

Use `Qwen/Qwen3-4B-Instruct-2507` rather than a thinking variant. Plain LoRA on attention+MLP projections is the critical reference. The H22-style composition should be architecture-mapped rather than copied literally: final-two-layer sparse edit means Qwen layers 35/34 (analogous to SmolLM2 29/28); complementary MLP LoRA means the four preceding layers 30-33 (analogous to 24-27). This mapping is a hypothesis, not validated fact.

### OLMoE expert-adapter pivot

Use `allenai/OLMoE-1B-7B-0125-Instruct`. In transformers 5.24.1, `OlmoeSparseMoeBlock` exposes `gate` plus a batched `OlmoeExperts` parameter set (`gate_up_proj` and `down_proj` shaped by expert). Router profiling can capture `top_k_index` and `top_k_weights` without modifying weights. Expert down-projection adapters must be token-local/weight-like operations; unlike full-sequence inserted attention heads, they do not depend on prior-token hidden-state availability, so normal cached generation should be valid. This claim still needs explicit smoke verification because adapters implemented through hooks can otherwise repeat the slider-cache mistake.
