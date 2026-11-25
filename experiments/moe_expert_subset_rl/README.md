# Experiment 4 (Hypothesis B): MoE Expert-Routing Localization + Soft Router-L2 & RDC

**Priority:** #4 Best MoE Competition Track Upgrade (**7.5/10**)  
**Target Model:** `unsloth/gpt-oss-20b-BF16` (32 experts/layer; full vLLM sync & `routing_replay=True` in Halo v1.3.0rc1)

## Core Hypothesis
Profiling `gpt-oss-20b` router activations on contrastive success-vs-failure rollouts identifies the top-K (e.g. 8/32 = 25%) task-critical experts per layer. Masking gradients on non-selected experts while keeping routing soft via **parameter-space `Router-L2` anchoring** and **Router Drift Control (RDC) response filtering** prevents catastrophic forgetting and off-policy MoE degradation during online/async GRPO.

## Supporting Literature
- **`arXiv:2610.07332` (*Structuring MoE Expert Selection*, Oct 2026):** Base MoEs exhibit natural operation-aligned expert specialization; preserving it yields +5.4 to +12.9 pts on agentic benchmarks, whereas hard-clamping routing causes ~140% relative log-likelihood collapse.
- **`arXiv:2610.21682` (*Router Drift Control / RDC*, Oct 2026):** Filtering high rollout-vs-trainer routing TV-drift sequences improves async MoE RL on AIME24/25 by +5.5 pts over TIS and +4.5 pts over R3.
- **`arXiv:2609.08115` (*Router Prior Bias*, Sep 2026):** Uniform load-balancing loss drops math accuracy by 10–14 pts; simple `Router-L2` matches full RPB within 0.6 pts, while hard routing (`RPB-hard`) severely degrades performance.
- **`arXiv:2407.01906` (*ESFT*, EMNLP 2024):** Updating only task-specialized MoE experts matches/beats full SFT while preserving OOD capability.

## Halo v1.3.0rc1 Compatibility & Gotchas
- **Fused 3D `[E_local, K, N]` expert tensors:** `EPMoELayerBase` packs all local experts per EP rank into a single 3D parameter (`compute.py:77-88`, `expert_lora.py:58-60`). **Never set `requires_grad=False` asymmetrically across EP ranks** (deadlocks DeepEP combine at `layer.py:344-355`). Use `register_ep_expert_grad_mask()` keeping `requires_grad=True` uniformly + `weight_decay: 0.0`.
- **Why `gpt-oss-20b` first:** `Gemma 4 26B-A4B` lacks `routing_replay` and has 48.9% bf16 router mismatch with vLLM (`agent-docs/models/gemma4.md:76`), while `Nemotron 3.5 Lightning` blocks online GRPO weight sync (`agent-docs/models/nemotron-h.md:48,57`).
- **Native grouped expert LoRA:** Requires `lora_r % 8 == 0` (`expert_lora.py:26`) and `rollout_adapter_sync: false` (`adapter_sync.py:187-188`).

## Files
- `scripts/profile_moe_experts.py` — Contrastive expert selector + `register_ep_expert_grad_mask()` + `router_l2_anchor_loss()` + `rdc_filter_mask()`.
- `configs/gptoss-20b-expert-subset-grpo.yaml` — EP4 + native grouped expert LoRA (`r=64`, `moe_balancing: none`, `weight_decay: 0.0`).

## Compute & Go/No-Go Criteria
- **Pilot cost (`gpt-oss-20b`, 4-arm 50-step GRPO at EP4):** ~12–24 GPU-hrs (~$35–$100).
- **GO:** Top-8/32 expert-masked + Router-L2/RDC matches or beats full grouped-LoRA AND beats random 8/32 expert control by ≥2 pts with lower OOD degradation.
