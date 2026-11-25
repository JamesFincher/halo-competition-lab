# Experiment 9: INSEP — Refusal-Direction Inseparability (White Circle Security Track)

**Priority:** #1 Security / Safety Competition Differentiator (**8.5/10** for White Circle Security Track)  
**Target Models:** `Qwen/Qwen3-4B` (pilot) → `google/gemma-4-26B-A4B-it` / `unsloth/gpt-oss-20b-BF16`

## Core Hypothesis
Standard safety alignment concentrates refusal along a single linearly separable residual-stream direction, making open-weight models trivially vulnerable to white-box **abliteration** (subtracting the difference-in-means refusal direction from write-out matrices — the exact opposite of our Stage-1 edit!). **INSEP (`arXiv:2611.28314`, Nov 20, 2026)** uses adversarial eigendecomposition to find the most separable harmful-vs-benign prompt subsets and trains a low-rank adapter to **minimize linear separability along that refusal direction** while preserving refusal accuracy, benign helpfulness, and MMLU.

## Supporting Literature
- **`arXiv:2611.28314` (*INSEP: Refusal-Direction Inseparability as a Defense Against Abliteration*, Brugere et al., Nov 20, 2026):** Ranked **#1 on mean composite score** across 8 LLMs (1B–14B), 6 defenses, and 5 white-box abliteration attacks while preserving coherence, directness, low over-refusal, and MMLU.

## Halo v1.3.0rc1 Compatibility
- Uses standard Halo PEFT LoRA (`use_peft: true`, `lora_r: 16`) + auxiliary separability-reduction penalty on middle-to-late residual-stream activations (`scripts/insep_separability_probe.py`), merging cleanly back into BF16 hub weights for vLLM/competition serving.
- Directly pairs with White Circle's policy/security evaluation suite: resists both prompt-injection AND white-box abliteration without over-refusing benign requests.

## Files
- `scripts/insep_separability_probe.py` — Measures layer-wise harmful-vs-benign Fisher linear separability along the difference-in-means refusal direction (before and after Halo LoRA training) to verify abliteration resistance.
- `configs/qwen3-4b-insep-lora.yaml` — Halo LoRA config targeting the highest-separability layers.

## Compute & Go/No-Go Criteria
- **Pilot cost (`Qwen3-4B`):** ~2–4 GPU-hrs (~$8–$15).
- **GO:** Post-abliteration unsafe compliance stays ≤10% (vs >70% on standard safety SFT/LoRA) while benign false-refusal stays ≤5% and MMLU drops ≤1 pp.
