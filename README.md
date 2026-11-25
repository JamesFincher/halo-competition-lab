# Halo Competition Lab

Preparation repository for the **White Circle Halo Post-Training Competition** (Dec 1–10, 2026; built on [`whitecircle/halo`](https://github.com/whitecircle/halo) `v1.3.0rc1`, pinned at commit `3a21b4dbdb56cb840d42a59e98e40f967a1d2807`).

This repo provides:
1. **End-to-end verified Halo SFT + held-out evaluation** on a tiny dense model (`HuggingFaceTB/SmolLM2-135M-Instruct`, LoRA `r=16`) for **Agent Tool-Call Policy & Secret-Safety Compliance** — runnable on CPU for rapid offline iteration before spending cloud GPU minutes.
2. **Competition-ready Halo configs** for accepted partner model families:
   - `google/gemma-4-26B-A4B-it` — Online GRPO + White Circle policy rewards (3 GPUs: 1 vLLM + 2 trainer EP2)
   - `unsloth/gpt-oss-20b-BF16` — Single-node 8×GPU EP8 + native grouped expert LoRA SFT
   - `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` — Single-node 8×GPU EP8 full SFT
3. **Zero-waste cloud-GPU launch & preflight tooling**:
   - Offline config/dataset validator (`scripts/preflight_check.py`)
   - Pinned Halo Docker launcher for Nebius/cloud H100/H200/B200 nodes (`scripts/launch_nebius_gpu.sh`)
   - Nebius SkyPilot template (`configs/skypilot-nebius-h200-b200.yaml`)
   - Nebius Token Factory OpenAI-compatible data generator & LLM judge (`scripts/tokenfactory_generate_and_judge.py`)

---

## Verified Tiny-Model Halo SFT + Eval (CPU Smoke Test)

**Task:** Policy-enforcing agent tool router across 6 tools (`read_file`, `write_file`, `run_sql`, `run_shell`, `send_http`, `search_docs`) and 4 balanced classes (`ALLOW`, `SECRET_EXFIL`, `DESTRUCTIVE_OP`, `PROMPT_INJECTION`), with strict JSON output.
- **Dataset:** 160 train / 40 held-out eval rows (`data/policy_train.jsonl`, `data/policy_eval.jsonl`), **0 prompt overlap** (verified by `scripts/preflight_check.py`).
- **Trainer:** Real `whitecircle/halo` v1.3.0rc1 (`scripts/training/sft.py`, `DistributedSFTTrainer`, PEFT LoRA `r=16`, `alpha=32`, 4.88M / 139.4M trainable params = 3.50%, 2 epochs / 36 steps).
- **Upstream note:** `halo/model_loading/parallel/loader.py:197` passes `device_map={"": get_local_rank()}` (int `0`), which `transformers>=5` resolves as an accelerator device on CPU-only hosts; `scripts/run_halo_cpu.py` substitutes `device_map="cpu"` only when `not torch.cuda.is_available()`, leaving upstream Halo untouched.

### Held-Out Results (40 Unseen Tasks, Greedy Decoding)

| Metric | Base (`SmolLM2-135M-Instruct`) | Halo LoRA (`checkpoint-36`) | Delta |
|---|---:|---:|---:|
| `json_valid_rate` | 7.5% | **100.0%** | +92.5 pp |
| `action_accuracy` | 5.0% | **77.5%** | +72.5 pp |
| `policy_accuracy` | 5.0% | **57.5%** | +52.5 pp |
| `unsafe_refusal_rate` | 0.0% | **100.0%** | +100.0 pp |
| `benign_false_refusal_rate` | 0.0% | 90.0% | +90.0 pp (over-refusal) |
| `exact_compliance_rate` | 0.0% | **55.0%** | +55.0 pp |

Full JSON output: [`results/eval_summary.json`](results/eval_summary.json).

### Key Takeaway for Scaling to Partner Models
With 3 refusal categories vs 1 benign category (75% refusal in SFT), pure SFT learns strict JSON formatting (100%) and catches 100% of unsafe tool calls, but over-refuses benign requests (90% false refusal). This directly motivates the **two-stage competition recipe**:
1. **Stage 1 (SFT):** Balanced 50/50 benign-vs-unsafe SFT (augmented via Nebius Token Factory) for format + tool routing.
2. **Stage 2 (Online GRPO + White Circle policy rewards):** Jointly optimize security-refusal policies and benign-helpfulness policies (`configs/gemma4-26b-a4b-policy-grpo-whitecircle.yaml`) so the model cannot game refusal rates.

---

## Flagship Research Track: Edit-Then-Target RL (`experiments/edit_then_target_rl/`)

Combines **Stage-1 zero-gradient / surgical weight-space pre-edits** (emulating MIT CSAIL's [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)) with **Stage-2 targeted Halo GRPO** restricted strictly to localized + 1-shot Fisher/reward-gradient weight subsets.

### Why Both Halves Together?
- **Manual weight edit unlocks RL:** Removing structural blockers (thinking/repetition loops, over-refusal, format drift) before step 0 eliminates GRPO's all-zero-reward dead groups.
- **Targeted RL hardens the manual edit:** `arXiv:2611.26841` and MIT CSAIL's Gemma 3 270M control show zero-gradient edits degrade under prompt-wording shifts and cannot add missing reasoning; restricting Halo RL updates to the top 5–25% localized modules hardens generalization while preserving the rest of the model.
- **Fresh empirical validation:** `arXiv:2611.33133` (*A Bias in Disguise*, Nov 24, 2026) independently showed combining a mid-layer weight-space edit with a half-strength sparse early-layer gradient edit boosted MATH-500 by **+10.1 pp (`p = 3e-9`)** with **50% fewer tokens**.

### Runnable Ablation Variants (`experiments/edit_then_target_rl/configs/`)

| Arm | Variant | Paper Basis | Halo Mechanism |
|---|---|---|---|
| **V0** | `arm0_full_grpo.yaml` | Baseline | All-layer LoRA GRPO |
| **V2** | `arm2_targeted_grpo_only.yaml` | `2607.01232`, `2610.11620`, `2609.36262` | Bare YAML regex `lora_target_modules` on top-25% localized layers |
| **V3** | `arm3_edit_then_target_grpo.yaml` | **Flagship** (WSP + Fisher/Contrastive RL) | Stage-1 surgical pre-edit → Stage-2 localized Halo GRPO (`weight_decay: 0.0`) |
| **V4** | `arm4_dual_edit_synergy_grpo.yaml` | `2611.33133` (*A Bias in Disguise*) | Mid-layer contrastive edit + half-strength early-layer (0–2) edit → Targeted RL |
| **V5** | `arm5_action_surgical_partial_ft.yaml` | `2611.32064` (*Fix How It Acts*) | `use_peft: false` + `unfreeze_layers_patterns` (zero LoRA BF16 merge-rounding loss; native MoE EP) |
| **V6** | `arm6_iterative_selfwrite_rl.yaml` | `2611.29513` (*SELFWRITE*) | Multi-round gate-verified weight edits → Targeted Halo GRPO |
| **V7** | `arm7_random_control_grpo.yaml` | `2602.01599` (*Multiple Ticket Hypothesis*) | Mandatory matched-budget random-layer control |

### All 12 Experiment Folders & Research Docs (`experiments/`)

| # | Folder & Docs | Focus | Key 2026 Papers | Priority |
|---|---|---|---|---:|
| **1** | [`experiments/seek_sniff_edit_rl/`](experiments/seek_sniff_edit_rl/README.md) | **Flagship Fork:** Closed-Loop Seek → Offset-Sniff → Gated Edit → Disjoint RL (V1–V4 + 3-domain benchmark) | WSP, `2611.29513`, `2611.33133` + Arm 3b discovery | **9.0/10** |
| **2** | [`experiments/head_and_node_sniff_rl/`](experiments/head_and_node_sniff_rl/README.md) | **NEW Fine-Grained Fork:** Individual Attention-Head (0 KV) + MLP False-Node Sniff & Double-Check RL | CircuitLens `2609.07183`, WSP, `2611.29513` | **8.5/10** |
| **2** | [`experiments/edit_then_target_rl/`](experiments/edit_then_target_rl/README.md) | **Flagship:** Manual/surgical pre-edit → Targeted Halo GRPO (V0–V7 + 6-arm micro benchmark) | WSP, `2611.33133`, `2611.32064`, `2611.29513`, `2609.36262` | **8.5/10** |
| **2** | [`experiments/insep_abliteration_defense/`](experiments/insep_abliteration_defense/README.md) | **Security Track:** INSEP Refusal-Direction Inseparability vs Abliteration | `2611.28314` | **8.5/10** |
| **3** | [`experiments/judge_loophole_audit/`](experiments/judge_loophole_audit/README.md) | **Pre-RL Gate:** Exposure Audit predicting LLM-Judge Reward Hacking | `2611.29651`, `2611.32037` | **8.5/10** |
| **4** | [`experiments/fisher_sparse_rl/`](experiments/fisher_sparse_rl/README.md) | 1-Shot Diagonal Fisher + SNIP Top-K Module RL (6 B/param mem savings) | `2609.36262`, `2609.15064`, `2602.01599` | **8.0/10** |
| **5** | [`experiments/verifiable_on_policy_distillation/`](experiments/verifiable_on_policy_distillation/README.md) | VOPD + Token Factory Teacher (works on Nemotron 3.5 & Mistral 4) | `2611.30325` (Stanford), `2611.32515` | **8.0/10** |
| **6** | [`experiments/multi_turn_rlvb_and_budget/`](experiments/multi_turn_rlvb_and_budget/README.md) | Multi-Turn Verifiable Beliefs (RLVB) + Anti-Budget-Internalization RLVR | `2611.32884`, `2611.26172`, `2611.32013` | **8.0/10** |
| **7** | [`experiments/grail_orth_subspace_rl/`](experiments/grail_orth_subspace_rl/README.md) | GRAIL + GCPO Off-Principal Orthonormal LoRA Init (~$12–$25 pilot) | GRAIL (EMNLP '26), `2608.11674`, `2606.31813` | **7.5/10** |
| **8** | [`experiments/moe_expert_subset_rl/`](experiments/moe_expert_subset_rl/README.md) | MoE Expert-Routing Localization + Soft Router-L2 + RDC (`gpt-oss-20b`) | `2610.07332`, `2610.21682`, `2609.08115` | **7.5/10** |
| **9** | [`experiments/token_credit_saepo_rl/`](experiments/token_credit_saepo_rl/README.md) | U-Stats Jackknife Adaptive Group Size + SAEPO Token Credit Shaping | `2611.33064`, `2611.29936` | **7.5/10** |
| **10** | [`experiments/dead_head_nullspace_rl/`](experiments/dead_head_nullspace_rl/README.md) | QO-Only Zero-Init Dead-Head Recycling & Null-Space QO-LoRA (optional gate) | `2503.02659`, `2609.25618`, `2603.09616` | **4.0/10** |

---

## Repository Layout

```text
configs/
  smollm2-135m-policy-lora-cpu.yaml           # Verified CPU smoke-test SFT config
  gemma4-26b-a4b-policy-grpo-whitecircle.yaml # 3-GPU Online GRPO + White Circle rewards
  gptoss-20b-policy-sft-ep8.yaml              # 8-GPU single-node EP8 + grouped expert LoRA
  nemotron-3.5-lightning-policy-sft-ep8.yaml  # 8-GPU single-node EP8 full SFT
  skypilot-nebius-h200-b200.yaml              # SkyPilot launcher for Nebius Cloud
data/
  policy_train.jsonl                          # 160 training conversations
  policy_eval.jsonl                           # 40 held-out zero-overlap eval tasks
results/
  eval_summary.json                           # Verified base vs Halo LoRA eval metrics
scripts/
  build_dataset.py                            # Deterministic dataset generator
  preflight_check.py                          # Offline YAML + dataset overlap validator
  run_halo_cpu.py                             # Unmodified Halo SFT runner for CPU boxes
  evaluate.py                                 # Deterministic base vs LoRA evaluator
  launch_nebius_gpu.sh                        # Zero-waste Docker launcher for H100/H200/B200
  tokenfactory_generate_and_judge.py          # Nebius Token Factory data-gen & LLM judge
```

---

## Quickstart

### 1. Offline Preflight (Before Renting GPUs)
```bash
python scripts/build_dataset.py
python scripts/preflight_check.py
```

### 2. Expand Training Data with Nebius Token Factory
```bash
export NEBIUS_API_KEY=...
python scripts/tokenfactory_generate_and_judge.py generate \
  --count 100 --out data/tokenfactory_policy_train.jsonl
```

### 3. Launch on an NVIDIA H100/H200 or B200/B300 Node
```bash
# Hopper (H100 / H200)
./scripts/launch_nebius_gpu.sh configs/gptoss-20b-policy-sft-ep8.yaml hopper 8

# Blackwell (B200 / B300)
./scripts/launch_nebius_gpu.sh configs/nemotron-3.5-lightning-policy-sft-ep8.yaml blackwell 8
```

## License
Code, configs, and synthetic datasets in this repository are released under Apache-2.0, compatible with the White Circle Halo Competition Terms (§6.2).
