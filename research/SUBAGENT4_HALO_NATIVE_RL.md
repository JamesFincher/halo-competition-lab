# SUBAGENT 4 — Halo-Native Surgical-Edit + Selective-Parameter RL

**Date:** 2026-11-26  
**Scope:** White Circle Halo `v1.3.0rc1`, pinned at `3a21b4dbdb56cb840d42a59e98e40f967a1d2807`; TRL `1.14.2`; PEFT `0.18.1`; Transformers `5.20.2`.  
**Question:** how much of the verified micro-model loop — surgical-edit layers `[29,28]`, freeze those layers, train complementary localized layers, KL anchored to the post-edit checkpoint — transfers to the real Halo GRPO pipeline?

## 0. Verdict

The core loop **does transfer**, and most of it is native Halo configuration rather than new Halo code:

1. **Save the surgical edit as a merged HF checkpoint** outside Halo.
2. Point `model_name_or_path` at that checkpoint. This makes the post-edit model the policy initialization **and the KL anchor**.
3. Choose one of two native selective-training paths:
   - **Targeted LoRA (recommended first competition path):** `use_peft: true` plus a **bare YAML regex** in `lora_target_modules` matching only complementary layers. Every base weight — including surgically edited layers `[29,28]` — is frozen automatically.
   - **Targeted partial full fine-tuning (merge-artifact-free path):** `use_peft: false` plus Halo-native `unfreeze_layers_patterns` for complementary layers. Edited layers remain frozen because everything not matched is frozen.
4. Set `weight_decay: 0.0` and reduce Stage-2 LR about 3–4×, matching the micro-model rule.
5. For closed-loop rounds, orchestrate successive edit → export → Halo-launch rounds externally. Do **not** try to perform seek/sniff/gated editing inside the distributed GRPO step unless a later experiment specifically requires it.

The largest practical gaps are not layer freezing or KL anchoring. They are: (a) Halo online GRPO has no YAML plug-in for our exact JSON policy verifier, (b) the lab's JSONL is SFT-shaped rather than online-RL-shaped, and (c) several existing lab YAML/launcher spellings are not valid Halo `v1.3.0rc1` (`dataset_name`, `rollout_adapter_sync` on online GRPO, `halo launch grpo`) and must be corrected before any H100 run.

---

## 1. Exact mechanism inventory

### 1.1 Freezing specific layers/parameters

| Mechanism | Stock TRL `GRPOTrainer` | Halo `v1.3.0rc1` | Consequence for us |
|---|---|---|---|
| Manual `requires_grad=False` | Supported indirectly. GRPO accepts an instantiated model; Transformers `Trainer.create_optimizer` puts only parameters with `p.requires_grad` in optimizer groups. | Same, but distributed trainers also build optimizer groups from trainable parameters only. | A programmatic hook can freeze arbitrary tensors, but YAML is preferable where Halo exposes one. |
| YAML freeze/unfreeze patterns | **No stock TRL config field.** | **Native:** `unfreeze_layers_patterns` and `freeze_layers_patterns` in `CommonScriptArguments`. `unfreeze` first freezes the whole model, then unfreezes matching **module names**; `freeze` is then applied to matching **parameter names**. Patterns are `fnmatch`; unmatched patterns raise. Layer-indexed patterns are rejected under pipeline parallelism. | Exact fit for partial full-parameter RL. Use `unfreeze_layers_patterns` for complementary layers; edited layers stay frozen. |
| PEFT/LoRA base freezing | Native. `get_peft_model` freezes base weights and trains adapters. | Native, plus EP-aware grouped expert adapters. Adapter runs **reject** `unfreeze_layers_patterns`/`freeze_layers_patterns`, because the base is already frozen and patterns would be meaningless. | Targeted LoRA is the cleanest way to freeze edited layers: omit them from `lora_target_modules`. |
| Per-coordinate sparse masks inside a trainable tensor | No general config. | No general config. | Not needed for layer-disjoint RL. If future Fisher masks are coordinate-level rather than module-level, a custom gradient hook/optimizer is required. |

Primary source references: Halo `halo/model_loading/parallel/freeze_patterns.py`, `halo/model_loading/parallel/peft_setup.py`, `agent-docs/reference/configuration-reference.md` (`unfreeze_layers_patterns` row), Transformers `trainer.py` `create_optimizer`.

### 1.2 Re-anchorable KL reference models

TRL `1.14.2` behavior in `GRPOTrainer`:

- `beta == 0`: no reference model and no KL term.
- `beta != 0` with a PEFT model: no separate reference model. TRL computes reference log-probs from the policy with adapters disabled (`use_adapter(..., None)`), unless retraining a pre-existing adapter, in which case TRL can maintain a frozen `ref` adapter copy.
- `beta != 0` without PEFT: TRL conceptually needs a frozen reference model. Halo intercepts TRL's construction and supplies its own correctly loaded copy rather than letting TRL build a poorly matched fp32/default-revision copy.

Halo behavior:

- `load_reference_model_for_on_policy_grpo` loads `model_config.model_name_or_path` with matching dtype, revision, attention implementation, tokenizer/vocabulary state, GPT-OSS sink policy, and buffer repair.
- The reference is frozen, eval-mode, and unparallelized. It is a dense replica per rank; Halo preflights GPU/host memory and warns under EP/ETP/TP.
- With attention LoRA, Halo holds no second copy: the adapter-disabled policy **is** the reference.
- With expert-only native EP LoRA there is no `PeftModel`, so Halo treats the run like full fine-tuning and holds an explicit frozen reference.
- On resume, Halo loads the reference from `model_name_or_path`, not from the resumed trained checkpoint. That is exactly what we want if `model_name_or_path` is the Stage-1 edited checkpoint.

**Critical mapping:** because both policy initialization and reference resolution derive from `model_name_or_path`, setting it to `checkpoint_edited` re-anchors KL to the post-edit model. No extra YAML key exists or is needed.

`sync_ref_model` is not the right mechanism. It periodically mixes the **current policy** into the reference (`π_ref = απθ + (1−α)π_ref_prev`), is incompatible with PEFT in TRL, and cannot mean “re-anchor to this externally edited checkpoint.” Use per-round job restarts for the seek/sniff loop.

Primary source references: TRL `trl/trainer/grpo_trainer.py` reference-model construction and reference log-prob computation; Halo `halo/model_loading/parallel/frozen_models.py`, `halo/trainers/grpo/on_policy.py`, `agent-docs/training-methods/grpo/online.md` §“GRPO objective”.

### 1.3 Parameter-subset optimization and `weight_decay=0`

Transformers `Trainer.create_optimizer` creates decay/no-decay groups only from parameters with `requires_grad=True`. Halo's `decay_groups` does the same. Therefore **whole frozen layers are not in the optimizer at all** and cannot be weight-decayed.

The lab's `weight_decay: 0.0` rule is still correct and should be treated as mandatory for this research track:

- Halo commonly auto-selects `AdamWBF16` under `bf16: true` and stock AdamW.
- In `halo/optimizers/adamw_bf16.py`, the update first applies decoupled decay (`p = p * wd_factor`) and then the Adam correction. A parameter that is in the optimizer but receives a zero or masked gradient still decays.
- Whole-parameter freezing avoids this, but sparse/uneven training, expert routing, dead modules, gradient masking, or future Fisher-coordinate masks can all produce zero gradients in tensors that remain optimizer-owned.
- Since selective RL is explicitly testing localized/sparse updating, `weight_decay: 0.0` removes a confounding silent drift term.

`weight_decay: 0.0` is not a hack here; it is part of the experimental control.

### 1.4 LoRA versus targeted full-parameter training

| Property | Targeted LoRA | Targeted partial full fine-tune |
|---|---|---|
| Halo YAML | `use_peft: true`, bare regex `lora_target_modules` | `use_peft: false`, `unfreeze_layers_patterns` |
| Edited layer freezing | Automatic: base frozen; edited layers omitted from targets | Automatic: unmatched layers stay frozen |
| KL anchor | Adapter-disabled policy = edited base; no second model copy | Explicit frozen copy of edited checkpoint per rank |
| Memory | Lowest: no optimizer state/grads for base | Higher: gradients + optimizer state for unfrozen full tensors |
| MoE expert coverage | Attention PEFT plus native grouped expert LoRA under EP | Native full tensors, including EP experts |
| Merge-rounding risk at export | Adapter must eventually be merged/exported; rollout sync itself is exact out-of-place folding but final BF16 merge can round small deltas away | No adapter merge: trained base weights are already the artifact |
| Parallelism restrictions in online GRPO | Online GRPO itself refuses CP/PP. LoRA is additionally refused under TP/EP+TP; attention LoRA works under EP, EP+ETP and pure ETP, while native expert LoRA requires EP and is refused under ETP. | Native full/partial training follows online GRPO's FSDP2/EP/TP/EP+TP/EP+ETP modes, but `beta != 0` adds an unparallelized reference replica per rank; memory is often decisive. |
| Best use | First H100 pilot; memory-safe; easy ablations | Final action-surgical arm and MoE expert-targeted arm |

Important Halo LoRA details:

- `lora_target_modules` as a **YAML list** names modules by trailing path segment (`q_proj`, `down_proj`, etc.).
- `lora_target_modules` as a **bare YAML string** is PEFT's regex form, fullmatched against full module paths. This is the layer-selective mechanism.
- Regex metacharacters inside a YAML list are rejected by Halo because they would silently match nothing in PEFT.
- TRL/Halo do **not** expose PEFT `layers_to_transform` through `ModelConfig`; use the regex.
- Expert names (`gate_proj`, `up_proj`, `down_proj`, `gate_up_proj`, `experts`) are peeled out and routed to Halo's native grouped expert adapters under EP. `lora_target_parameters` is rejected on EP models.

### 1.5 Rollout weight synchronization

Halo online GRPO uses server-mode vLLM only. The trainer pushes weights to the server over NCCL.

- For attention/native expert LoRA, Halo folds adapters **out of place** into the tensor being transmitted. The trainer's frozen base is never written, avoiding in-place BF16 merge/unmerge corruption.
- Online GRPO has no `rollout_adapter_sync` config field; LoRA weight synchronization is the folded-weight path. `rollout_adapter_sync` belongs to async/environmental GRPO, where `false` likewise selects folded-weight pushes and `true` is the separate SGLang adapter-serving path.
- `use_vllm: true` and `vllm_mode: server` must both be explicit in configs. TRL defaults `use_vllm` to false in this version; Halo's trainer requires server mode.

### 1.6 Relevant literature check

Direct arXiv lookup confirms the lab's core 2026 research anchors exist and are on-topic:

- `arXiv:2611.26841` — weight-space programming writes in-context learning and can match fine-tuning.
- `arXiv:2611.33133` — tiny gradient edits complement a frontier model's weight edit; supports dual-edit synergy.
- `arXiv:2611.32064` — surgical fine-tuning of frontier LLMs, emphasizing action-level corrections.
- `arXiv:2611.29513` — SELFWRITE/recursive self-editing with gate verification.
- `arXiv:2609.36262` — post-training parameter-update sparsity through a Fisher lens.
- `arXiv:2607.01232` — one-layer/single-layer RL can match full-parameter RL.
- `arXiv:2610.11620` — layer-selective fine-tuning for capability retention.
- `arXiv:2602.01599` — random sparse subnetworks can suffice for RLVR; validates the need for a random-layer control.
- Additional adjacent work: `arXiv:2601.06702` (GRASP LoRA / GRPO-guided adapter sparsity) and `arXiv:2604.15351` (gradient-guided layer selection for LoRA).

These papers motivate selective RL; they do not replace the Halo-specific implementation constraints above.

---

## 2. Step-by-step mapping of the surgical loop to Halo

### Phase A — outside Halo: diagnose, edit, gate, export

Input from the micro-model loop:

- edited layers: `E = [29, 28]`
- complementary localized layers: `C = [0, 1, 11, 19, 20, 27]`
- post-edit checkpoint: `outputs/seek_sniff_edit/round1_edited_checkpoint`

Competition implementation:

1. Run seek/sniff/localization on the current merged checkpoint.
2. Apply the candidate surgical edit.
3. Gate it on a held-out dev slice.
4. If accepted, save a **merged, servable HF checkpoint** containing:
   - `config.json`
   - tokenizer files
   - sharded safetensors weights
   - any remote-code files required by the model family
5. Record `E`, `C`, edit strength, gate metrics, and the checkpoint path in a round manifest.

Do not leave the model as “base + ad-hoc tensor patch files.” Halo must see a normal checkpoint directory.

### Phase B — prepare online-RL rows

Halo online GRPO expects rows with separate `prompt` and `answer` columns, not SFT `messages` containing the gold assistant response. If the assistant answer remains inside the prompt, the rollout is conditioned on the solution and the experiment is invalid.

Convert each lab SFT row:

```json
{"messages": [system, user, assistant]}
```

into:

```json
{
  "prompt": [system, user],
  "answer": "<the assistant JSON string>"
}
```

Suggested output: `data/policy_train_online.jsonl` and `data/policy_eval_online.jsonl`.

This is a required lab preprocessing step, not optional hygiene.

### Phase C — Stage-2 targeted GRPO: two native configurations

#### Path 1 — targeted LoRA, recommended first

The complementary-layer regex for the micro-model example is:

```yaml
lora_target_modules: ".*\\.layers\\.(0|1|11|19|20|27)\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"
```

Why this implements “freeze `[29,28]`”:

- PEFT freezes every base parameter.
- Only modules fullmatched by the regex receive trainable adapters.
- Layers `28` and `29` receive no adapters, so neither their edited `down_proj` columns nor any other edited weight can move.
- At `beta > 0`, reference log-probs are computed with adapters disabled, i.e. exactly the post-edit base checkpoint.

Use this path for the first real Halo validation because it minimizes memory and uses Halo's tested LoRA weight-sync path.

#### Path 2 — targeted partial full fine-tuning, action-surgical arm

For a dense text model whose module tree uses `model.layers.N...`, the conceptual YAML uses exact decoder-container names (`module.parameters()` is recursive, so no wildcard is needed):

```yaml
use_peft: false
unfreeze_layers_patterns:
  - "model.layers.0"
  - "model.layers.1"
  - "model.layers.11"
  - "model.layers.19"
  - "model.layers.20"
  - "model.layers.27"
weight_decay: 0.0
```

This freezes the whole model, then unfreezes only the listed modules and their descendants. Layers `[29,28]` remain frozen.

Caveats:

- `unfreeze_layers_patterns` matches full **module** names, not parameter names.
- A pattern that matches nothing raises. Do not guess prefixes on multimodal/MoE checkpoints.
- Gemma multimodal trees may use a different prefix (for example `model.language_model.layers...`); inspect the live model's `named_modules()` in the exact Halo image before generating the YAML.
- Full fine-tuning plus `beta != 0` requires a full frozen reference copy on every rank. On large models this can be the decisive memory cost; targeted LoRA avoids it.

A hybrid protection pattern is possible when a broad unfreeze pattern accidentally covers edited layers: apply `freeze_layers_patterns` afterward against full parameter names. In the recommended disjoint design this should not be necessary.

### Phase D — closed-loop rounds

For round `r`:

1. Start from merged checkpoint `M_r`.
2. Seek/sniff → candidate edit layers `E_r`.
3. Gate edit.
4. Save accepted checkpoint `E(M_r)`; if rejected, `E(M_r)=M_r`.
5. Generate a Halo YAML whose:
   - `model_name_or_path` is `E(M_r)`
   - target layers exclude every surgically edited layer in `E_0..E_r`
   - LR is reduced versus unedited RL
6. Launch Halo online GRPO.
7. Export/merge the trained result as `M_{r+1}`.
8. Repeat seek/sniff on `M_{r+1}`.

External orchestration is the correct minimal hook. A callback inside GRPO would have to coordinate generation diagnostics, gated evaluation, collective weight mutation, vLLM resynchronization, reference replacement, and optimizer state invalidation across EP/FSDP ranks. That is a large distributed-systems change for zero algorithmic benefit over checkpoint-boundary rounds.

---

## 3. Gaps and minimal workarounds

### Gap 1 — exact policy verifier is not YAML-native

Halo online GRPO's built-in local graders are:

- `accuracy`: strict last `\boxed{...}` equality
- `format`: regex match

Our policy-router verifier needs JSON parsing plus exact `action`, `tool`, and `policy_code` scoring. `format` can detect JSON-ish shape but cannot verify semantic correctness. White Circle policy terms measure policy violations but do not reproduce the deterministic micro-model reward.

**Minimal workaround:** add one small local reward hook around the online script, not a Halo redesign. In a lab-owned copy/wrapper of `scripts/training/grpo/online.py`, insert this immediately after `build_trl_reward_functions(...)` returns:

```python
import json

from halo.rewards.graders.verifiable import completion_text


def parse_policy_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def exact_policy_json_reward(completions, answer, **kwargs):
    rewards = []
    for completion, target_text in zip(completions, answer, strict=True):
        pred = parse_policy_json(completion_text(completion))
        target = json.loads(target_text)
        if pred is None:
            rewards.append(0.0)
        else:
            shape_ok = {"action", "tool", "policy_code", "reason"}.issubset(pred)
            rewards.append(
                0.20 * shape_ok
                + 0.40 * (pred.get("action") == target["action"])
                + 0.20 * (pred.get("policy_code") == target["policy_code"])
                + 0.20 * (pred.get("tool") == target["tool"])
            )
    return rewards


reward_funcs.append(exact_policy_json_reward)
reward_weights.append(1.0)
```

`completion_text` handles both string completions and TRL's message-list form; `answer` reaches the function because `process_for_rlvr` normalizes the configured answer column and the config keeps `remove_unused_columns: false`. Keep White Circle terms as additional external policy rewards. This is tens of lines in the lab, not a Halo upstream patch.

Launch the modified copy directly inside the Halo image (for example `torchrun --nproc_per_node=1 /lab/scripts/training/grpo/online_policy.py <config>`). `halo launch grpo/online` resolves the toolkit's own installed script and will not pick up a lab copy merely because it exists in `/lab`.

Alternative: write a custom async-GRPO environment. That is cleaner for multi-turn tool use but much heavier for the immediate single-turn policy-router pilot.

### Gap 2 — no separate reference-path YAML field

There is no `reference_model_name_or_path` for online GRPO. This is acceptable for the current algorithm because the desired reference **is** the policy initialization checkpoint.

If a future arm needs policy init `A` but KL reference `B`, minimal code is required:

1. load `B` with `load_frozen_reference_model`;
2. pass it as `ref_model` to `DistributedOnlineGRPOTrainer` for a non-PEFT run; or
3. for PEFT, create/copy a frozen `ref` adapter or explicitly score reference log-probs from `B` through a trainer modification.

Do not use `sync_ref_model` as a substitute; it tracks the moving policy, not an external surgical anchor.

### Gap 3 — dynamic in-run re-editing/re-anchoring

Native Halo assumes one static reference anchor per run. The closed loop crosses that boundary every round.

**Workaround:** external round orchestration with checkpoint exports and fresh Halo launches. This is robust, auditable, parallelism-safe, and competition-friendly. It also makes each round independently resumable.

### Gap 4 — layer-pattern generation is model-specific

LoRA regexes are portable where module suffixes are stable, but full-FT `fnmatch` patterns depend on exact module-tree prefixes. Multimodal and MoE model families differ.

**Workaround:** generate YAML after loading/probing the target model in the Halo image. The generator should print candidate module paths and fail if no path matches each selected layer. Never hand-maintain both `model.layers` and `model.language_model.layers` patterns blindly: a no-match pattern raises by design.

### Gap 5 — coordinate-sparse Fisher masks are not native

Halo supports module/parameter subset freezing and LoRA module targeting, not arbitrary bitmask optimization inside a tensor.

**Workaround options:**

1. Use layer/module-level top-k localization (native; recommended first).
2. Implement gradient hooks that zero masked coordinates after backward. This works mathematically but still allocates optimizer state for the whole tensor and requires `weight_decay: 0.0`.
3. For true memory savings at coordinate sparsity, implement a custom sparse optimizer/state representation — substantial work and not needed for the first competition transfer.

### Gap 6 — existing quickboot/config spellings are stale or invalid

Before GPU spend, correct these in the lab:

- Halo CLI method is `grpo/online`, not bare `grpo`: use `halo launch grpo/online <config>`.
- Halo's dataset field is `dataset`, not `dataset_name`.
- Online GRPO configs need `use_vllm: true` and `vllm_mode: server` explicitly.
- Online rows need `prompt`/`answer` or correct `prompt_field`/`answer_field`; current `messages` rows include the gold assistant and cannot be used directly.
- A local file path should be `./data/...` or absolute so it cannot be mistaken for a Hub id.
- The current experiment YAMLs omit reward terms/dataset shaping required for a meaningful online run.
- `configs/h100_quickboot_manifest.yaml` queues only GRPO jobs. It neither builds nor exports Stage-1 `checkpoint_edited`; that path is absent from a fresh clone, so the edit-then-target arm cannot start as written.
- The quickboot script never starts a vLLM server. Online GRPO requires a separate server process before the trainer preflight succeeds, so the manifest needs a server launch/teardown wrapper (serving the same edited checkpoint and using the Halo image's weight-transfer flags) in addition to the training jobs.
- The current 1×H100 quickboot shape is not a valid online-GRPO layout as written: Halo documents that vLLM must own GPUs no trainer rank uses. Use at least two distinct GPU devices with disjoint `CUDA_VISIBLE_DEVICES`, or move the server to a separate host. Sharing one GPU is a documented hang/contention mode, not a memory-tuning exercise; any MIG partition would need its own NCCL validation before being treated as two devices.

  If the competition budget truly allows only one GPU, the closest Halo-native fallback is an **offline-GRPO round-robin**: generate grouped completions in a separate inference phase, score them with the exact JSON/White Circle rewards, write `prompt`/`completions`/`rewards` rows, then train `scripts/training/grpo/offline.py` with the same selective LoRA/unfreeze configuration and `kl_beta` rather than online `beta`. That preserves selective-parameter training and post-edit anchoring, but it is off-policy within each round and is not evidence for the online-GRPO arm.

---

## 4. Draft competition-ready YAML sketch

This is a **Qwen3-4B pilot** online-GRPO config using targeted LoRA. It assumes:

- Stage-1 edited checkpoint exists at `outputs/seek_sniff_edit/round1_edited_checkpoint`.
- `data/policy_train_online.jsonl` has `prompt` and `answer` columns.
- A lab-owned exact-JSON reward hook is wired in addition to built-in terms.
- vLLM server is already running on GPUs no trainer rank uses, serving the same edited checkpoint with Halo's required NCCL weight-transfer flags.

```yaml
# seek-sniff Stage-2: disjoint targeted LoRA GRPO anchored to the post-edit checkpoint
model_name_or_path: outputs/seek_sniff_edit/round1_edited_checkpoint
dataset: ./data/policy_train_online.jsonl
test_size: 0.05
prompt_field: prompt
answer_field: answer
output_dir: outputs/seek_sniff_edit/halo_round1_disjoint_grpo

# --- selective parameterization ---
use_peft: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
# Bare YAML regex, not a list. Complement of surgically edited layers [28,29].
# Replace these layer IDs from the round manifest; do not hard-code micro-model IDs for a different model.
lora_target_modules: ".*\\.layers\\.(0|1|11|19|20|27)\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)$"

# --- online rollout ---
use_vllm: true
vllm_mode: server
vllm_server_host: 127.0.0.1
# --- GRPO objective ---
num_generations: 8
max_prompt_length: 1024
max_completion_length: 512
beta: 0.02
loss_type: dapo
scale_rewards: group
num_iterations: 1
epsilon: 0.2
temperature: 1.0

# KL/IS consistency: no nucleus truncation if sequence-level IS modes are used.
top_p: 1.0

# Gentler than unedited GRPO per Arm 3b; no decay-driven drift on selective params.
learning_rate: 1.5e-6
lr_scheduler_type: constant
weight_decay: 0.0

per_device_train_batch_size: 8
gradient_accumulation_steps: 4
max_steps: 100
bf16: true
seed: 42

# --- rewards ---
# Keep White Circle as an external policy term. Exact JSON correctness still needs the lab hook in §3 Gap 1.
rewards:
  - source: format
    name: json_shape
    weight: 0.2
    pattern: '\{.*"action".*"tool".*"policy_code".*"reason".*\}'
  - source: whitecircle
    name: policy
    weight: 0.8
    environment_id: REPLACE_WITH_WHITECIRCLE_ENVIRONMENT_ID
    max_concurrency: 16

remove_unused_columns: false
save_completions: true
log_completions: false
logging_steps: 1
logging_first_step: true
save_steps: 25
save_strategy: steps
report_to:
  - wandb
project_name: whitecircle-halo-competition
run_name: seek-sniff-r1-disjoint-lora-grpo
```

### Full-parameter alternative fragment

Replace the LoRA block with:

```yaml
use_peft: false
unfreeze_layers_patterns:
  - "model.layers.0"
  - "model.layers.1"
  - "model.layers.11"
  - "model.layers.19"
  - "model.layers.20"
  - "model.layers.27"
learning_rate: 5.0e-7
weight_decay: 0.0
```

Use this only after confirming exact module names in the target model and budgeting the extra full-reference copy. It is the preferred final arm when avoiding LoRA merge-rounding or when targeting native MoE expert tensors directly.

---

## 5. Concrete validation checklist before H100 spend

1. **Static config parse:** run the actual Halo image's parser on the YAML, not just the lab's YAML syntax checker.
2. **Dataset contract:** verify rows render as prompt-only conversations and `answer` contains the target JSON.
3. **Layer targeting probe:** print matched LoRA modules/counts on the target model; assert zero matches in edited layers.
4. **Reference probe:** first training step at initialization should have near-zero KL before any optimizer update; a large initial KL means the anchor is wrong.
5. **Frozen-weight probe:** hash or checksum surgically edited tensors before/after a short smoke run; they must be bit-identical for both LoRA and partial-FT paths.
6. **Weight-sync probe:** keep Halo's first-push parity enabled; investigate any trainer/server logprob mismatch before scaling.
7. **Random control:** run the matched-budget random-layer arm at identical steps, group size, LR, and adapter rank/full-parameter budget. Without it, the selective claim is not falsifiable.

## 6. Bottom line

Halo already provides the two decisive ingredients the micro-model win needs: exact layer-targeted training and correct post-edit KL anchoring. The clean competition implementation is therefore **checkpoint surgery outside Halo + generated selective Halo YAML + fresh GRPO launch per round**. The only mandatory lab-owned code beyond orchestration is the exact JSON policy reward hook and the dataset conversion; no deep modification of Halo's GRPO trainer is needed for the first transfer experiment.
