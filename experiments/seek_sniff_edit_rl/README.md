# Experiment 11: Seek → Offset-Sniff → Surgical Edit → Disjoint RL Loop

**Priority:** #1 Novel Flagship Fork (**9.0/10** — directly extends our verified Arm 3b micro-model win)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (verified live on CPU) → `Qwen/Qwen3-4B` → `unsloth/gpt-oss-20b-BF16`

## Core Idea (Plain English)
Instead of editing weights blindly once, run a **closed diagnostic loop** that isolates *only* the buggy ("afflicted") weights using **minimal-pair offset prompts**:

1. **SEEK:** Run diagnostic prompts on the current model to catch exact failures (e.g., safe commands falsely blocked, or subtle attacks missed).
2. **SNIFF (Offset Minimal Pairs):** For each failure, generate a matched **offset prompt** that flips *only* the real safety boundary while keeping surface wording almost identical (e.g., `read_file("docs/readme.md")` vs `read_file(".env")`; `SELECT count(*) FROM orders` vs `DROP TABLE orders`). Compare which layers/neurons light up on the **false trigger** vs the **true safety trigger**:
   \[
   \text{afflicted\_dir}_l = \big(\mu_{\text{false\_fail}, l} - \mu_{\text{benign\_pass}, l}\big) - \gamma \cdot \text{proj}_{\mu_{\text{true\_unsafe}, l} - \mu_{\text{benign\_pass}, l}}\big(\mu_{\text{false\_fail}, l} - \mu_{\text{benign\_pass}, l}\big)
   \]
   Subtracting the orthogonal projection onto the **real safety direction** isolates *only* the false-paranoia/buggy component so surgical edits don't break real safety!
3. **GATED SURGICAL EDIT:** Project out the afflicted direction from the top-K MLP `down_proj` columns (`~0.01%–0.03%` of weights) and **accept only if a fast validation gate improves** (SELFWRITE-style accept/reject, `arXiv:2611.29513`).
4. **DISJOINT TARGETED RL:** Lock the surgically edited layers, **re-anchor the KL reference to the edited checkpoint**, and run gentle GRPO (`weight_decay=0.0`) on the **complementary localized layers** (our verified Arm 3b rule!).
5. **LOOP:** Re-seek remaining errors on the updated model and repeat!

## 4 Forked Variants Tested Live
- **Fork V1 (`offset_sniff_edit_only`):** 2 rounds of pure Seek → Offset-Sniff → Gated Surgical Edit (0 gradient steps).
- **Fork V2 (`single_pass_sniff_edit_rl`):** 1 round of Orthogonal Offset-Sniff Edit → Disjoint Re-Anchored RL.
- **Fork V3 (`closed_loop_sniff_edit_rl` — Flagship Loop):** 2 full closed-loop rounds of `Seek → Offset-Sniff → Gated Edit → Disjoint RL → Re-Seek → Re-Sniff → Gated Edit → Disjoint RL`.
- **Fork V4 (`dual_layer_sniff_edit_rl`):** Combines Orthogonal Offset-Sniff with `arXiv:2611.33133` (*A Bias in Disguise* — half-strength early-layer `0–1` sniff edit + late-layer sniff edit → disjoint middle-layer RL).

## Files
- `scripts/run_seek_sniff_edit_loop.py` — Runs all 4 forks end-to-end with per-round sniff diagnostics, per-step JSONL telemetry, and 40-task unseen held-out evaluations.
- `configs/qwen3-4b-seek-sniff-edit-rl.yaml` — Halo v1.3.0rc1 GPU config for scaling the winning fork to `Qwen3-4B`.
- `results/SEEK_SNIFF_EDIT_REPORT.md` — Tracked live benchmark results, per-round sniff logs, and scoreboard.
