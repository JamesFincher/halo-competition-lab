# Discovery State — Halo Competition Lab (as of 2026-11-26)
Condensed evidence base for hypothesis generation. Base model: SmolLM2-135M-Instruct (SFT). Behavioral eval: 40-item exact-compliance / unsafe-carryout / benign-false-refusal. Capability: continuation-token benchmark (sum/token/byte-normalized MMLU/ARC-Easy/HellaSwag).

## VALIDATED POSITIVES
1. Ultra-sparse surgical edit (16 cols/layer, strength 0.9, output-side projection): 80/93.3/0, reproducible across doses 8-16 cols. exp85 variant hit 82.5/93.3/0 with 32 total edited coordinates. Behavioral control is essentially SOLVED at 135M — the champion recipe.
2. H22 composition (ultra-sparse edit + post-edit KL-anchored complementary MLP-only LoRA L24-27): all 3 seeds EXACTLY 82.5/93.3/0 — first zero-variance result ever. Exported merged artifact matches adapter. BUT random-layer LoRA control also hits 82.5 — gain is generic adaptation, not layer-specific repair. Ceiling not broken.
3. Cache-corrected inserted slider attention heads steer behavior: L20 width64 72.5/90/0 single seed. Heads train in ~32 steps.
4. Sequential training beats joint training for multi-head banks (70/90/0 vs 0/0/0 collapse) — destructive interference is an optimization phenomenon, not architectural.
5. Width-16 heads eliminate false refusals selectively; positive-gate mechanism kills ALL false refusals at zero exact-compliance cost.

## FALSIFIED / NEGATIVE (high confidence)
1. Capability transfer via inserted heads at 135M: FALSIFIED across every objective family — full-text causal, answer-upweighted, answer-only, benchmark-format-matched, contrastive choice, budgets 32-512 steps. Validation loss improves (-0.27 to -0.47) but continuation-benchmark transfer is exactly zero everywhere. Dose-response: more capability training actively destroys behavior (37.5% exact @32steps -> 20% @512steps). Capability and behavior objectives directly interfere.
2. All blending architectures lose to the plain sparse edit: routed width-mixture (25/96.7/100), capability×behavior blend (40/100/90), cross-layer cascade (37.5/100/80), subtractive gates (null). Complexity doesn't pay at this scale.
3. RL stacked on edits consistently fails (GRPO, vector RL, edit-then-RL) — collapses to over-refusal or no gain.
4. Convergent ~80-82.5% exact-compliance ceiling across 4 independent method families (sparse edit, GRPO, vector RL, H22) — the ceiling is a property of the model or eval, not the method.
5. Keyed/warm GQA composition heads collapse. Merge-aware rounding training is performance-negative.

## MEASUREMENT FACTS (critical interpretation context)
- Legacy real_benchmark scorer is structurally length-biased: uniform-logit noise scores MMLU26/ARC32/HellaSwag26 vs real baseline 26/38/32 — artifact floor nearly equals baseline. ALL legacy benchmark deltas are noise. Continuation-token scoring is the only credible capability measurement.
- KV-cache generation artifact: slider heads saw T=1 after first token (no backward attention) in all pre-fix experiments — caused both behavioral collapse AND benchmark inflation via logit flattening. L20 "cursed layer" inversion was entirely this artifact; cache-corrected L20 is strong (72.5/90/0) and may beat L27.
- Baseline continuation scoring: MMLU sum/token/byte=10/30/30, ARC 60/50/50, HellaSwag 30/20/10 — normalization spread matters.

## OPEN QUESTIONS
- Is the 82.5% ceiling a model-capacity property or an eval-saturation property? Untested.
- Does capability transfer become possible at 1B+ scale (representational headroom hypothesis)?
- Is L20 robustly the better insertion layer (seed replication exp150 running)?
- Why does generic post-edit adaptation (random-layer LoRA) match targeted repair?
- Can the positive-gate false-refusal mechanism compose with the sparse edit for a free +benign improvement?
- Seed fragility: slider single-head outcomes vary wildly; H22 composition eliminates it — mechanism unknown.

## INFRASTRUCTURE
- H100 SXM pipeline runs 16 experiments in parallel; ~5-25 min per 135M experiment. Feeder/hypothesizer/tracker agent flywheel generates, ranks, grades experiments continuously. 82+ artifacts on HF dataset heartandkarma/halo-competition-lab-results. Competition target: White Circle Halo (aligned training on halo:hopper docker image).
