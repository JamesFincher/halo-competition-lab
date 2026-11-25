# Experiment 18: 7-Behavior Neural Fingerprint Atlas & Location Cartography

**Priority:** Mechanistic Cartography (**9.0/10**)  
**Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (30 layers, 46,080 MLP neurons, `d=576`)

## 7 Fingerprinted Behaviors
Using our fastest winning camera from Experiment 17 (**Direct Logit Attribution / Logit Lens — 0 backward passes**), we map the exact layer-by-layer and neuron-by-neuron location pattern of 7 distinct behaviors:

1. **True Safety Refusal** — refusing genuinely dangerous requests (`ref` vs `call`)
2. **False Over-Refusal (Paranoia)** — falsely refusing benign tool requests (`ref` vs `call`, on Halo SFT `checkpoint-36`)
3. **Strict JSON Syntax** — emitting raw `{` vs conversational prose (`Sure`)
4. **Sycophancy / Agreement Bias** — agreeing (`Yes`) vs pushing back (`No`) when the user states a false claim
5. **Verifiable Arithmetic** — emitting the right sum vs a close distractor
6. **Factual Recall** — retrieving the right capital/fact vs a distractor
7. **Hedging / Verbosity** — emitting filler (`As` / `Sure`) vs direct answers

## Outputs
- `scripts/run_behavior_atlas.py` — Fingerprints all 7 behaviors, computes depth center-of-mass, peak layers, top-5 signature neurons, and 7×7 circuit-overlap matrix (cosine & top-64 neuron Jaccard), and renders `results/behavior_fingerprint_atlas.svg`.
- `results/behavior_fingerprint_atlas.svg` — Standalone 7×30 layer fingerprint heatmap + 7×7 circuit-collision matrix.
- `results/BEHAVIOR_ATLAS_REPORT.md` — Full location map and cross-talk findings.
