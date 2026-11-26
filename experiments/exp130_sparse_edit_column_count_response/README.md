# Experiment 130 — Sparse Validated-Edit Column-Count Response

**Single controlled variable:** edited columns per layer: 4, 8, 16, or 32 (8/16/32/64 total across layers 29+28). Strength is fixed at 0.90; direction measurement, seed/data, evaluation, and operator are fixed.

**Why:** exp98 tested only 32/64/128 columns per layer and favored the sparsest arm. exp85's strong 80–82.5% exact results used only 16 columns per layer (32 total), below exp98's lowest arm. This experiment fills the critical sparse dose region.

**Evaluation:** before/after 40-task heldout behavioral eval plus before/after `scripts/real_benchmark.py`; TransformerLens layer geometry probe. Results are written with defensive JSON serialization.
