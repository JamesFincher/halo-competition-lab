# Experiment 129 — Sparse Validated-Edit Strength Response

**Single controlled variable:** projection strength: 0.50, 0.70, 0.90, or 1.20. Edit count is fixed at 16 columns per layer (32 total across layers 29+28), the sparse operating region indicated by exp85/exp98.

**Why:** exp95's strength sweep was measured at 64 columns per layer, so strength and edit count were confounded across experiments. This sweep isolates strength at the sparse operating point.

**Evaluation:** before/after 40-task heldout behavioral eval plus before/after `scripts/real_benchmark.py`; TransformerLens layer geometry probe. Results are written with defensive JSON serialization.
