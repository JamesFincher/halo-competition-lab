# Experiment 147 — corrected capability/behavior additive-union replication

The landed exp135 artifact is incomplete: it started before the full-strength additive-union control was added. That omission matters because the original convex interpolation cannot distinguish “the two head functions interfere” from “a convex mixture must dilute one objective.”

This corrected replication reruns the same cache-corrected sequential training and blend family, including:

1. separately trained capability-only and behavior-only heads,
2. full-strength additive union control,
3. equal convex blend,
4. learned scalar convex blend.

Same width-64/L27 zero-init architecture, seed, split-decontaminated capability corpus, behavioral grouped REINFORCE, continuation benchmark, and before/after behavioral eval. Output: `results/additive_capability_behavior_union_summary.json`.
