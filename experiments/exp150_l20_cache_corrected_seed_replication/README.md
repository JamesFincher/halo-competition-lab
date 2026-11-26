# Experiment 150 — cache-corrected L20 width-64 seed replication

Exp146 reversed the depth interpretation: legacy-cache L20 collapsed behaviorally and produced artifactual benchmark gains, while cache-corrected L20 reached **72.5 / 90 / 0** at seed 42. This experiment isolates seed robustness only.

Fixed across seeds 7/42/123: L20 insertion, width 64, zero-init identity, slider-fork grouped REINFORCE, 32 steps, group size 4, lr 3e-3, identical ordered prompt stream, `use_cache=False` for reward generation and behavioral evaluation, 48-token behavioral eval. Measures behavioral eval, legacy benchmark, continuation benchmark, and the exp146 flattening/length diagnostics.

Output: `results/l20_cache_corrected_seed_replication_summary.json`.
