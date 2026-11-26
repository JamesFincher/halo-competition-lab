# Exp 151 — Cache-corrected depth re-ablation

Staged replication experiment, **not autoqueued**. It re-measures plain width-64 slider insertion depth at L20/L24/L27/L29 under cache-free reward generation and cache-free behavioral evaluation, across seeds 7/42/123.

Rationale: exp146 falsified exp91's L20-collapse/L27-win conclusion as a generation-cache artifact and showed cache-corrected L20 at 72.5/90/0 versus L27 at 45/100/100 in one seed. Exp150 first tests whether L20 is seed-robust. If exp150 is interpretable, activate this experiment by copying `scripts/cache_corrected_depth_reablation_staged.py` to `scripts/run_cache_corrected_depth_reablation.py`, compile-checking, full-smoking, committing, and pushing.

Fixed protocol: width 64, ordered 12-benign/12-unsafe prompt stream, 32 grouped REINFORCE steps, group size 4, lr 3e-3, zero-init alpha, no cache during generation/eval, continuation benchmark, and entropy/margin/length diagnostics.
