# Exp160: OLMoE checkpoint transfer and practical export evaluation

No-training evaluation using the completed exp154 adapter checkpoints. For selective,
random, and diffuse expert-adapter arms:

1. measures in-domain continuation and held-out transfer benchmarks before merging;
2. merges the low-rank deltas into expert `down_proj` weights and zeroes the external bank;
3. repeats both benchmark suites after merging; and
4. reports pre-vs-post score movement as practical export fidelity.

This complements exp157/exactness and exp159/precision diagnosis: even if bf16 export is
not bitwise exact, the operationally important question is whether benchmark-level behavior
is preserved and whether exp154 capability movement generalizes beyond train-split benchmark
formats.
