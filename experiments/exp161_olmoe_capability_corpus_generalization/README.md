# Staged Exp161: capability corpus generalization ablation

Intentionally inactive (`staged_*.py`) until exp158's source/format mechanism results are
available. Under fixed random-expert adapter selection, initialization, rank, optimizer, and
update budget, compares:

- original benchmark-train mixture (`capability_train_scaled.jsonl`),
- benchmark-independent general corpus (`capability_train_general_v1.jsonl`), and
- a 50/50 benchmark+general mixture.

Primary readout: held-out transfer benchmarks (ARC-Challenge, OpenBookQA, Winogrande,
TruthfulQA MC1), with in-domain continuation benchmarks and entity-disjoint behavior v2 as
secondary/collateral measurements.
