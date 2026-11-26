# exp166 Qwen3 mixed-batch interleaving

STAGED conditional follow-up, intentionally not autoqueued (`staged_*.py`). Activate after exp164 if the 25% replay point (or another representable dose) shows a meaningful behavior cost.

Question: at exactly matched behavior/capability example counts, does mixing behavior and capability examples within every batch preserve behavior better than exp152-style alternating homogeneous batches while retaining capability transfer?

At defaults (300 steps, batch 8, replay every 4), the alternating reference sees 1,800 behavior + 600 capability examples. The mixed arm sees 6 behavior + 2 capability examples in every batch, also 1,800 + 600. This isolates interleaving granularity rather than dose.

Measurement: v2 entity-disjoint behavior plus in-domain and held-out transfer continuation benchmarks.
