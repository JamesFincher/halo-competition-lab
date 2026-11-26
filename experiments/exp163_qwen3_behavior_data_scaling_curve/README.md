# Exp163: Qwen3 behavior data-scaling curve

Exp152 showed scaled-corpus LoRA dominating original-corpus repetition (99.2% vs 91.7%
entity-disjoint exact compliance). This experiment maps the operating curve at 400/800/1600/3200
unique v3 counterfactual rows under the same Qwen3-4B model, LoRA rank/alpha/targets, optimizer,
and fixed 300-step x batch-8 update budget. Subsets are nested complete counterfactual pairs, so
the curve tests unique-data/repetition scaling rather than pair-composition noise. Primary surface:
entity-disjoint behavior eval v3 (same heldout rows as v2), with continuation benchmarks as collateral.
