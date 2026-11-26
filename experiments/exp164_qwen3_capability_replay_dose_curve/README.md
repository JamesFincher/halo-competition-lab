# Exp164: Qwen3 capability-replay dose curve

Exp152 showed capability replay costs 3.4 entity-disjoint exact-compliance points at a fixed
25% replay rate (95.8% vs 99.2%). This experiment maps the dose curve at deterministic replay
rates of 0%, 1/16, 1/8, 1/4, and 1/2 of the same fixed 300-step x batch-8 budget, holding the
winning v2 behavior corpus, Qwen3-4B LoRA configuration, optimizer, and capability data constant.
Primary readout: behavior exactness/unsafe-refusal/benign-false-refusal versus in-domain and
held-out transfer continuation benchmarks.
