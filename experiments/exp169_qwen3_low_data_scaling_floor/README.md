# Exp169: Qwen3 low-data scaling floor

Question: where is the low-data floor for entity-disjoint behavioral compliance when the fixed exp152 update budget forces high repetition?

Design: nested complete counterfactual-pair subsets of v3 at 50/100/200/400 unique rows, fixed 300x8=2400 examples (48/24/12/6 approximate epochs), fixed LoRA initialization, row-level behavioral plus continuation/transfer measurement.

Motivation: the streamed exp163 log shows rows_400 already at 100% exact compliance, so the original 400-2400 fixed-compute curve is ceilinged. This follow-up measures the acquisition threshold below 400 rows rather than adding higher, uninformative points.
