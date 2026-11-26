# exp165 Qwen3 capability replay source controls

STAGED conditional follow-up, intentionally not autoqueued (`staged_*.py`). Activate only after exp164 identifies a behavior-preserving replay dose worth explaining.

Question: is the behavior/capability replay tradeoff caused by coherent benchmark-specific content, coherent benchmark-independent content, answer-format statistics, or simply displacement of behavior-training steps?

Controls at the exp152 reference dose (one replay batch every four steps):
- behavior-only 300 steps: upper behavior anchor
- behavior-only 225 steps: behavior-exposure control matching the replay arm's behavior-step count
- benchmark replay: original scaled MMLU/ARC/HellaSwag train mixture
- general-corpus replay: benchmark-independent capability corpus
- shuffled benchmark replay: answer rotation destroys context-answer coherence while preserving answer distribution/format

Measurement is v2 entity-disjoint behavior plus in-domain and held-out transfer continuation benchmarks.
