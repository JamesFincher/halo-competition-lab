# exp159 — OLMoE export precision diagnosis

No-training diagnostic using exp154 checkpoints. Compares pre-merge adapter execution to merged-weight execution in bf16 and fp32 at both full-logit and isolated expert-output levels. This distinguishes an algebra/orientation bug from bf16 floating-point amplification through large trained adapter deltas.
