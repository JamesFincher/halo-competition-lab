# exp155 — OLMoE expert-surface controls

Complements exp154 rather than duplicating it. Under identical answer-only capability training, this experiment compares:

1. adapters in the lowest answer-token-routed experts (activation necessity control), and
2. dense attention LoRA rank 12 across q/k/v/o (exactly parameter-matched on real OLMoE to exp154's 8 experts/layer x rank 8 x 16 layers = 3,145,728 effective adapter parameters).

The comparison tells whether any capability movement in exp154 is specific to benchmark-selective MoE experts, merely requires activated parameters, or is dominated by ordinary dense adaptation. It uses the same decontaminated capability corpus, frozen base/router, checkpoint/resume, behavior collateral, continuation benchmarks, and cache diagnostics.
