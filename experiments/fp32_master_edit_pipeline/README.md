# Experiment 34 — fp32-Master Edit → bf16 Cast Production Pipeline

Wave-1 finding: direct bf16 editing inverts fp32 results (DLA 15% vs random 47.5%). Comp-critical: apply surgical edit in fp32 master weights, then cast to bf16 — does the edit survive? Arms: fp32 reference, deterministic cast, stochastic-rounding cast, direct-bf16 control. Inspired by wave-1 swarm findings (internal research loop).
