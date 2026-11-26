# Experiment 33 — Dtype Sensitivity of Surgical Edits

Wave-1 observation: identical DLA edits gave 80% exact / 90% unsafe refusal in CPU fp32 but apparent safety collapse in CUDA bf16. Directly measures edit precision vs dtype — critical for competition transfer since Halo production trains in bf16. Informs whether edits need fp32 application + stochastic rounding.
