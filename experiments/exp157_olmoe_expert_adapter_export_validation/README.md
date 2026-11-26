# exp157 — OLMoE expert-adapter export validation

No-training follow-up to exp154. Expert adapters are linear in the expert intermediate activation, so each one should merge exactly into `down_proj` as `B @ A * alpha/rank`. This experiment loads exp154's final adapter checkpoints, performs that merge in fresh base weights, and verifies logits plus greedy generation against the pre-merge adapted model before running behavior/continuation canaries on the merged form.

It answers the exportability requirement without retraining and distinguishes a real reusable weight artifact from an in-session-only wrapper result.
