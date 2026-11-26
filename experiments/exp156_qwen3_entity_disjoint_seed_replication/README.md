# exp156 — Qwen3-4B entity-disjoint seed replication

Runs the critical plain-LoRA reference on behavior corpus v2 across seeds 42, 7, and 123. Corpus v2 has disjoint train/heldout deployment entities and zero shared user-prompt 8-grams, so exact-compliance gains cannot be explained by copied entity/action surface strings.

The evaluation decomposes exact-compliance failures hierarchically into invalid JSON, action, policy, and tool-field errors; it also reports independent tool accuracy, per-tool exact rates, send_http rare-tool performance, action confusion, continuation benchmarks, memory, and checkpoint-resumable training state. The original 40-task eval is retained only as a bridge to earlier experiments.
