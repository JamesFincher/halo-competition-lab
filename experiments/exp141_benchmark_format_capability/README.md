# Experiment 141 — benchmark-format capability training

**Controlled variable:** training-text surface format only, while answer-only token weighting is fixed.

Exp132 showed that full-text capability training improves in-distribution validation loss but does not yet produce credible continuation-benchmark transfer. Exp140 isolates whether context tokens dilute answer learning. This experiment isolates the remaining train/evaluation format mismatch: the continuation benchmark scores `context + " " + answer` with a bare question/passage context, while the original capability corpus uses lettered options followed by an explicit `Answer:`/`Continuation:` marker.

## Arms

1. `lettered_answer_only_reference` — original lettered/marker corpus, answer-only loss.
2. `benchmark_direct_answer_only` — same underlying rows in exact continuation-style format (`question/context + " " + answer/continuation`), answer-only loss with explicit answer character spans.

Both arms use the same single zero-init width-64 head at layer 27, same seed, same batch positions, same underlying rows and answer payloads, same optimizer budget, no behavior reward, before/after behavioral eval, legacy benchmark, continuation benchmark, and full/answer validation-loss measurements on both corpus formats.

## Outputs

`results/benchmark_format_capability_summary.json` contains per-arm behavioral/benchmark deltas, continuation-benchmark metrics at higher default sensitivity (`n=150`), matched-format validation deltas, corpus provenance, pairing diagnostics, TransformerLens geometry probe, and training traces.
