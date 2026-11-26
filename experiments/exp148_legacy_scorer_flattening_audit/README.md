# Experiment 148 — legacy scorer / logit-flattening audit

Measurement-only audit of the legacy `scripts/real_benchmark.py` scoring rule:

```text
score = -mean_full_sequence_loss * sequence_token_length / choice_character_length
```

The merged SFT model is wrapped read-only with logit scales `1.0, 0.5, 0.25, 0.1, 0.0`, where scale 0 is exactly uniform. For each scale and benchmark the experiment records exact legacy accuracy, clean continuation sum/token/byte accuracy, correct-vs-best margins, answer character/token lengths, and tie-inclusive and unique longest/shortest selection rates. Static correct-is-longest/shortest baselines are reported independently of the model.

This experiment does not retrain or duplicate exp146. exp146 tests the actual cache-mode behavioral-collapse arms; exp148 isolates whether the legacy scorer itself can manufacture gains under flattening. Output: `results/legacy_scorer_flattening_audit_summary.json`.
