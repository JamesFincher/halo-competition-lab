# Experiment 144 / H22 — ultra-sparse edit + anchored MLP LoRA

Combines the two validated winners: canonical ultra-sparse surgical edit (16 columns/layer, strength 0.9) and post-edit-KL-anchored complementary MLP-only LoRA at layers 24–27. H22 runs seeds 7/42/123 and evaluates exported merged artifacts. Controls at seed 42: LoRA-only/no-edit, deterministically sampled random-layer MLP ([0,3,20,23], seed42), edited-layer-inclusion negative control, vector-only lower bound. Includes canonical-edit replication, behavior, benchmark, adapter-vs-export evals, and per-step telemetry.
