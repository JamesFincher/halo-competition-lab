# Staged Exp168: counterfactual-pair structure ablation

Intentionally inactive (`staged_*.py`) until exp163/exp167 establish the behavior-scaling
curve. At fixed 1,600 examples and 200 optimizer updates, compares:

- complete rows from 800 counterfactual pairs in the corpus's normal distributed order,
- one row each from 1,600 pairs (more context diversity, no complete-pair coverage),
- one row each from the same 800 pairs repeated for a second epoch, and
- the same complete 800 pairs reordered so counterparts are adjacent in training batches.

The design separates pair coverage, diversity, repetition, and batch adjacency while holding
Qwen3-4B, LoRA initialization/configuration, optimizer budget, evaluation surfaces, and seed fixed.
Activate only if the scaling curve shows a meaningful benefit worth mechanistically interpreting.
