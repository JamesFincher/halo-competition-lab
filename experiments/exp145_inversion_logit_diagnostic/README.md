# Experiment 145 — inversion logit-degeneracy diagnostic

Replicates the L20 behavioral-collapse versus L27 control contrast and adds distribution diagnostics to determine whether benchmark inversion is compatible with logit flattening rather than capability gain. Measures next-token entropy, top-1 probability, KL-to-uniform, correct-minus-best-incorrect continuation margins, and positive-margin rate on MMLU test probes. Generation collapse cannot directly contaminate teacher-forced continuation scoring; the artifact hypothesis is indirect distribution flattening.
