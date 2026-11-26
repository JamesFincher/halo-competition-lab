# Experiment 146 — cache-mode inversion logit-degeneracy diagnostic

Replicates the L20 behavioral-collapse versus L27 control contrast in both legacy KV-cache and cache-corrected modes and adds distribution diagnostics to determine whether benchmark inversion is compatible with logit flattening rather than capability gain. Measures next-token entropy, top-1 probability, KL-to-uniform, correct-minus-best-incorrect continuation margins, and positive-margin rate, answer-length distributions, and shortest-answer selection rates on MMLU test probes. Generation collapse cannot directly contaminate teacher-forced continuation scoring; the artifact hypothesis is indirect distribution flattening.


This is the decisive cache-mode version. The earlier exp145 script is cache-corrected-only and, if it runs before this patch, should be interpreted as a partial corrected-training diagnostic rather than a replication of the original pre-cache inversion.
