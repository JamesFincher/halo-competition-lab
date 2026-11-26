# Benchmark Failure Tree — Reverse-Engineering Bad Outputs

Parallel fork: run real benchmark items, split correct vs incorrect, christmas-tree probe the activation differences (the failure signature), then causally test which weights produce bad outputs by suppressing failure-correlated directions and measuring the benchmark delta. Direct answer to: which weights contribute to bad results.
