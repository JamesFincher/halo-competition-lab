# Experiment 41 — Original-Operator GPU Replication (Validation Test)

Runs the original exp12/17 surgical pipeline verbatim — contrastive activation sniff → per-layer resid false-directions → dev-gated strength sweep (0.45/0.70/0.90) — on H100, after exp39 proved unembed directions are anti-safety at every strength and crude variance directions don't replicate. If the original replicates its 80%+ exact compliance here, we have a validated base operator for all future novel methods; if not, the CPU-era results themselves are suspect. Either answer is load-bearing for the whole lab.
