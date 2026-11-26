# Exp167: Qwen3 behavior fixed-epoch scaling

Complement to exp163. Exp163 holds total compute fixed and varies exposed unique rows
(400/800/1600/2400), measuring diversity versus repetition. Exp167 trains exactly one
pass over nested stratified subsets of 400/800/1600/3200 rows, so update count scales
with data count. Comparing the two curves separates data diversity at fixed compute
from data+compute scaling at fixed epochs.

Warmup and cosine decay are proportional to each arm's step count; LoRA initialization
is fixed. Primary measurement is entity-disjoint behavior eval v3/v2 heldout plus
in-domain and held-out transfer continuation collateral.
