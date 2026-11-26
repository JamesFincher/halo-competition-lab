# Experiment 32 — Targeted-Circuit RL

Novel: use DLA circuit localization as the RL training mask itself — gradient-masked so ONLY the exact top-64 circuit neuron columns of complementary layers train (everything else frozen). vs full-layer control at matched layer count. Tests whether extreme surgical RL preserves safety better than layer-level disjoint RL.
