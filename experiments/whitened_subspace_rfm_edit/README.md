# Experiment 16: Safety-Whitened Rank-k Subspace Projection + Closed-Form Strength

**Papers:** `arXiv:2607.02396` (*Fast Multi-dimensional Refusal Subspaces via RFM-AGOP*, Jul 2026), `arXiv:2603.27518` (*Over-Refusal and Representation Subspaces*, Mar 2026), `arXiv:2605.05220` (*MidSteer: Optimal Affine Framework*, Apr 2026), `arXiv:2610.05190` (*FORGE*, Oct 2026)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (verified live) → `Qwen/Qwen3-4B` → `unsloth/gpt-oss-20b-BF16`

## Core Math (Multi-Dimensional Safety-Whitened Subspace vs Rank-1 Line)
`arXiv:2607.02396` and `arXiv:2603.27518` show that refusal and over-refusal occupy a **multi-dimensional subspace**, not a single 1D line.  
Instead of subtracting one unwhitened mean vector \(d_l = \mu_{\text{fail}} - \mu_{\text{pass}}\), we compute the **Safety-Whitened / Orthogonalized Rank-\(k\) Subspace**:
1. Extract per-sample false-refusal deltas \(\Delta_F \in \mathbb{R}^{N \times d}\) and true-safety deltas \(\Delta_S \in \mathbb{R}^{N \times d}\).
2. Project \(\Delta_F\) out of the top-\(m\) principal true-safety subspace \(Q_S\) (exact Gram-Schmidt / null-space constraint):
   \[\tilde{\Delta}_F = \Delta_F (I - Q_S Q_S^T)\]
3. Take the top-\(k\) right singular vectors \(U_k \in \mathbb{R}^{d \times k}\) of \(\tilde{\Delta}_F\) (`k = 1, 2, 3, 4`) and apply **Rank-\(k\) Subspace Projection**:
   \[W_{\text{down}} \leftarrow W_{\text{down}} - s \cdot U_k U_k^T W_{\text{down}}\]
   leaving the true-safety subspace \(Q_S\) mathematically orthogonal (preserved!) while erasing multi-mode false refusal in a single shot.

## Files
- `scripts/run_whitened_subspace_edit.py` — Benchmarks Rank-1 vs Safety-Nullspace Rank-\(k\) (`k=1,2,3,4`) zero-step edits + Closed-Form strength estimate on 40 unseen tasks.
- `configs/qwen3-4b-whitened-subspace-grpo.yaml` — Halo v1.3.0rc1 GPU config for scaling to `Qwen3-4B`.
- `results/WHITENED_SUBSPACE_REPORT.md` — Tracked scoreboard and subspace singular values.
