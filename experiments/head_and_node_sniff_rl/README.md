# Experiment 12: Head & False-Node Seek, Sniff & Double-Check RL

**Priority:** Fine-Grained Mechanistic Extension (**8.5/10**)  
**Target Models:** `HuggingFaceTB/SmolLM2-135M-Instruct` (270 Q-heads, 46,080 MLP neurons — verified live) → `Qwen/Qwen3-4B` → `unsloth/gpt-oss-20b-BF16`

## Core Idea (Plain English)
Upgrades **Seek & Sniff** from layer-level granularity down to **individual attention heads** and **individual MLP neurons ("false nodes")**, with an immediate **Double-Check Gate before moving on**:

1. **Attention-Head Sniff (0 KV touched):** Hooks the input to `self_attn.o_proj` across every layer, splits by query head `h`, and scores each head's **false-trigger firing vs true-safety firing** (`false_l2` vs `safety_l2`). Identifies the exact top "false-trigger heads" without ever touching shared GQA `k_proj`/`v_proj`.
2. **MLP False-Node Sniff:** Hooks the intermediate activations entering `mlp.down_proj` across all 46,080 neurons and ranks every neuron by `false_node_score = |mu_false - mu_benign| - alpha * |mu_unsafe - mu_benign|` — isolating individual neurons that spike on false blocks but NOT on real threats.
3. **Sequential Double-Check Gate (Before Moving On):**
   - **Check 1 (Attention Heads):** Project out the false direction from ONLY the top-K false-trigger `o_proj` head slices → immediately double-check on dev slice → keep only if dev reward improves without breaking safety (else roll back).
   - **Check 2 (MLP False Nodes):** Project out ONLY the top-K false MLP neurons (`down_proj[:, false_nodes]`) → immediately double-check → keep only if improves.
   - **Stage 2 (Disjoint Re-Anchored RL):** Lock edited layers/heads, anchor KL to the double-checked checkpoint, and run gentle targeted GRPO on complementary layers.

## Files
- `scripts/run_head_and_node_sniff.py` — Runs head-level + neuron-level false-node sniffing and benchmarks 4 arms (Head-Only, False-Node-Only, Combined Head+Node Double-Check, Combined + Disjoint RL) on 40 unseen held-out tasks.
- `configs/qwen3-4b-head-node-double-check-grpo.yaml` — Halo v1.3.0rc1 GPU config for scaling to `Qwen3-4B`.
- `results/HEAD_NODE_SNIFF_REPORT.md` — Tracked live results, top false heads & false nodes map, and scoreboard.
