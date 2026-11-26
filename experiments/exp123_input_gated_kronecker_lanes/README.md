# Experiment 123 — Input-Gated Kronecker-Factored Adaptation Lanes

**Research inspiration:** `research/HYPOTHESES.md` **H6** + **arXiv:2611.33236** (*Kronecker Input-Gated Rank Adaptation for Parameter-Efficient Fine-Tuning*).

## Hypothesis

A frozen random Kronecker factor can multiply trainable adapter rank without trainable parameters, while a quantized input gate can allocate rank only to harder decision states. This may beat matched LoRA on parameter-normalized post-edit policy acquisition.

## Method

1. Start from the validated post-edit checkpoint and freeze all base weights.
2. For q/o/up/down matrices in complementary layers 24–27, use exact Kronecker geometry \(\Delta W=A\otimes(B_1\operatorname{diag}(g(x))B_2)\) with \(p=3\).
3. Freeze random \(A\), train \(B_1/B_2\), and use a shared input gate quantized on a four-step lattice with straight-through gradients.
4. Compare input-gated all/attention/MLP lanes, input-independent constant gates, frozen-random gates, and matched targeted LoRA r16.
5. Use identical post-edit KL anchoring, exact-verifier GRPO rewards, seeds, prompts, and `weight_decay=0`; report gate occupancy and exact trainable parameter counts.

## Run

```bash
python3 experiments/exp123_input_gated_kronecker_lanes/scripts/run_input_gated_kronecker.py
```

## Output

`results/input_gated_kronecker_summary.json` plus `results/step_metrics.jsonl` — adaptivity diagnostic, gate lattice statistics, reward/dead-group/KL traces, parameter-normalized arm outcomes, final policy/safety metrics, wall time, and device.
