#!/usr/bin/env python3
"""Experiment 17: 6-Probe 'Christmas-Tree' Node-Tracking Benchmark (SmolLM2-135M).

Benchmarks 6 distinct mechanistic node-tracking cameras head-to-head on the EXACT same model,
exact same top-64 neuron edit budget, and exact same 40 unseen verifiable tasks:
  1. Direct Logit Attribution (DLA / Logit Lens — 0 backward passes!)
  2. Safety-Filtered Activation Spike (False-Node Ratio)
  3. 1-Pass RelP Causal Attribution (arXiv:2508.21258)
  4. 5-Step Integrated Gradients (IG-5)
  5. Flicker / Variance Spike (Var-Ratio)
  6. Multi-Probe Reciprocal-Rank Consensus (DLA + False-Node + RelP)

Also renders a standalone visual SVG heatmap (results/christmas_tree_heatmap.svg) showing
all 30 layers lighting up across probes.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/head_and_node_sniff_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/relp_causal_sniff_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402
import run_head_and_node_sniff as hn_sniff  # noqa: E402
import run_relp_causal_sniff as relp_mod  # noqa: E402


@torch.no_grad()
def probe_dla_and_variance(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    n_pairs: int = 8,
) -> tuple[dict, dict]:
    """Probe 1 (Direct Logit Attribution / Logit Lens, 0 backward passes) + Probe 5 (Variance Flicker)."""
    t0 = time.time()
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)
    inter_dim = int(model.config.intermediate_size)

    prefix_suffix = '{"action": "'
    tok_call = tokenizer.encode("call", add_special_tokens=False)[0]
    tok_ref = tokenizer.encode("ref", add_special_tokens=False)[0]

    # Unembedding refusal-vs-call direction: w_diff [D]
    lm_w = model.lm_head.weight.data.float().cpu()
    w_diff = lm_w[tok_ref] - lm_w[tok_call]  # positive = pushes toward 'refuse'

    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_pairs]
    unsafe_rows = [r for r in train_rows if r["target"]["action"] == "refuse"][:n_pairs]
    canonical_refusal = '{"action": "refuse", "tool": null, "policy_code": "DESTRUCTIVE_OP", "reason": "Refused."}'

    pass_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + r["assistant_text"] for r in benign_rows]
    fail_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + canonical_refusal for r in benign_rows]
    dec_ben_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + prefix_suffix for r in benign_rows]
    dec_uns_texts = [base_rl.build_prompt(tokenizer, r["messages"]) + prefix_suffix for r in unsafe_rows]

    def _collect_samples(texts: list[str], last_tok_only: bool) -> list[torch.Tensor]:
        mats = [[] for _ in range(n_layers)]
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=256).to(device)
            cap = {}
            hooks = []
            for idx, lyr in enumerate(layers):
                def _mk(i: int):
                    def _hk(_m, inp, _out):
                        x = inp[0][0, -1, :] if last_tok_only else inp[0][0, -24:, :].mean(dim=0)
                        cap[i] = x.detach().float().cpu()
                    return _hk
                hooks.append(lyr.mlp.down_proj.register_forward_hook(_mk(idx)))
            try:
                _ = model(**enc)
            finally:
                for h in hooks:
                    h.remove()
            for idx in range(n_layers):
                mats[idx].append(cap[idx])
        return [torch.stack(m, dim=0) for m in mats]

    # DLA at exact decision token
    t_dla0 = time.time()
    dec_ben = _collect_samples(dec_ben_texts, last_tok_only=True)
    dec_uns = _collect_samples(dec_uns_texts, last_tok_only=True)
    dla_scores_by_layer = {}
    dla_nodes_by_layer = {}
    dla_layer_heat = []
    for li in range(n_layers):
        w_down = layers[li].mlp.down_proj.weight.data.float().cpu()  # [D, inter]
        # Direct logit contribution per unit neuron activation: w_diff @ W_down -> [inter]
        neuron_to_refuse_logit = torch.matmul(w_diff, w_down)
        # Neurons actively pushing 'refuse' on BENIGN prompts (minus true unsafe baseline)
        ben_push = (dec_ben[li].mean(dim=0) * neuron_to_refuse_logit).clamp(min=0.0)
        uns_push = (dec_uns[li].mean(dim=0) * neuron_to_refuse_logit).clamp(min=0.0)
        score = ben_push - 0.35 * uns_push
        dla_scores_by_layer[li] = score
        _, idx = torch.topk(score, k=64)
        dla_nodes_by_layer[li] = [int(i) for i in idx.tolist()]
        dla_layer_heat.append(round(float(score.clamp(min=0).topk(16).values.sum().item()), 4))
    dla_sec = round(time.time() - t_dla0, 2)

    # Flicker / Variance Ratio across pass vs fail samples
    t_var0 = time.time()
    m_pass = _collect_samples(pass_texts, last_tok_only=False)
    m_fail = _collect_samples(fail_texts, last_tok_only=False)
    var_nodes_by_layer = {}
    var_layer_heat = []
    for li in range(n_layers):
        vf = m_fail[li].var(dim=0, unbiased=False)
        vp = m_pass[li].var(dim=0, unbiased=False)
        ratio = vf / (vp + 1e-4)
        _, idx = torch.topk(ratio, k=64)
        var_nodes_by_layer[li] = [int(i) for i in idx.tolist()]
        var_layer_heat.append(round(float(ratio.topk(16).values.mean().item()), 4))
    var_sec = round(time.time() - t_var0, 2)

    return (
        {"wall_sec": dla_sec, "nodes_by_layer": dla_nodes_by_layer, "layer_heat": dla_layer_heat, "scores": dla_scores_by_layer},
        {"wall_sec": var_sec, "nodes_by_layer": var_nodes_by_layer, "layer_heat": var_layer_heat},
    )


def probe_integrated_gradients_5(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_rows: list[dict],
    device: torch.device,
    n_pairs: int = 4,
    ig_steps: int = 5,
) -> dict:
    """Probe 4: 5-Step Path-Integrated Gradients on mlp.down_proj inputs at the decision token."""
    t0 = time.time()
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)
    inter_dim = int(model.config.intermediate_size)
    prefix_suffix = '{"action": "'
    tok_call = tokenizer.encode("call", add_special_tokens=False)[0]
    tok_ref = tokenizer.encode("ref", add_special_tokens=False)[0]

    benign_rows = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_pairs]
    ig_acc = [torch.zeros(inter_dim) for _ in range(n_layers)]
    for p in model.parameters():
        p.requires_grad_(False)

    for ben_r in benign_rows:
        p_ben = base_rl.build_prompt(tokenizer, ben_r["messages"]) + prefix_suffix
        enc = tokenizer(p_ben, return_tensors="pt", truncation=True, max_length=256).to(device)
        # Integrate alpha in {0.2, 0.4, 0.6, 0.8, 1.0} scaling down_proj input
        grad_sum = [torch.zeros(inter_dim) for _ in range(n_layers)]
        base_acts = {}
        for step_i in range(1, ig_steps + 1):
            alpha = step_i / float(ig_steps)
            caps = {}
            hooks = []
            for idx, lyr in enumerate(layers):
                def _mk(i: int, a: float):
                    def _hk(_m, inp):
                        x = (inp[0].detach() * a).requires_grad_(True)
                        caps[i] = x
                        if a == 1.0:
                            base_acts[i] = inp[0][0, -1, :].detach().float().cpu()
                        return (x,)
                    return _hk
                hooks.append(lyr.mlp.down_proj.register_forward_pre_hook(_mk(idx, alpha)))
            out = model(**enc)
            margin = out.logits[0, -1, tok_call] - out.logits[0, -1, tok_ref]
            margin.backward()
            for h in hooks:
                h.remove()
            for idx in range(n_layers):
                grad_sum[idx] += caps[idx].grad[0, -1, :].detach().float().cpu()
        for idx in range(n_layers):
            ig_acc[idx] += (base_acts[idx] * (grad_sum[idx] / ig_steps)).abs()

    inv = 1.0 / max(len(benign_rows), 1)
    nodes_by_layer = {}
    layer_heat = []
    for li in range(n_layers):
        s = ig_acc[li] * inv
        _, idx = torch.topk(s, k=64)
        nodes_by_layer[li] = [int(i) for i in idx.tolist()]
        layer_heat.append(round(float(s.topk(16).values.sum().item()), 4))
    return {"wall_sec": round(time.time() - t0, 2), "nodes_by_layer": nodes_by_layer, "layer_heat": layer_heat}


def build_svg_heatmap(probe_heats: dict[str, list[float]], out_path: Path) -> None:
    """Render a clean standalone SVG 'Christmas Tree' layer heatmap across all 5 probes (30 layers)."""
    probe_names = list(probe_heats.keys())
    n_layers = 30
    cell_w, cell_h = 24, 30
    left_pad, top_pad = 190, 56
    width = left_pad + n_layers * cell_w + 30
    height = top_pad + len(probe_names) * (cell_h + 10) + 55

    def _color(v_norm: float) -> str:
        # Dark slate -> warm amber -> bright red-gold ("Christmas tree" lights!)
        v = max(0.0, min(1.0, v_norm))
        if v < 0.5:
            t = v * 2.0
            r = int(20 + t * 215)
            g = int(32 + t * 140)
            b = int(52 - t * 28)
        else:
            t = (v - 0.5) * 2.0
            r = int(235 + t * 20)
            g = int(172 - t * 112)
            b = int(24 + t * 30)
        return f"rgb({r},{g},{b})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text x="20" y="30" fill="#f8fafc" font-family="ui-sans-serif,system-ui,sans-serif" font-size="16" font-weight="700">SmolLM2-135M "Christmas Tree" Layer Activation Map (5 Mechanistic Probes × 30 Layers)</text>',
    ]
    for li in range(n_layers):
        x = left_pad + li * cell_w + cell_w // 2
        lines.append(f'<text x="{x}" y="{top_pad - 8}" fill="#94a3b8" font-family="ui-monospace,monospace" font-size="10" text-anchor="middle">L{li}</text>')

    for pi, name in enumerate(probe_names):
        y = top_pad + pi * (cell_h + 10)
        vals = probe_heats[name]
        mx = max(vals) if max(vals) > 0 else 1.0
        lines.append(f'<text x="{left_pad - 12}" y="{y + 20}" fill="#e2e8f0" font-family="ui-sans-serif,system-ui,sans-serif" font-size="12" font-weight="600" text-anchor="end">{name}</text>')
        for li, v in enumerate(vals):
            x = left_pad + li * cell_w
            c = _color(v / mx)
            lines.append(f'<rect x="{x + 1}" y="{y}" width="{cell_w - 2}" height="{cell_h}" rx="4" fill="{c}"/>')

    lines.append(f'<text x="20" y="{height - 16}" fill="#94a3b8" font-family="ui-sans-serif,system-ui,sans-serif" font-size="11">Normalized per-probe intensity (dark = quiet, amber/red = hottest problem nodes). Notice consensus hotspot at Layers 27–29.</text>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cpu")
    out_dir = ROOT / "experiments/christmas_tree_node_probes/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = base_rl.load_rows(ROOT / "data/policy_train.jsonl")
    eval_rows = base_rl.load_rows(ROOT / "data/policy_eval.jsonl")
    ben = [r for r in train_rows if r["target"]["action"] == "call_tool"]
    uns = [r for r in train_rows if r["target"]["action"] == "refuse"]
    dev_rows = ben[-5:] + uns[-5:]

    print("[Setup] Loading SmolLM2-135M + Halo SFT checkpoint-36...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    base_raw = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).to(device)
    sft_model = PeftModel.from_pretrained(base_raw, str(ROOT / "checkpoints/smollm2-135m-policy-lora/checkpoint-36")).merge_and_unload()
    sft_model.eval()

    # Run all 5 individual cameras
    print("[Probe 1 & 5] Direct Logit Attribution (0-grad) + Variance Flicker...", flush=True)
    dla_res, var_res = probe_dla_and_variance(sft_model, tokenizer, train_rows, device, n_pairs=8)

    print("[Probe 2] Safety-Filtered Activation Spike (False-Node Ratio)...", flush=True)
    t2 = time.time()
    fn_prof = hn_sniff.sniff_heads_and_false_nodes(sft_model, tokenizer, train_rows, device, n_pairs=8, safety_penalty=0.45)
    fn_sec = round(time.time() - t2, 2)
    fn_layer_heat = [round(float(fn_prof["resid_false_dirs"][li].norm().item()), 4) for li in range(30)]

    print("[Probe 3] 1-Pass RelP Causal Attribution...", flush=True)
    relp_res = relp_mod.compute_relp_causal_scores(sft_model, tokenizer, train_rows, device, n_pairs=8)
    relp_layer_map = {r["layer"]: r["total_causal"] for r in relp_res["layer_causal_rank"]}
    relp_layer_heat = [relp_layer_map[li] for li in range(30)]

    print("[Probe 4] 5-Step Integrated Gradients (IG-5)...", flush=True)
    ig_res = probe_integrated_gradients_5(sft_model, tokenizer, train_rows, device, n_pairs=4, ig_steps=5)

    # Probe 6: Multi-Probe Reciprocal Rank Consensus (DLA + False-Node + RelP)
    consensus_nodes = {}
    for li in range(30):
        rrf: dict[int, float] = {}
        for nl in (dla_res["nodes_by_layer"][li], fn_prof["false_nodes_by_layer"][li], relp_res["causal_nodes_by_layer"][li]):
            for rank, nid in enumerate(nl):
                rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (60.0 + rank)
        consensus_nodes[li] = [nid for nid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:64]]

    # Render visual Christmas-Tree SVG heatmap!
    probe_heats = {
        "1. Direct Logit Lens (0-grad)": dla_res["layer_heat"],
        "2. False-Node Spike": fn_layer_heat,
        "3. 1-Pass RelP Causal": relp_layer_heat,
        "4. 5-Step Integrated Grads": ig_res["layer_heat"],
        "5. Variance Flicker Ratio": var_res["layer_heat"],
    }
    svg_path = out_dir / "christmas_tree_heatmap.svg"
    build_svg_heatmap(probe_heats, svg_path)
    print(f"[Visual] Saved Christmas-Tree heatmap -> {svg_path}", flush=True)

    # Head-to-head verifiable 40-task benchmark: edit top-64 nodes picked by EACH probe on layers [29, 28]!
    edit_layers = [29, 28]
    probes = [
        ("probe1_dla_logit_lens", dla_res["nodes_by_layer"], dla_res["wall_sec"]),
        ("probe2_false_node_spike", fn_prof["false_nodes_by_layer"], fn_sec),
        ("probe3_relp_causal", relp_res["causal_nodes_by_layer"], relp_res["wall_sec"]),
        ("probe4_integrated_grads_5", ig_res["nodes_by_layer"], ig_res["wall_sec"]),
        ("probe5_variance_flicker", var_res["nodes_by_layer"], var_res["wall_sec"]),
        ("probe6_consensus_rrf", consensus_nodes, round(dla_res["wall_sec"] + fn_sec + relp_res["wall_sec"], 2)),
    ]

    scoreboard = {}
    print("\n[Bake-Off] Evaluating all 6 probes on identical 40 unseen verifiable tasks (layers [29, 28], top-64 nodes, 0 RL steps)...", flush=True)
    for name, nodes_map, p_sec in probes:
        m_ed, gate_meta = hn_sniff.double_check_gate(
            sft_model, tokenizer, dev_rows, device,
            lambda m, st, nm=nodes_map: hn_sniff.apply_false_node_mlp_edit(
                m, edit_layers, nm, fn_prof["resid_false_dirs"], 64, st
            ),
            name, strengths=(0.45, 0.70, 0.90),
        )
        ev = base_rl.evaluate_heldout(m_ed, tokenizer, eval_rows, device)
        scoreboard[name] = {
            "probe_wall_sec": p_sec,
            "top5_layer29_nodes": nodes_map[29][:5],
            "gate_accepted": gate_meta["accepted"],
            "accepted_strength": gate_meta.get("strength", 0.0),
            "eval": ev,
        }
        print(
            f"  [{name:26s}] t={p_sec:5.2f}s | st={gate_meta.get('strength', 0.0):.2f} | "
            f"exact={ev['exact_compliance_rate']}% | policy={ev['policy_accuracy']}% | "
            f"unsafe={ev['unsafe_refusal_rate']}% | ben_fr={ev['benign_false_refusal_rate']}% | rew={ev['mean_reward']:.3f}",
            flush=True,
        )

    summary = {"edit_layers": edit_layers, "layer_heats": probe_heats, "scoreboard": scoreboard}
    (out_dir / "probe_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved {out_dir / 'probe_benchmark_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
