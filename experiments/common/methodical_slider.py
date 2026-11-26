"""Shared utilities for methodical single-variable slider-head ablations."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
sys.path.insert(0, str(ROOT / "experiments/common"))
import run_micro_targeted_rl as base_rl  # noqa: E402


class PlainSliderHead(nn.Module):
    """Exact causal single-head slider shape used by slider_head_fork v1."""

    def __init__(self, dim: int, head_dim: int = 64, include_alpha: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(dim, head_dim, bias=False)
        self.k = nn.Linear(dim, head_dim, bias=False)
        self.v = nn.Linear(dim, head_dim, bias=False)
        self.o = nn.Linear(head_dim, dim, bias=False)
        if include_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))

    def attention_delta(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.full((T, T), -1e9, device=x.device, dtype=scores.dtype), diagonal=1)
        att = torch.softmax(scores + mask, dim=-1)
        return self.o(att @ v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.attention_delta(x)


class SharedAlphaSliderBank(nn.Module):
    """N parallel plain heads at one layer behind one shared exact-zero alpha."""

    def __init__(self, dim: int, head_dim: int, count: int):
        super().__init__()
        self.heads = nn.ModuleList([PlainSliderHead(dim, head_dim, include_alpha=False) for _ in range(count)])
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = sum(head.attention_delta(x) for head in self.heads)
        return x + self.alpha * delta


def install_plain_slider(model: torch.nn.Module, layer: int, head: PlainSliderHead):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        shifted = head(hidden)
        return (shifted,) + tuple(out[1:]) if isinstance(out, tuple) else shifted

    return model.model.layers[layer].register_forward_hook(hook)


def ordered_fork_rows(train_rows: list[dict], n_each: int = 12) -> list[dict]:
    """Match slider_head_fork's fixed prompt stream exactly (all benign, then unsafe)."""
    return [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_each] + [
        r for r in train_rows if r["target"]["action"] == "refuse"
    ][:n_each]


def train_plain_slider(
    model: torch.nn.Module,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    head: PlainSliderHead,
    layer: int,
    steps: int = 32,
    group_size: int = 4,
    lr: float = 3e-3,
    max_new_tokens: int = 48,
    alpha_clip: float | None = None,
    alpha_logit_scale: float = 1.0,
) -> tuple[object, list[dict]]:
    """Reproduce slider_head_fork optimization: head-only, grouped REINFORCE, no KL/BC."""
    handle = install_plain_slider(model, layer, head)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        row = rows[step % len(rows)]
        ids = tokenizer(base_rl.build_prompt(tokenizer, row["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
        prompt_len = int(ids["input_ids"].shape[1])
        model.eval()
        head.eval()
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        rewards = []
        for gi in range(group_size):
            text = tokenizer.decode(gen[gi][prompt_len:], skip_special_tokens=True)
            rew, _ = base_rl.score_completion(text, row["target"])
            rewards.append(float(rew))
        rew_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = rew_t - rew_t.mean()
        model.train()
        head.train()
        loss = torch.zeros((), device=device)
        for gi in range(group_size):
            lp = base_rl.completion_logprobs(model, gen[gi].unsqueeze(0), prompt_len)
            loss = loss - adv[gi] * lp
        loss = loss / group_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
        opt.step()
        if alpha_clip is not None:
            with torch.no_grad():
                head.alpha.clamp_(-alpha_clip, alpha_clip)
        rec = {
            "step": step,
            "mean_reward": round(float(rew_t.mean().item()), 4),
            "reward_std": round(float(rew_t.std(unbiased=False).item()), 4),
            "alpha": round(float(head.alpha.detach().item()), 6),
            "loss": round(float(loss.detach().item()), 5),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(f"    step {step:02d}: rew={rec['mean_reward']:.2f} alpha={rec['alpha']:.5f} grad={grad_norm:.3f}", flush=True)
    return handle, log


def probe_layer_geometry(hf_model, tokenizer, rows: list[dict], device: torch.device, layer: int = 27) -> dict:
    """TransformerLens probe of residual geometry and attention entropy at one layer."""
    from tlens_utils import cache_text, load_hooked_hf_model, residual_hook_name

    tl = load_hooked_hf_model(hf_model, tokenizer, device, torch.float32)
    resid_name = residual_hook_name(layer, "post")
    pattern_name = f"blocks.{layer}.attn.hook_pattern"
    final_by_class = {"call_tool": [], "refuse": []}
    entropies = []
    with torch.no_grad():
        for row in rows:
            text = base_rl.build_prompt(tokenizer, row["messages"])
            _tokens, _logits, cache = cache_text(tl, text)
            h = cache[resid_name][0, -1, :].detach().float().cpu()
            final_by_class[row["target"]["action"]].append(h)
            try:
                pat = cache[pattern_name].detach().float()
            except Exception:
                pat = None
            if pat is not None:
                ent = -(pat.clamp_min(1e-12) * pat.clamp_min(1e-12).log()).sum(dim=-1).mean().cpu()
                entropies.append(float(ent))
    out = {}
    pooled = []
    for action, vals in final_by_class.items():
        if vals:
            x = torch.stack(vals)
            pooled.append(x)
            out[f"{action}_final_resid_norm_mean"] = round(float(x.norm(dim=-1).mean()), 5)
    if all(final_by_class.values()):
        xb = torch.stack(final_by_class["call_tool"])
        xu = torch.stack(final_by_class["refuse"])
        mb, mu = xb.mean(0), xu.mean(0)
        out["benign_unsafe_mean_cosine"] = round(float(torch.nn.functional.cosine_similarity(mb, mu, dim=0)), 5)
        x = torch.cat([xb, xu], 0)
        x = x - x.mean(dim=0, keepdim=True)
        sv = torch.linalg.svdvals(x)
        p = (sv ** 2) / max(float((sv ** 2).sum()), 1e-12)
        out["resid_effective_rank"] = round(float(torch.exp(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())), 4)
    if entropies:
        out["attention_entropy_mean"] = round(sum(entropies) / len(entropies), 5)
    del tl
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def head_parameter_count(head: nn.Module) -> int:
    return sum(int(p.numel()) for p in head.parameters())


def probe_layer27_geometry(hf_model, tokenizer, rows: list[dict], device: torch.device) -> dict:
    return probe_layer_geometry(hf_model, tokenizer, rows, device, layer=27)


def probe_layers_geometry(hf_model, tokenizer, rows: list[dict], device: torch.device, layers: list[int]) -> dict:
    """TransformerLens probe of multiple layers using one wrapped model and one cache pass per row."""
    from tlens_utils import cache_text, load_hooked_hf_model, residual_hook_name

    tl = load_hooked_hf_model(hf_model, tokenizer, device, torch.float32)
    resid = {li: {"call_tool": [], "refuse": []} for li in layers}
    entropies = {li: [] for li in layers}
    with torch.no_grad():
        for row in rows:
            text = base_rl.build_prompt(tokenizer, row["messages"])
            _tokens, _logits, cache = cache_text(tl, text)
            for li in layers:
                h = cache[residual_hook_name(li, "post")][0, -1, :].detach().float().cpu()
                resid[li][row["target"]["action"]].append(h)
                try:
                    pat = cache[f"blocks.{li}.attn.hook_pattern"].detach().float()
                except Exception:
                    pat = None
                if pat is not None:
                    entropies[li].append(float((-(pat.clamp_min(1e-12) * pat.clamp_min(1e-12).log()).sum(dim=-1).mean()).cpu()))
    out = {}
    for li in layers:
        rec = {}
        if resid[li]["call_tool"] and resid[li]["refuse"]:
            xb = torch.stack(resid[li]["call_tool"])
            xu = torch.stack(resid[li]["refuse"])
            mb, mu = xb.mean(0), xu.mean(0)
            rec["call_tool_final_resid_norm_mean"] = round(float(xb.norm(dim=-1).mean()), 5)
            rec["refuse_final_resid_norm_mean"] = round(float(xu.norm(dim=-1).mean()), 5)
            rec["benign_unsafe_mean_cosine"] = round(float(torch.nn.functional.cosine_similarity(mb, mu, dim=0)), 5)
            x = torch.cat([xb, xu], 0)
            x = x - x.mean(dim=0, keepdim=True)
            sv = torch.linalg.svdvals(x)
            p = (sv ** 2) / max(float((sv ** 2).sum()), 1e-12)
            rec["resid_effective_rank"] = round(float(torch.exp(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())), 4)
        if entropies[li]:
            rec["attention_entropy_mean"] = round(sum(entropies[li]) / len(entropies[li]), 5)
        out[str(li)] = rec
    del tl
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


class IndependentLayerSliderBank(nn.Module):
    """Plain slider heads distributed across layers, each with its own alpha."""

    def __init__(self, dim: int, head_dim: int, layers: list[int]):
        super().__init__()
        self.layers = list(layers)
        self.layer_heads = nn.ModuleDict({str(li): PlainSliderHead(dim, head_dim) for li in self.layers})

    def alphas(self) -> dict[int, float]:
        return {li: float(self.layer_heads[str(li)].alpha.detach().item()) for li in self.layers}


def install_independent_layer_bank(model: torch.nn.Module, bank: IndependentLayerSliderBank) -> list[object]:
    handles = []
    for li in bank.layers:
        handles.append(install_plain_slider(model, li, bank.layer_heads[str(li)]))
    return handles


def train_independent_layer_bank(
    model: torch.nn.Module,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    bank: IndependentLayerSliderBank,
    steps: int = 32,
    group_size: int = 4,
    lr: float = 3e-3,
    max_new_tokens: int = 48,
    alpha_clip: float | None = None,
    alpha_penalty: float = 0.0,
) -> tuple[list[object], list[dict]]:
    """Fork-style joint multi-layer head training with optional post-step alpha clamp."""
    handles = install_independent_layer_bank(model, bank)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(bank.parameters(), lr=lr)
    log: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        row = rows[step % len(rows)]
        ids = tokenizer(base_rl.build_prompt(tokenizer, row["messages"]), return_tensors="pt", truncation=True, max_length=256).to(device)
        prompt_len = int(ids["input_ids"].shape[1])
        model.eval()
        bank.eval()
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        rewards = []
        for gi in range(group_size):
            text = tokenizer.decode(gen[gi][prompt_len:], skip_special_tokens=True)
            rew, _ = base_rl.score_completion(text, row["target"])
            rewards.append(float(rew))
        rew_t = torch.tensor(rewards, device=device, dtype=torch.float32)
        adv = rew_t - rew_t.mean()
        model.train()
        bank.train()
        loss = torch.zeros((), device=device)
        for gi in range(group_size):
            lp = base_rl.completion_logprobs(model, gen[gi].unsqueeze(0), prompt_len)
            loss = loss - adv[gi] * lp
        loss = loss / group_size
        penalty_value = 0.0
        if alpha_penalty:
            penalty = sum(head.alpha.pow(2).sum() for head in bank.layer_heads.values())
            penalty_value = float(penalty.detach().item())
            loss = loss + alpha_penalty * penalty
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0).item())
        opt.step()
        if alpha_clip is not None:
            with torch.no_grad():
                for head in bank.layer_heads.values():
                    head.alpha.clamp_(-alpha_clip, alpha_clip)
        alphas = {str(k): round(v, 6) for k, v in bank.alphas().items()}
        rec = {
            "step": step,
            "mean_reward": round(float(rew_t.mean().item()), 4),
            "reward_std": round(float(rew_t.std(unbiased=False).item()), 4),
            "alphas": alphas,
            "alpha_min": round(min(bank.alphas().values()), 6),
            "alpha_max": round(max(bank.alphas().values()), 6),
            "loss": round(float(loss.detach().item()), 5),
            "alpha_l2_sum": round(penalty_value, 8),
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        log.append(rec)
        if step % 8 == 0 or step == steps - 1:
            print(f"    step {step:02d}: rew={rec['mean_reward']:.2f} alphas={alphas} grad={grad_norm:.3f}", flush=True)
    return handles, log
