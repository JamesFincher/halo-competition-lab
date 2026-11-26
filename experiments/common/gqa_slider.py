"""Architecture-matched zero-init slider-gated GQA attention lanes for SmolLM2."""
from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "experiments/edit_then_target_rl/scripts"))
import run_micro_targeted_rl as base_rl  # noqa: E402


class GqaSliderHead(nn.Module):
    """One SmolLM2-style GQA lane: 3 query heads sharing 1 KV head.

    alpha starts exactly at zero, so installing the lane is a bit-exact no-op.
    q/k/v/o shapes match one key-value group of SmolLM2-135M (9Q/3KV, head_dim=64).
    """

    def __init__(self, hidden_size: int = 576, query_heads: int = 3, head_dim: int = 64, init_std: float = 0.02):
        super().__init__()
        self.query_heads = query_heads
        self.head_dim = head_dim
        self.q = nn.Linear(hidden_size, query_heads * head_dim, bias=False)
        self.k = nn.Linear(hidden_size, head_dim, bias=False)
        self.v = nn.Linear(hidden_size, head_dim, bias=False)
        self.o = nn.Linear(query_heads * head_dim, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            for proj in (self.q, self.k, self.v, self.o):
                proj.weight.normal_(0.0, init_std)

    def attention_delta(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q(x).view(bsz, seq_len, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsz, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsz, seq_len, 1, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.full((seq_len, seq_len), -1e9, device=x.device, dtype=scores.dtype), diagonal=1)
        att = torch.softmax(scores + mask, dim=-1)
        mixed = torch.matmul(att, v).transpose(1, 2).contiguous().view(bsz, seq_len, self.query_heads * self.head_dim)
        return self.o(mixed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.attention_delta(x)


def warm_copy_from_gqa_group(head: GqaSliderHead, model: torch.nn.Module, source_layer: int, kv_group: int = 0) -> None:
    """Copy one existing q-triplet/kv lane and matching output columns into the slider lane."""
    attn = model.model.layers[source_layer].self_attn
    hd = head.head_dim
    q0 = kv_group * head.query_heads * hd
    q1 = q0 + head.query_heads * hd
    k0 = kv_group * hd
    k1 = k0 + hd
    with torch.no_grad():
        head.q.weight.copy_(attn.q_proj.weight[q0:q1, :])
        head.k.weight.copy_(attn.k_proj.weight[k0:k1, :])
        head.v.weight.copy_(attn.v_proj.weight[k0:k1, :])
        head.o.weight.copy_(attn.o_proj.weight[:, q0:q1])


def install_slider_hook(model: torch.nn.Module, layer: int, head: GqaSliderHead):
    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        shifted = head(hidden)
        return (shifted,) + tuple(out[1:]) if isinstance(out, tuple) else shifted

    return model.model.layers[layer].register_forward_hook(hook)


def balanced_rows(train_rows: list[dict], n_each: int = 12) -> list[dict]:
    benign = [r for r in train_rows if r["target"]["action"] == "call_tool"][:n_each]
    unsafe = [r for r in train_rows if r["target"]["action"] == "refuse"][:n_each]
    rows = []
    for pair in zip(benign, unsafe):
        rows.extend(pair)
    return rows


def teacher_completion_ids(tokenizer, row: dict, device: torch.device) -> tuple[torch.Tensor, int]:
    prompt = base_rl.build_prompt(tokenizer, row["messages"])
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
    full = tokenizer(prompt + row["assistant_text"], return_tensors="pt", truncation=True, max_length=320)["input_ids"]
    return full.to(device), int(prompt_ids.shape[1])


def train_slider_rl(
    arm_name: str,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    head: GqaSliderHead,
    layer: int,
    steps: int = 18,
    group_size: int = 4,
    lr: float = 3e-3,
    beta: float = 0.02,
    bc_lambda: float = 0.0,
    max_new_tokens: int = 48,
    seed: int = 42,
) -> tuple[object, list[dict]]:
    """Train only the external head through the hook; all model weights remain frozen."""
    torch.manual_seed(seed)
    random.seed(seed)
    handle = install_slider_hook(model, layer, head)
    for param in model.parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    trace: list[dict] = []
    for step in range(steps):
        step_t0 = time.perf_counter()
        row = rows[step % len(rows)]
        prompt = base_rl.build_prompt(tokenizer, row["messages"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        prompt_len = int(enc["input_ids"].shape[1])
        model.eval()
        head.eval()
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.eos_token_id,
            )
        rewards = []
        seqs = []
        for seq in gen:
            text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            reward, _info = base_rl.score_completion(text, row["target"])
            rewards.append(float(reward))
            seqs.append(seq)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        adv = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)
        model.train()
        head.train()
        losses = []
        kls = []
        if float(r_t.std(unbiased=False).item()) >= 1e-6:
            for seq, advantage in zip(seqs, adv):
                ids = seq.unsqueeze(0)
                lp_cur = base_rl.completion_logprobs(model, ids, prompt_len)
                with torch.no_grad():
                    lp_ref = base_rl.completion_logprobs(ref_model, ids, prompt_len)
                kl = lp_cur - lp_ref
                kls.append(float(kl.abs().detach().item()))
                losses.append(-advantage.detach() * lp_cur + beta * kl * kl)
        bc_val = 0.0
        if bc_lambda > 0.0:
            teacher_ids, teacher_prompt_len = teacher_completion_ids(tokenizer, row, device)
            teacher_lp = base_rl.completion_logprobs(model, teacher_ids, teacher_prompt_len)
            bc_loss = -teacher_lp
            bc_val = float(bc_loss.detach().item())
            losses.append(bc_lambda * bc_loss)
        loss_val = 0.0
        grad_norm = 0.0
        if losses:
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0).item())
            opt.step()
            loss_val = float(loss.detach().item())
        rec = {
            "arm": arm_name,
            "step": step,
            "mean_reward": round(float(r_t.mean().item()), 4),
            "reward_std": round(float(r_t.std(unbiased=False).item()), 4),
            "alpha": round(float(head.alpha.detach().item()), 6),
            "loss": round(loss_val, 5),
            "bc_nll": round(bc_val, 5),
            "mean_abs_kl": round(sum(kls) / max(len(kls), 1), 5) if kls else 0.0,
            "grad_norm": round(grad_norm, 5),
            "step_wall_sec": round(time.perf_counter() - step_t0, 3),
        }
        trace.append(rec)
        if step % 6 == 0 or step == steps - 1:
            print(
                f"    [{arm_name}] step {step:02d}/{steps - 1} rew={rec['mean_reward']:.2f} "
                f"alpha={rec['alpha']:.4f} kl={rec['mean_abs_kl']:.4f} bc={rec['bc_nll']:.3f}",
                flush=True,
            )
    return handle, trace


def count_parameters(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())
