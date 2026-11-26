"""TransformerLens utilities for Halo activation probing/patching.

The container does not ship TransformerLens. Experiments call ensure_transformer_lens()
at startup so the GPU container installs the pinned HookedTransformer-compatible line
before probing. TransformerLens 4 removed HookedTransformer, so keep the <4 pin.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def ensure_transformer_lens() -> None:
    try:
        from transformer_lens import HookedTransformer  # noqa: F401
        return
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformer-lens<4"])


def load_hooked_hf_model(hf_model: torch.nn.Module, tokenizer, device: torch.device, dtype: torch.dtype = torch.float32):
    """Wrap an already merged Halo HF checkpoint in HookedTransformer.

    Weight folding/centering is disabled so probe geometry stays aligned with the
    served HF model used by validated_edit.py and the behavioral evaluator.
    """
    ensure_transformer_lens()
    from transformer_lens import HookedTransformer
    import transformer_lens.loading_from_pretrained as loading

    # SmolLM2 is Llama-architecture but absent from TransformerLens' allowlist.
    # Registering the exact HF id lets the generic LlamaForCausalLM config/state-dict
    # conversion path handle it while hf_model supplies the merged Halo weights.
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    if model_id not in loading.OFFICIAL_MODEL_NAMES:
        loading.OFFICIAL_MODEL_NAMES.append(model_id)

    return HookedTransformer.from_pretrained(
        model_id,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=str(device),
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        default_prepend_bos=False,
        move_to_device=True,
    )


def cache_text(model, text: str):
    """Tokenize and cache one text with TransformerLens; returns (tokens, logits, cache)."""
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    return tokens, logits, cache


def residual_hook_name(layer: int, position: str = "post") -> str:
    if position not in {"pre", "mid", "post"}:
        raise ValueError("position must be pre/mid/post")
    return f"blocks.{layer}.hook_resid_{position}"


def mlp_activation_hook_name(layer: int) -> str:
    return f"blocks.{layer}.mlp.hook_post"
