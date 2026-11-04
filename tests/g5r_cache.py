"""G5-R KV-cache construction and length guards (v1.0, 2026-11-04).

Why: GraniteMoeHybrid (IBM Granite 4.0) does NOT create a cache when
`use_cache=True` and `past_key_values=None`. It logs a warning and returns no cache.
Decode then silently attends ONLY to the current token. This was observed in the first
Granite 8K run (every decode record had T = 1; decode perplexity 2,967). Every driver must
therefore create the cache explicitly and check its length after prefill.
"""
from __future__ import annotations

import torch


def make_cache(model, batch_size: int):
    """Return an initialised cache appropriate for the model, or None (model default)."""
    cfg = model.config
    if getattr(cfg, "model_type", "") == "granitemoehybrid":
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
        return HybridMambaAttentionDynamicCache(cfg, batch_size, dtype=model.dtype, device="cpu")
    return None


def cache_len(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    raise TypeError(type(cache))


def kv_bytes(cache) -> int:
    n = 0
    if hasattr(cache, "key_cache") and isinstance(cache.key_cache, list):
        for t in list(cache.key_cache) + list(cache.value_cache):
            n += t.numel() * t.element_size()
        return n
    for layer in getattr(cache, "layers", []):
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if t is not None:
                n += t.numel() * t.element_size()
    return n


def require_len(cache, expected: int, where: str):
    got = cache_len(cache)
    if got != expected:
        raise RuntimeError(f"KV cache length guard failed at {where}: got {got}, expected {expected}")
