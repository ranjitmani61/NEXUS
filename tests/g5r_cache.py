"""G5-R KV-cache construction and length guards (v1.0, 2026-11-04).

Why: GraniteMoeHybrid (IBM Granite 4.0) does NOT create a cache when
`use_cache=True` and `past_key_values=None`. It logs a warning and returns no cache.
Decode then silently attends ONLY to the current token. This was observed in the first
Granite 8K run (every decode record had T = 1; decode perplexity 2,967). Every driver must
therefore create the cache explicitly and check its length after prefill.
"""
from __future__ import annotations

import torch


_FIXED = None


def _fixed_hybrid_cache_class():
    """Second Granite defect (found 2026-11-04 by per-layer q/K/V probing).

    transformers 4.57.1 HybridMambaAttentionDynamicCache.get_seq_length() returns
    key_cache[layer].shape[-2] even while the layer still holds its placeholder
    torch.tensor([[]] * batch_size) of shape (batch, 0), so an EMPTY cache reports
    length == batch_size (1 or 4), not 0. A prefill into a fresh cache then starts its
    positions at batch_size: every cached key is rotated with RoPE positions shifted by
    batch_size relative to later queries. Layer-0 prefix keys differed by 15.9% while
    values were bit-identical. Decode perplexity on PG-19 book 3340 at 8K was 298,
    against 8.3 for the cache-free forward. HF generate() passes cache_position
    explicitly, which hides the bug. Fix: an empty layer reports length 0.
    """
    global _FIXED
    if _FIXED is None:
        from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache

        class FixedHybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
            def get_seq_length(self, layer_idx=0):
                layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
                if len(self.key_cache) <= layer_idx:
                    return 0
                t = self.key_cache[layer_idx]
                if t.dim() < 4 or t.numel() == 0:
                    return 0
                return t.shape[-2]

        _FIXED = FixedHybridMambaAttentionDynamicCache
    return _FIXED


def make_cache(model, batch_size: int):
    """Return an initialised cache appropriate for the model, or None (model default)."""
    cfg = model.config
    if getattr(cfg, "model_type", "") == "granitemoehybrid":
        return _fixed_hybrid_cache_class()(cfg, batch_size, dtype=model.dtype, device="cpu")
    return None


@__import__("torch").no_grad()
def decode_consistency(model, ids, T: int, steps: int = 3) -> dict:
    """Standing check: logits from prefill(T)+cached decode must match one full forward.
    ids: 1-D LongTensor with >= T + steps tokens. Returns max |logit diff| and max KL."""
    import torch
    full = model(input_ids=ids[: T + steps].view(1, -1), use_cache=False, logits_to_keep=steps).logits[0].double()
    pre = model(input_ids=ids[:T].view(1, -1), past_key_values=make_cache(model, 1), use_cache=True, logits_to_keep=1)
    require_len(pre.past_key_values, T, "decode_consistency prefill")
    cache, dec = pre.past_key_values, []
    for i in range(steps):
        # decode step i consumes ids[T+i] at position T+i; full-forward logits row i is position T+i
        o = model(input_ids=ids[T + i: T + i + 1].view(1, 1), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        dec.append(o.logits[0, -1].double())
    dec = torch.stack(dec)
    ref = full
    lf, ld = torch.log_softmax(ref, -1), torch.log_softmax(dec, -1)
    return {"T": T, "steps": steps, "max_abs_logit_diff": float((dec - ref).abs().max()),
            "max_kl": float((lf.exp() * (lf - ld)).sum(-1).max()),
            "argmax_agree": float((dec.argmax(-1) == ref.argmax(-1)).double().mean())}


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
