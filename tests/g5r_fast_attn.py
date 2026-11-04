#!/usr/bin/env python3
"""G5-R implementable (timed) attention paths for decode on CPU.

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md.

These are the paths whose wall-clock time is measured. They do NOT use the float64
analysis reference. Every candidate pays for: incremental page metadata updates,
bound computation, ordering, page gathers, exact scoring of visited pages,
certificate evaluation, fallback, and the final attention.

  dense      : stock transformers SDPA (PyTorch CPU kernel)
  quest:B    : Quest-style top-(B/16) pages by min/max bound, first 2 layers dense
  nx_box     : NEXUS certified min/max screening, page 32, log-domain certificate,
               doubling rounds (1,2,4,... pages) with incremental exact scoring;
               falls back to dense SDPA when all pages are needed
  nx_sph     : same with mean/radius bound
  nx_one     : one-page admission (all heads of a layer) else dense SDPA

Metadata lives in a per-layer dict and is updated whenever new pages fill, inside
the attention call, so its cost is inside the measured step time. Initial build after
prefill is triggered by the first decode step and is charged to that step (and also
reported separately via `build_seconds`).
"""
from __future__ import annotations

import math
import time

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

try:
    from transformers import AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
except Exception:  # pragma: no cover
    AttentionMaskInterface = sdpa_mask = None

TOL = 0.02


class State:
    method = "dense"
    meta: dict = {}
    attn_seconds = 0.0
    build_seconds = 0.0
    timing = False
    pages_read = 0
    pages_total = 0
    fallbacks = 0
    calls = 0
    dense_calls = 0
    dense_native_gqa_calls = 0


ST = State()


def reset(method: str):
    ST.method = method
    ST.meta = {}
    ST.attn_seconds = ST.build_seconds = 0.0
    ST.pages_read = ST.pages_total = ST.fallbacks = ST.calls = 0
    ST.dense_calls = ST.dense_native_gqa_calls = 0


def _update_meta(layer, key, ps, need_sphere):
    """key [B, Hkv, T, D]. Build/extend full-page metadata incrementally."""
    B, hkv, T, d = key.shape
    full = T // ps
    m = ST.meta.get((layer, ps))
    built = 0 if m is None else m["full"]
    if full > built:
        t0 = time.perf_counter()
        kp = key[:, :, built * ps: full * ps].reshape(B, hkv, full - built, ps, d).float()
        new = {"kmin": kp.amin(3), "kmax": kp.amax(3)}
        if need_sphere:
            c = kp.mean(3)
            new["kmean"] = c
            new["krad"] = (kp - c[:, :, :, None]).norm(dim=-1).amax(3)
        if m is None:
            m = {k: v for k, v in new.items()}
        else:
            for k, v in new.items():
                m[k] = torch.cat([m[k], v], 2)
        m["full"] = full
        ST.meta[(layer, ps)] = m
        ST.build_seconds += time.perf_counter() - t0
    return m, full


def _bounds(qf, m, kind):
    """qf [B, Hkv, G, D] float32 (already scaled). returns [B, Hkv, G, P]."""
    if kind == "box":
        return (torch.einsum("bkgd,bkpd->bkgp", qf.clamp(min=0), m["kmax"]) +
                torch.einsum("bkgd,bkpd->bkgp", qf.clamp(max=0), m["kmin"]))
    return torch.einsum("bkgd,bkpd->bkgp", qf, m["kmean"]) + qf.norm(dim=-1, keepdim=True) * m["krad"][:, :, None, :]


def _gather_pages(x, idx, ps):
    """x [B, Hkv, T, D]; idx [B, Hkv, G, k] page ids -> [B, Hkv, G, k*ps, D]."""
    B, hkv, T, d = x.shape
    G, k = idx.shape[2], idx.shape[3]
    P = idx.max().item() + 1 if idx.numel() else 0
    xp = x[:, :, : (T // ps) * ps].reshape(B, hkv, T // ps, ps, d)
    ix = idx.reshape(B, hkv, G * k)[..., None, None].expand(-1, -1, -1, ps, d)
    g = torch.gather(xp, 2, ix)                                        # [B, Hkv, G*k, ps, d]
    return g.reshape(B, hkv, G, k * ps, d)


def _attend(qf, Ks, Vs, mask=None):
    """qf [B,Hkv,G,D] float32 scaled; Ks/Vs [B,Hkv,G,L,D] -> out [B,Hkv,G,Dv]."""
    s = torch.einsum("bkgd,bkgld->bkgl", qf, Ks.float())
    if mask is not None:
        s = s.masked_fill(~mask, -math.inf)
    p = torch.softmax(s, -1)
    return torch.einsum("bkgl,bkgld->bkgd", p, Vs.float())


def _mask_is_noop(mask, klen):
    if mask is None:
        return True
    m = mask[..., :klen]
    return bool(m.all()) if m.dtype == torch.bool else bool((m == 0).all())


def _dense(module, query, key, value, attention_mask, scaling, **kw):
    """Dense decode baseline (v1.1, 2026-11-04).

    v1.0 always called the transformers SDPA wrapper. For IBM Granite (whose legacy mask
    builder skips the mask only when the implementation is literally named "sdpa"), every
    decode step received a 4D mask, so the wrapper took its repeat_kv path (5x K/V copies)
    and a masked kernel: 37.1 ms against 7.9 ms per layer at 8K. That handicapped the Granite
    dense baseline. For a single decode query with no masked positions the mask is a no-op,
    so torch SDPA with native GQA is called directly. Qwen already received mask=None
    (native GQA), so its v1.0 timings are unaffected."""
    if query.shape[2] == 1:
        ST.dense_calls += 1
    if query.shape[2] == 1 and _mask_is_noop(attention_mask, key.shape[2]):
        ST.dense_native_gqa_calls += 1
        o = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=scaling, enable_gqa=True)
        return o.transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kw)


def fast_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if query.shape[2] != 1 or ST.method == "dense":
        if ST.timing and query.shape[2] == 1:
            t0 = time.perf_counter()
            out = _dense(module, query, key, value, attention_mask, scaling, **kwargs)
            ST.attn_seconds += time.perf_counter() - t0
            return out
        return _dense(module, query, key, value, attention_mask, scaling, **kwargs)
    t0 = time.perf_counter()
    out = _decode(module, query, key, value, attention_mask, scaling, **kwargs)
    ST.attn_seconds += time.perf_counter() - t0
    ST.calls += 1
    return out


def _decode(module, query, key, value, attention_mask, scaling, **kwargs):
    layer = int(module.layer_idx)
    B, hq, _, d = query.shape
    hkv, T = key.shape[1], key.shape[2]
    G = hq // hkv
    scale = float(scaling)
    meth = ST.method
    if meth.startswith("quest:"):
        budget = int(meth.split(":")[1])
        ps = 16
        if layer < 2 or T // ps <= budget // ps:
            ST.pages_read += T // ps; ST.pages_total += T // ps
            return _dense(module, query, key, value, attention_mask, scaling, **kwargs)
        m, full = _update_meta(layer, key, ps, False)
        qf = query[:, :, 0].float().reshape(B, hkv, G, d) * scale
        ub = _bounds(qf, m, "box")
        idx = torch.topk(ub, budget // ps, dim=-1).indices
        Ks, Vs = _gather_pages(key, idx, ps), _gather_pages(value, idx, ps)
        if T > full * ps:
            kp = key[:, :, full * ps:][:, :, None].expand(-1, -1, G, -1, -1)
            vp = value[:, :, full * ps:][:, :, None].expand(-1, -1, G, -1, -1)
            Ks, Vs = torch.cat([Ks, kp], 3), torch.cat([Vs, vp], 3)
        o = _attend(qf, Ks, Vs)
        ST.pages_read += idx.shape[-1]; ST.pages_total += full
        return o.reshape(B, hq, 1, -1).transpose(1, 2).to(query.dtype), None
    if meth in ("nx_box", "nx_sph", "nx_one"):
        ps = 32
        kind = "sph" if meth == "nx_sph" else "box"
        m, full = _update_meta(layer, key, ps, kind == "sph")
        if full == 0:
            return _dense(module, query, key, value, attention_mask, scaling, **kwargs)
        qf = query[:, :, 0].float().reshape(B, hkv, G, d) * scale
        ub = _bounds(qf, m, kind)                                   # [B,Hkv,G,P]
        lbeta = math.log(ps) + ub
        order = torch.argsort(ub, dim=-1, descending=True)
        lb_s = torch.gather(lbeta, -1, order)
        logU_suffix = torch.logcumsumexp(lb_s.flip(-1), -1).flip(-1)   # [B,Hkv,G,P]
        # partial page exact log-mass
        if T > full * ps:
            sp = torch.einsum("bkgd,bktd->bkgt", qf, key[:, :, full * ps:].float())
            lpart = torch.logsumexp(sp, -1)
        else:
            lpart = torch.full(ub.shape[:-1], -math.inf)
        kmax_round = 1 if meth == "nx_one" else full
        k = 1
        la_s = None
        while True:
            k = min(k, full)
            idx = order[..., :k]
            prev = 0 if la_s is None else la_s.shape[-1]
            if k > prev:
                Ks_new = _gather_pages(key, idx[..., prev:k], ps)                 # [B,Hkv,G,(k-prev)*ps,D]
                s_new = torch.einsum("bkgd,bkgld->bkgl", qf, Ks_new.float())
                la_new = torch.logsumexp(s_new.reshape(*s_new.shape[:-1], k - prev, ps), -1)
                la_s = la_new if la_s is None else torch.cat([la_s, la_new], -1)
            logZ = torch.logcumsumexp(torch.cat([lpart[..., None], la_s], -1), -1)   # [..,k+1]
            logU = torch.cat([logU_suffix[..., :k], torch.full_like(lpart[..., None], -math.inf)], -1)
            logU = logU if k < full else logU
            if k < full:
                logU[..., k] = logU_suffix[..., k]
            cert = 2.0 / (1.0 + torch.exp(logZ - logU))
            ok = cert <= TOL
            passed = ok.any(-1)
            if bool(passed.all()) or k >= kmax_round:
                break
            k *= 2
        if not bool(passed.all()):
            ST.fallbacks += 1
            ST.pages_read += full; ST.pages_total += full
            return _dense(module, query, key, value, attention_mask, scaling, **kwargs)
        j = torch.argmax(ok.int(), -1)                                  # first passing prefix per head
        kk = int(j.max().item())
        if kk == full and meth != "nx_one":
            ST.pages_read += full; ST.pages_total += full
            return _dense(module, query, key, value, attention_mask, scaling, **kwargs)
        idx = order[..., :max(kk, 1)]
        Ks, Vs = _gather_pages(key, idx, ps), _gather_pages(value, idx, ps)
        pos = torch.arange(idx.shape[-1]).repeat_interleave(ps)
        mask = pos[None, None, None, :] < j[..., None]
        if T > full * ps:
            kp = key[:, :, full * ps:][:, :, None].expand(-1, -1, G, -1, -1)
            vp = value[:, :, full * ps:][:, :, None].expand(-1, -1, G, -1, -1)
            Ks, Vs = torch.cat([Ks, kp], 3), torch.cat([Vs, vp], 3)
            mask = torch.cat([mask, torch.ones(*mask.shape[:-1], T - full * ps, dtype=torch.bool)], -1)
        o = _attend(qf, Ks, Vs, mask)
        ST.pages_read += idx.shape[-1]; ST.pages_total += full
        return o.reshape(B, hq, 1, -1).transpose(1, 2).to(query.dtype), None
    raise ValueError(meth)


def register():
    AttentionInterface.register("g5r_fast", fast_attention)
    if AttentionMaskInterface is not None:
        AttentionMaskInterface.register("g5r_fast", sdpa_mask)
