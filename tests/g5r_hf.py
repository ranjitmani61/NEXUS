#!/usr/bin/env python3
"""G5-R Hugging Face integration: real-model decode with NEXUS/Quest attention.

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md.

Registers an attention implementation "g5r" in transformers' AttentionInterface.
During prefill (query length > 1) it ALWAYS calls the stock SDPA implementation, so
every method sees an identical, dense prefill (as in Quest). During decode
(query length 1, batch 1) the controller mode decides:

  dense  : stock SDPA output
  shadow : stock SDPA output returned (model unchanged); all candidates/oracles are
           evaluated on the real q/K/V in float64 and recorded
  live   : the selected candidate's output replaces attention (errors propagate
           through later layers and later decode steps)

Recorded quantities are analysis-only; none of the float64 reference work is
charged to any candidate. Timing is measured separately (tests/g5r_timing.py).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

import g5r_core as C

try:  # transformers >= 4.53 separates mask creation by implementation name
    from transformers import AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
except Exception:  # pragma: no cover
    AttentionMaskInterface = None
    sdpa_mask = None

QUEST_BUDGETS = (512, 1024, 2048, 4096)
QUEST_DENSE_LAYERS = 2


@dataclass
class Controller:
    mode: str = "dense"              # dense | shadow | live
    method: str = ""                 # live method: nx_box, nx_sph, nx_val, nx_one, quest:<B>
    ps: int = 32                     # NEXUS page size (archived default)
    tol: float = C.TOL
    records: list = field(default_factory=list)
    step: int = 0
    capture_samples: list = field(default_factory=list)
    capture_prob: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(20261104))
    live_stats: list = field(default_factory=list)
    stale_test: bool = False


CTL = Controller()


def _dense(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)


def _to64(query, key, value):
    return query[0, :, 0, :].double(), key[0].double(), value[0].double()


def _shadow_record(layer: int, q, K, V, scale: float, sdpa_out: torch.Tensor):
    ps = CTL.ps
    ref = C.dense_reference(q, K, V, scale, ps)
    meta = C.build_meta(K, V, ps)
    ub_box = C.box_bound(q, meta, ref.g, scale)
    ub_sph = C.sphere_bound(q, meta, ref.g, scale)
    box = C.greedy_mass_certificate(ref, V, ub_box, CTL.tol)
    sph = C.greedy_mass_certificate(ref, V, ub_sph, CTL.tol)
    val = C.value_certificate(ref, V, meta, ub_box, CTL.tol)
    orm = C.oracle_mass(ref, CTL.tol)
    ors = C.oracle_score(ref, CTL.tol)
    ref16 = C.StepRef(scores=ref.scores, p=ref.p, out=ref.out, vmax=ref.vmax, g=ref.g,
                      T=ref.T, full=ref.T // 16, ps=16)
    meta16 = C.build_meta(K, V, 16)
    box16 = C.greedy_mass_certificate(ref16, V, C.box_bound(q, meta16, ref.g, scale), CTL.tol)
    orm16 = C.oracle_mass(ref16, CTL.tol)
    rec = {
        "layer": layer, "step": CTL.step, "T": ref.T, "full32": ref.full, "full16": ref16.full,
        "capture_maxabs": float((sdpa_out.double() - ref.out).abs().max()),
        "capture_outnorm": float(ref.out.norm(dim=-1).max()),
        "vmax": ref.vmax.numpy().copy(),
        "outnorm": ref.out.norm(dim=-1).numpy().copy(),
        "top1_page_mass": None,
    }
    for name, r in (("box", box), ("sph", sph), ("val", val), ("box16", box16)):
        rec[f"{name}_k"] = r["k"].numpy().copy()
        rec[f"{name}_cert"] = r["cert"].numpy().copy()
        rec[f"{name}_err"] = r["err"].numpy().copy()
        rec[f"{name}_errrel"] = r["err_rel"].numpy().copy()
        rec[f"{name}_viol"] = int(r["viol"].sum())
        if "bound_violations" in r:
            rec[f"{name}_boundviol"] = r["bound_violations"]
        if "naive_stops_earlier" in r:
            rec[f"{name}_naive_early"] = r["naive_stops_earlier"]
        if "sel" in r:
            rec[f"{name}_union"] = C.kv_union_pages(r["sel"], ref.g).numpy().copy()
    rec["val_grid_step"] = val["grid_step"]
    rec["ormass_k"] = orm.numpy().copy()
    rec["orscore_k"] = ors.numpy().copy()
    rec["ormass16_k"] = orm16.numpy().copy()
    for B in QUEST_BUDGETS:
        qr = C.quest_select(ref16, V, meta16, q, scale, B)
        rec[f"quest{B}_err"] = qr["err"].numpy().copy()
        rec[f"quest{B}_errrel"] = qr["err_rel"].numpy().copy()
        rec[f"quest{B}_union"] = qr["union_pages"].numpy().copy()
    # page-mass concentration diagnostics
    s = ref.scores.amax(-1)
    a, a_part, _ = C._page_mass_stats(ref, s)
    tot = a.sum(-1) + a_part
    rec["top1_page_mass"] = (a.amax(-1) / tot).numpy().copy() if ref.full else np.zeros(q.shape[0])
    rec["page0_mass"] = (a[:, 0] / tot).numpy().copy() if ref.full else np.zeros(q.shape[0])
    CTL.records.append(rec)
    if CTL.capture_prob > 0 and CTL.rng.random() < CTL.capture_prob:
        CTL.capture_samples.append({"layer": layer, "step": CTL.step, "scale": scale,
                                    "q": q.numpy().copy(), "K": K.numpy().copy(), "V": V.numpy().copy(),
                                    "box_k": rec["box_k"], "sph_k": rec["sph_k"], "ormass_k": rec["ormass_k"],
                                    "orscore_k": rec["orscore_k"], "box_err": rec["box_err"],
                                    "box_cert": rec["box_cert"]})


def _live_output(layer: int, q, K, V, scale: float):
    m = CTL.method
    if m.startswith("quest:"):
        B = int(m.split(":")[1])
        ref16 = C.dense_reference(q, K, V, scale, 16)
        if layer < QUEST_DENSE_LAYERS:
            return ref16.out, {"pages_frac": 1.0}
        meta16 = C.build_meta(K, V, 16)
        qr = C.quest_select(ref16, V, meta16, q, scale, B)
        f = max(ref16.full, 1)
        return qr["out"], {"pages_frac": float(qr["union_pages"].double().mean() / f)}
    ref = C.dense_reference(q, K, V, scale, CTL.ps)
    meta = C.build_meta(K, V, CTL.ps)
    if m in ("nx_box", "nx_one", "nx_val"):
        ub = C.box_bound(q, meta, ref.g, scale)
    elif m == "nx_sph":
        ub = C.sphere_bound(q, meta, ref.g, scale)
    else:
        raise ValueError(m)
    if m == "nx_val":
        r = C.value_certificate(ref, V, meta, ub, CTL.tol)
        k = r["k"]
    else:
        r = C.greedy_mass_certificate(ref, V, ub, CTL.tol)
        k = r["k"]
        if m == "nx_one":
            # one-page admission else dense fallback (archived G4 candidate)
            admit = k <= 1
            out = torch.where(admit[:, None], r["out"], ref.out)
            return out, {"pages_frac": float(torch.where(admit, k.double(), float(ref.full)).mean() / max(ref.full, 1)),
                         "admit": float(admit.double().mean()), "viol": int(r["viol"].sum())}
    return r["out"], {"pages_frac": float(k.double().mean() / max(ref.full, 1)), "viol": int(r["viol"].sum())}


def g5r_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    if CTL.mode == "dense" or query.shape[2] != 1:
        return _dense(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    if query.shape[0] != 1:
        raise NotImplementedError("analysis modes support batch 1 only")
    scale = float(scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1]))
    layer = int(getattr(module, "layer_idx", -1))
    q, K, V = _to64(query, key, value)
    if CTL.mode == "shadow":
        out, _ = _dense(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        _shadow_record(layer, q, K, V, scale, out[0, 0])
        return out, None
    if CTL.mode == "live":
        o, stats = _live_output(layer, q, K, V, scale)
        stats.update({"layer": layer, "step": CTL.step})
        CTL.live_stats.append(stats)
        return o.to(query.dtype)[None, None], None
    raise ValueError(CTL.mode)


def register():
    AttentionInterface.register("g5r", g5r_attention)
    if AttentionMaskInterface is not None and sdpa_mask is not None:
        AttentionMaskInterface.register("g5r", sdpa_mask)
