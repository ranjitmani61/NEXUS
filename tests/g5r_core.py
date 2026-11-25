#!/usr/bin/env python3
"""G5-R core math: NEXUS page certificates, oracles and Quest on one decode step.

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md.

Everything operates on ONE layer at ONE decode position (query length 1) for one
sequence, with tensors already moved to float64:

  q      [Hq, D]        post-RoPE query of the new token, per query head
  K      [Hkv, T, D]    post-RoPE key cache including the new token
  V      [Hkv, T, Dv]   value cache including the new token
  scale  float          softmax scaling used by the model (e.g. 1/sqrt(D))

Grouped-query attention: query head h reads KV head h // (Hq // Hkv).

Page layout: pages of `ps` consecutive tokens from position 0. The last
incomplete page ("partial", length T mod ps, possibly 0) is ALWAYS read exactly by
every method, as in the archived specs and in Quest. Metadata exists for full
pages only.

Certificate (archived G0/G4 definition): visited exact mass Z (shifted by s),
unvisited bound mass U = sum_j n_j exp(u_j - s); output error normalised by
Vmax = max_i ||v_i|| (visible tokens of that KV head) is <= 2U/(Z+U) in exact
arithmetic.

This module computes, for analysis only, the FULL dense attention as reference.
That reference is never counted as part of any candidate's cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TOL = 0.02


def group_index(hq: int, hkv: int, device=None) -> torch.Tensor:
    if hq % hkv:
        raise ValueError("Hq must be a multiple of Hkv")
    return torch.arange(hq, device=device) // (hq // hkv)


@dataclass
class PageMeta:
    """Per-layer full-page metadata, [Hkv, P, ...]."""
    kmin: torch.Tensor
    kmax: torch.Tensor
    kmean: torch.Tensor
    krad: torch.Tensor   # max_i ||k_i - kmean||
    vmean: torch.Tensor
    vrad: torch.Tensor   # max_i ||v_i - vmean||
    full: int
    ps: int


def build_meta(K: torch.Tensor, V: torch.Tensor, ps: int, full: int | None = None) -> PageMeta:
    hkv, t, d = K.shape
    if full is None:
        full = t // ps
    Kp = K[:, : full * ps].reshape(hkv, full, ps, d)
    Vp = V[:, : full * ps].reshape(hkv, full, ps, V.shape[-1])
    kmean = Kp.mean(2)
    vmean = Vp.mean(2)
    return PageMeta(
        kmin=Kp.amin(2), kmax=Kp.amax(2), kmean=kmean,
        krad=(Kp - kmean[:, :, None]).norm(dim=-1).amax(2),
        vmean=vmean, vrad=(Vp - vmean[:, :, None]).norm(dim=-1).amax(2),
        full=full, ps=ps)


def box_bound(q: torch.Tensor, meta: PageMeta, g: torch.Tensor, scale: float) -> torch.Tensor:
    """sum_d max(q_d kmin_d, q_d kmax_d) * scale, [Hq, P]. Valid for scale > 0."""
    qp, qn = q.clamp(min=0), q.clamp(max=0)
    return (torch.einsum("hd,hpd->hp", qp, meta.kmax[g]) + torch.einsum("hd,hpd->hp", qn, meta.kmin[g])) * scale


def sphere_bound(q: torch.Tensor, meta: PageMeta, g: torch.Tensor, scale: float) -> torch.Tensor:
    """(q·c + ||q|| r) * scale, [Hq, P]."""
    return (torch.einsum("hd,hpd->hp", q, meta.kmean[g]) + q.norm(dim=-1, keepdim=True) * meta.krad[g]) * scale


def _first_true(mask: torch.Tensor) -> torch.Tensor:
    """Index of first True along last dim; mask must have >=1 True per row."""
    idx = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    big = torch.full_like(idx, mask.shape[-1] + 1)
    return torch.where(mask, idx, big).amin(-1)


def _gather_out(w: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """sum_t w[h,t] V[g(h),t,:] without replicating V per query head. [Hq, Dv]."""
    hkv = V.shape[0]
    hq = w.shape[0]
    return torch.einsum("kgt,ktd->kgd", w.reshape(hkv, hq // hkv, -1), V).reshape(hq, -1)


@dataclass
class StepRef:
    """Dense reference and exact page statistics for one decode step."""
    scores: torch.Tensor   # [Hq, T]
    p: torch.Tensor        # [Hq, T] exact softmax probabilities
    out: torch.Tensor      # [Hq, Dv]
    vmax: torch.Tensor     # [Hq]
    g: torch.Tensor
    T: int
    full: int
    ps: int


def dense_reference(q, K, V, scale, ps) -> StepRef:
    hq, hkv = q.shape[0], K.shape[0]
    g = group_index(hq, hkv, q.device)
    G = hq // hkv
    scores = torch.einsum("kgd,ktd->kgt", q.reshape(hkv, G, -1), K).reshape(hq, -1) * scale
    p = torch.softmax(scores, dim=-1)
    out = _gather_out(p, V)
    vmax = V.norm(dim=-1).amax(-1)[g]
    T = K.shape[1]
    return StepRef(scores=scores, p=p, out=out, vmax=vmax, g=g, T=T, full=T // ps, ps=ps)


def _selected_output(ref: StepRef, V, page_sel: torch.Tensor) -> torch.Tensor:
    """page_sel: bool [Hq, full]; partial page always included. Returns [Hq, Dv]."""
    hq = ref.p.shape[0]
    tok = torch.ones(hq, ref.T, dtype=torch.bool, device=ref.p.device)
    if ref.full:
        tok[:, : ref.full * ref.ps] = page_sel.repeat_interleave(ref.ps, dim=1)
    w = ref.p * tok
    return _gather_out(w, V) / w.sum(-1, keepdim=True)


def _err(ref: StepRef, out_s: torch.Tensor):
    diff = (out_s - ref.out).norm(dim=-1)
    return diff / ref.vmax, diff / ref.out.norm(dim=-1).clamp_min(1e-300)


def _page_mass_stats(ref: StepRef, s: torch.Tensor):
    """Exact shifted masses: full pages [Hq, full], partial [Hq], page max score [Hq, full]."""
    e = torch.exp(ref.scores - s[:, None])
    f, ps = ref.full, ref.ps
    if f:
        a = e[:, : f * ps].reshape(-1, f, ps).sum(-1)
        m = ref.scores[:, : f * ps].reshape(-1, f, ps).amax(-1)
    else:
        a = e.new_zeros(e.shape[0], 0)
        m = e.new_zeros(e.shape[0], 0)
    a_part = e[:, f * ps:].sum(-1)
    return a, a_part, m


def _log_page_masses(ref: StepRef):
    """log of unnormalised page masses: full pages [Hq, f], partial [Hq] (-inf if empty)."""
    hq, f, ps = ref.scores.shape[0], ref.full, ref.ps
    la = (torch.logsumexp(ref.scores[:, : f * ps].reshape(hq, f, ps), -1) if f
          else ref.scores.new_zeros(hq, 0))
    lp = (torch.logsumexp(ref.scores[:, f * ps:], -1) if ref.T > f * ps
          else torch.full((hq,), -math.inf, dtype=ref.scores.dtype))
    return la, lp


def greedy_mass_certificate(ref: StepRef, V, ub: torch.Tensor, tol: float = TOL) -> dict:
    """NEXUS greedy screening: visit pages by descending bound until 2U/(Z+U) <= tol.

    v1.1 (2026-11-04): evaluated in the LOG DOMAIN. v1.0 followed the archived pattern
    (shift by max upper bound, then exp), which underflows to U = 0 when bounds are
    very loose (observed on real Qwen2.5 layer-0 keys) and then reports a false
    certificate of 0. The naive evaluation is still computed and returned as
    `naive_false_cert` (cases where naive cert < true error) for the adversarial record.
    Returns pages visited k (full pages), certificate, true error, violation flags.
    """
    hq, f = ub.shape
    la, lp = _log_page_masses(ref)
    m = (ref.scores[:, : f * ref.ps].reshape(hq, f, ref.ps).amax(-1) if f else ub)
    bound_violation = int((m > ub + 1e-9 * (1 + ub.abs())).sum().item()) if f else 0
    order = torch.argsort(ub, dim=-1, descending=True)
    lb_s = torch.gather(math.log(ref.ps) + ub, 1, order)          # log bound masses, sorted
    la_s = torch.gather(la, 1, order)
    logZ = torch.logcumsumexp(torch.cat([lp[:, None], la_s], -1), -1)            # [Hq, f+1]
    ninf = torch.full((hq, 1), -math.inf, dtype=ub.dtype)
    logU = torch.cat([torch.logcumsumexp(lb_s.flip(-1), -1).flip(-1), ninf], -1)  # [Hq, f+1]
    cert = 2.0 / (1.0 + torch.exp(logZ - logU))
    cert[:, -1] = 0.0
    k = _first_true(cert <= tol)
    sel = torch.zeros(hq, f, dtype=torch.bool)
    if f:
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(f).expand(hq, f).contiguous())
        sel = rank < k[:, None]
    out_s = _selected_output(ref, V, sel)
    e, e_rel = _err(ref, out_s)
    c_at_k = torch.gather(cert, 1, k[:, None])[:, 0]
    viol = e > c_at_k * (1 + 1e-6) + 1e-12
    # naive (archived-pattern) evaluation, diagnostic only
    part_max = ref.scores[:, f * ref.ps:].amax(-1) if ref.T > f * ref.ps else torch.full((hq,), -math.inf, dtype=ub.dtype)
    s = torch.maximum(ub.amax(-1) if f else torch.full((hq,), -math.inf, dtype=ub.dtype), part_max)
    beta_s = torch.exp(lb_s - s[:, None])
    a_s = torch.exp(la_s - s[:, None])
    zero = ub.new_zeros(hq, 1)
    Zn = torch.exp(lp - s)[:, None] + torch.cat([zero, a_s.cumsum(-1)], -1)
    Un = (beta_s.sum(-1, keepdim=True) - torch.cat([zero, beta_s.cumsum(-1)], -1)).clamp_min(0)
    cn = torch.nan_to_num(2 * Un / (Zn + Un), nan=2.0)
    cn[:, -1] = 0.0
    kn = _first_true(cn <= tol)
    naive_stop_early = kn < k
    return {"k": k, "cert": c_at_k, "err": e, "err_rel": e_rel, "viol": viol,
            "bound_violations": bound_violation, "sel": sel, "out": out_s,
            "naive_k": kn, "naive_stops_earlier": int(naive_stop_early.sum()),
            "slack_mean": (ub - m).mean().item() if f else 0.0}


def oracle_mass(ref: StepRef, tol: float = TOL) -> torch.Tensor:
    """OR-MASS: minimum full pages (partial forced) with exact omitted mass so that
    2u/(Z+u) <= tol, i.e. visited mass >= (1 - tol/2) of total."""
    s = ref.scores.amax(-1)
    a, a_part, _ = _page_mass_stats(ref, s)
    total = a.sum(-1) + a_part
    a_s = torch.sort(a, dim=-1, descending=True).values
    cum = a_part[:, None] + torch.cat([a.new_zeros(a.shape[0], 1), a_s.cumsum(-1)], -1)
    return _first_true(cum >= (1 - tol / 2) * total[:, None] * (1 - 1e-12))


def oracle_score(ref: StepRef, tol: float = TOL) -> torch.Tensor:
    """OR-SCORE: exact lower bound on pages for ANY mass-certificate method whose
    per-page bound is >= the exact page max score (contract §7). Partial forced."""
    s = ref.scores.amax(-1)
    a, a_part, m = _page_mass_stats(ref, s)
    b = ref.ps * torch.exp(m - s[:, None])
    c = tol / (2 - tol)
    w_s = torch.sort(b + c * a, dim=-1, descending=True).values
    need = b.sum(-1) - c * a_part
    cum = torch.cat([a.new_zeros(a.shape[0], 1), w_s.cumsum(-1)], -1)
    return _first_true(cum >= need[:, None] * (1 - 1e-12))


def value_certificate(ref: StepRef, V, meta: PageMeta, ub: torch.Tensor, tol: float = TOL,
                      exact_grid_max: int = 64, bisect_iters: int = 48) -> dict:
    """NX-VAL: greedy by box bound; certificate = max_{0<=u<=beta} (u·d)/(Z+sum u) / Vmax
    with d_j = ||vmean_j - y_S|| + vrad_j. Evaluated on k-grid: every k if full <= exact_grid_max,
    else all k <= 16 plus ceil(full/64)-spaced k up to full. Reported k is the first passing grid k
    (conservative by at most the grid spacing)."""
    hq, f = ub.shape
    if f == 0:
        out_s = _selected_output(ref, V, torch.zeros(hq, 0, dtype=torch.bool))
        e, e_rel = _err(ref, out_s)
        return {"k": torch.zeros(hq, dtype=torch.long), "cert": torch.zeros(hq), "err": e,
                "err_rel": e_rel, "viol": torch.zeros(hq, dtype=torch.bool), "grid_step": 1}
    # v1.1: masses/numerators from the exactly normalised softmax p (no underflow from a
    # loose-bound shift); bound masses kept in the log domain and rescaled per grid point.
    lse = torch.logsumexp(ref.scores, -1)
    e_tok = ref.p
    hkv, dv = V.shape[0], V.shape[-1]
    G = hq // hkv
    page_num = torch.einsum("kgfp,kfpd->kgfd", e_tok[:, : f * ref.ps].reshape(hkv, G, f, ref.ps),
                            V[:, : f * ref.ps].reshape(hkv, f, ref.ps, dv)).reshape(hq, f, dv)
    part_num = _gather_out(e_tok[:, f * ref.ps:], V[:, f * ref.ps:])          # [Hq, Dv]
    a = e_tok[:, : f * ref.ps].reshape(hq, f, ref.ps).sum(-1)
    a_part = e_tok[:, f * ref.ps:].sum(-1)
    order = torch.argsort(ub, dim=-1, descending=True)
    lbeta_s = torch.gather(math.log(ref.ps) + ub - lse[:, None], 1, order)   # log bound mass in p-units
    a_s = torch.gather(a, 1, order)
    num_s = torch.gather(page_num, 1, order[:, :, None].expand(-1, -1, page_num.shape[-1]))
    cv = torch.gather(meta.vmean[ref.g], 1, order[:, :, None].expand(-1, -1, meta.vmean.shape[-1]))
    rv = torch.gather(meta.vrad[ref.g], 1, order)
    if f <= exact_grid_max:
        grid = torch.arange(f + 1)
        step = 1
    else:
        step = max(1, math.ceil(f / 64))
        grid = torch.unique(torch.cat([torch.arange(17), torch.arange(0, f + 1, step), torch.tensor([f])]))
    zero1 = a.new_zeros(hq, 1)
    Zc = a_part[:, None] + torch.cat([zero1, a_s.cumsum(-1)], -1)              # [Hq, f+1]
    Nc = part_num[:, None] + torch.cat([part_num.new_zeros(hq, 1, part_num.shape[-1]), num_s.cumsum(1)], 1)
    Zg = Zc[:, grid]                                                           # [Hq, G]
    Y = Nc[:, grid] / Zg.clamp_min(1e-300)[:, :, None]                         # [Hq, G, Dv]
    d2 = (cv.pow(2).sum(-1)[:, None, :] + Y.pow(2).sum(-1)[:, :, None] - 2 * torch.einsum("hgd,hpd->hgp", Y, cv)).clamp_min(0)
    dist = d2.sqrt() + rv[:, None, :]                                          # [Hq, G, f]
    unvisited = torch.arange(f)[None, None, :] >= grid[None, :, None]
    lb = torch.where(unvisited, lbeta_s[:, None, :], torch.full_like(dist, -math.inf))
    L = torch.maximum(torch.log(Zg.clamp_min(1e-300)), lb.amax(-1))            # per (head, grid) rescale
    bet = torch.exp(lb - L[..., None])
    Zg = torch.exp(torch.log(Zg.clamp_min(1e-300)) - L)
    lo = torch.zeros_like(Zg)
    hi = (dist * unvisited).amax(-1)
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        lhs = (bet * (dist - mid[..., None]).clamp_min(0)).sum(-1)
        go_up = lhs > mid * Zg
        lo = torch.where(go_up, mid, lo)
        hi = torch.where(go_up, hi, mid)
    cert_g = hi / ref.vmax[:, None]
    cert_g[:, -1] = 0.0 if grid[-1] == f else cert_g[:, -1]
    pos = _first_true(cert_g <= tol)
    k = grid[pos]
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(f).expand(hq, f).contiguous())
    sel = rank < k[:, None]
    out_s = _selected_output(ref, V, sel)
    e, e_rel = _err(ref, out_s)
    c_at_k = torch.gather(cert_g, 1, pos[:, None])[:, 0]
    viol = e > c_at_k * (1 + 1e-6) + 1e-12
    return {"k": k, "cert": c_at_k, "err": e, "err_rel": e_rel, "viol": viol, "grid_step": step, "out": out_s}


def quest_select(ref: StepRef, V, meta16: PageMeta, q, scale, budget: int) -> dict:
    """Quest-style: top-(budget // ps) full pages by box bound, partial page always read."""
    hq = q.shape[0]
    f = meta16.full
    kpages = min(f, budget // meta16.ps)
    sel = torch.zeros(hq, f, dtype=torch.bool)
    if kpages > 0:
        ub = box_bound(q, meta16, ref.g, scale)
        top = torch.topk(ub, kpages, dim=-1).indices
        sel.scatter_(1, top, True)
    out_s = _selected_output(ref, V, sel)
    e, e_rel = _err(ref, out_s)
    hkv = int(ref.g.max().item()) + 1
    union = torch.zeros(hkv, f, dtype=torch.bool)
    for kv in range(hkv):
        union[kv] = sel[ref.g == kv].any(0)
    return {"k": torch.full((hq,), kpages), "err": e, "err_rel": e_rel, "out": out_s,
            "union_pages": union.sum(-1), "sel": sel}


def kv_union_pages(sel: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Pages a KV head must actually read = union over its query heads. [Hkv]."""
    hkv = int(g.max().item()) + 1
    return torch.stack([sel[g == kv].any(0).sum() for kv in range(hkv)])
