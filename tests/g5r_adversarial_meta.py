#!/usr/bin/env python3
"""G5-R adversarial metadata tests on REAL captured (q, K, V) decode samples.

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md §10.

A. Stale metadata via page-slot reuse: in paged KV caches (vLLM-style block reuse),
   a page slot can receive new tokens while metadata from its previous contents
   survives if an update is missed. Simulated by replacing the metadata of page j
   with the metadata of another page of the same cache (real keys, wrong page).
B. Unguarded reduced-precision metadata: kmin/kmax rounded to nearest BF16 / FP16,
   bounds then computed in float64. Rounding can move kmax DOWN / kmin UP -> the bound
   can fall below the true page max score.
C. Guarded control: outward-rounded BF16 (kmin rounded down, kmax rounded up).
A2 (v1.1). Worst-case stale: stale pages carry the smallest-box page's metadata.
Also reports bound slack (upper bound minus exact page max score, in nats).

Counts bound violations (bound < exact page max score) and FALSE certificates
(true normalised error > reported certificate). Real data, offline, no timing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_core as C  # noqa: E402


def round_outward_bf16(x, up: bool):
    r = x.to(torch.bfloat16).double()
    m, e = torch.frexp(r)
    ulp = torch.ldexp(torch.ones_like(r), (e - 8).to(torch.int32))
    if up:
        return torch.where(r < x, r + ulp, r)
    return torch.where(r > x, r - ulp, r)


def variants(meta: C.PageMeta, rng):
    out = {}
    f = meta.full
    if f >= 2:
        perm = torch.tensor(rng.permutation(f))
        stale = C.PageMeta(**{**meta.__dict__})
        j = rng.integers(0, f, size=max(1, f // 8))           # 1/8 of pages carry stale metadata
        src = perm[j]
        kmin, kmax = meta.kmin.clone(), meta.kmax.clone()
        kmin[:, j], kmax[:, j] = meta.kmin[:, src], meta.kmax[:, src]
        stale.kmin, stale.kmax = kmin, kmax
        out["stale_slot_reuse"] = stale
        # v1.1 worst case: the stale pages carry the metadata of the page whose box is
        # smallest (lowest summed width), i.e. the most restrictive metadata present.
        width = (meta.kmax - meta.kmin).sum(-1)                # [Hkv, f]
        worst = C.PageMeta(**{**meta.__dict__})
        kmin, kmax = meta.kmin.clone(), meta.kmax.clone()
        for kv in range(meta.kmin.shape[0]):
            w = int(torch.argmin(width[kv]))
            kmin[kv, j], kmax[kv, j] = meta.kmin[kv, w], meta.kmax[kv, w]
        worst.kmin, worst.kmax = kmin, kmax
        out["stale_worst_smallest_box"] = worst
    for name, dt in (("bf16_nearest", torch.bfloat16), ("fp16_nearest", torch.float16)):
        m = C.PageMeta(**{**meta.__dict__})
        m.kmin, m.kmax = meta.kmin.to(dt).double(), meta.kmax.to(dt).double()
        out[name] = m
    g = C.PageMeta(**{**meta.__dict__})
    g.kmin, g.kmax = round_outward_bf16(meta.kmin, False), round_outward_bf16(meta.kmax, True)
    out["bf16_outward_guarded"] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+", type=Path)
    ap.add_argument("--ps", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(20261104)
    agg = {}
    slack, spread = {}, []
    n_samples = 0
    for p in a.samples:
        z = np.load(p, allow_pickle=True)
        for i in range(int(z["n"])):
            q = torch.from_numpy(z[f"q{i}"]); K = torch.from_numpy(z[f"K{i}"]); V = torch.from_numpy(z[f"V{i}"])
            sc = float(z[f"scale{i}"])
            ref = C.dense_reference(q, K, V, sc, a.ps)
            if ref.full < 2:
                continue
            n_samples += 1
            meta = C.build_meta(K, V, a.ps)
            ubb = C.box_bound(q, meta, ref.g, sc); ubs = C.sphere_bound(q, meta, ref.g, sc)
            m_exact = ref.scores[:, : ref.full * a.ps].reshape(q.shape[0], ref.full, a.ps).amax(-1)
            slack.setdefault("box", []).append((ubb - m_exact).flatten().numpy())
            slack.setdefault("sph", []).append((ubs - m_exact).flatten().numpy())
            spread.append((m_exact.amax(-1) - m_exact.amin(-1)).numpy())
            base = C.greedy_mass_certificate(ref, V, ubb)
            for name, m in [("exact_fp64", meta)] + list(variants(meta, rng).items()):
                r = C.greedy_mass_certificate(ref, V, C.box_bound(q, m, ref.g, sc))
                d = agg.setdefault(name, {"samples": 0, "heads": 0, "page_bounds": 0, "bound_violations": 0,
                                          "false_certificates": 0, "max_err_over_cert": 0.0,
                                          "heads_stopping_earlier_than_exact": 0, "max_true_err": 0.0})
                d["samples"] += 1; d["heads"] += q.shape[0]; d["page_bounds"] += q.shape[0] * ref.full
                d["bound_violations"] += r["bound_violations"]
                d["false_certificates"] += int(r["viol"].sum())
                d["heads_stopping_earlier_than_exact"] += int((r["k"] < base["k"]).sum())
                d["max_true_err"] = max(d["max_true_err"], float(r["err"].max()))
                ratio = (r["err"] / r["cert"].clamp_min(1e-300))[r["viol"]]
                if ratio.numel():
                    d["max_err_over_cert"] = max(d["max_err_over_cert"], float(ratio.max()))
    sl = {k: np.concatenate(v) for k, v in slack.items()}
    sp = np.concatenate(spread)
    bound_slack = {k: {"median_nats": float(np.median(v)), "p05_nats": float(np.quantile(v, .05)),
                       "p95_nats": float(np.quantile(v, .95)), "min_nats": float(v.min())} for k, v in sl.items()}
    bound_slack["exact_page_max_spread_within_head_median_nats"] = float(np.median(sp))
    res = {"version": "1.1", "date": "2026-11-04", "page_size": a.ps, "real_samples_used": n_samples,
           "bound_slack_upper_minus_exact_page_max": bound_slack,
           "sample_files": [p.name for p in a.samples], "results": agg,
           "note": "Real captured decode tensors; certificate math in float64 log domain; metadata variants as described in module docstring."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
