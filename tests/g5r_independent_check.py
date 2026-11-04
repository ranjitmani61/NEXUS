#!/usr/bin/env python3
"""G5-R independent verification of recorded page counts / certificates.

Version 1.1, 2026-11-04 (v1.1: mpmath certificate arithmetic, see greedy()). Separately written from tests/g5r_core.py: NumPy only,
explicit per-page loops, no shared helper code. Inputs are real (q, K, V) tensors
captured at random (layer, step) points during shadow runs, plus the values the
torch harness recorded for them. Also runs the ARCHIVED project oracle
(tests/_archived_g5_trace_oracle.py, byte-identical copy of the 2026-11-03
harness file, sha256 2d7d0c9d...) on the same real data.

Usage: python tests/g5r_independent_check.py SAMPLES.npz [...] --output results/g5r/independent_check.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 60
TOL = 0.02


def _load_archived():
    p = Path(__file__).parent / "_archived_g5_trace_oracle.py"
    spec = importlib.util.spec_from_file_location("archived_oracle", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_head(q, K, V, scale, ps):
    """Return dict of independently computed counts for ONE query head."""
    T = K.shape[0]
    scores = (K @ q) * scale
    full = T // ps
    # page metadata by explicit loop
    box_ub, sph_ub, pmax = [], [], []
    for j in range(full):
        Kj = K[j * ps:(j + 1) * ps]
        lo, hi = Kj.min(0), Kj.max(0)
        box_ub.append(float(np.sum(np.maximum(q * lo, q * hi)) * scale))
        c = Kj.mean(0)
        r = float(np.max(np.linalg.norm(Kj - c, axis=1)))
        sph_ub.append(float((q @ c + np.linalg.norm(q) * r) * scale))
        pmax.append(float(scores[j * ps:(j + 1) * ps].max()))
    part = scores[full * ps:]
    res = {"full": full}

    def greedy(ub):
        """v1.1: certificate decisions in 60-digit mpmath (unbounded exponent range).
        v1.0 used float64 with a running total `u = max(0, u - bet[j])`; on a real Qwen2.5
        head (bounds spanning 60 nats) that subtraction cancelled catastrophically, u hit 0
        and the checker stopped at 115/128 pages with a false certificate. The tail mass is
        now summed directly from the unvisited pages (no subtraction)."""
        ub = np.array(ub)
        order = [int(j) for j in np.argsort(-ub, kind="stable")]
        E = lambda x: mp.e ** mp.mpf(float(x))
        z = mp.fsum(E(x) for x in part) if part.size else mp.mpf(0)
        bet = [ps * E(b) for b in ub]
        page_mass = [mp.fsum(E(x) for x in scores[j * ps:(j + 1) * ps]) for j in range(full)]
        tail = [mp.mpf(0)] * (full + 1)
        for k in range(full - 1, -1, -1):
            tail[k] = tail[k + 1] + bet[order[k]]      # suffix sum, smallest terms first
        visited = []
        for k in range(full + 1):
            u = tail[k]
            cert = mp.mpf(0) if k == full else 2 * u / (z + u)
            if cert <= TOL:
                return k, float(cert), visited
            j = order[k]
            z += page_mass[j]
            visited.append(j)

    kb, cb, vis = greedy(box_ub)
    ks, cs, _ = greedy(sph_ub)
    res.update(box_k=kb, box_cert=cb, sph_k=ks, sph_cert=cs)
    # true error of box selection
    p = np.exp(scores - scores.max()); p /= p.sum()
    dense = p @ V
    mask = np.zeros(T, bool); mask[full * ps:] = True
    for j in vis:
        mask[j * ps:(j + 1) * ps] = True
    sel = (p * mask) @ V / (p * mask).sum()
    vmax = float(np.linalg.norm(V, axis=1).max())
    res["box_err"] = float(np.linalg.norm(sel - dense) / vmax)
    res["box_bound_valid"] = bool(all(b >= m - 1e-9 * (1 + abs(b)) for b, m in zip(box_ub, pmax)))
    res["sph_bound_valid"] = bool(all(b >= m - 1e-9 * (1 + abs(b)) for b, m in zip(sph_ub, pmax)))
    # OR-MASS (partial forced) by loop
    s = scores.max()
    a = np.array([np.exp(scores[j * ps:(j + 1) * ps] - s).sum() for j in range(full)])
    ap = float(np.exp(part - s).sum()) if part.size else 0.0
    A = a.sum() + ap
    acc, k = ap, 0
    for x in sorted(a, reverse=True):
        if acc >= (1 - TOL / 2) * A * (1 - 1e-12):
            break
        acc += x; k += 1
    res["ormass_k"] = k
    # OR-SCORE by loop
    b = np.array([ps * math.exp(m - s) for m in pmax])
    c = TOL / (2 - TOL)
    need = b.sum() - c * ap
    acc, k = 0.0, 0
    for x in sorted(b + c * a, reverse=True):
        if acc >= need * (1 - 1e-12):
            break
        acc += x; k += 1
    res["orscore_k"] = k
    # unforced OR-MASS for comparison with archived oracle (partial page treated as an ordinary page)
    pages = list(a) + ([ap] if part.size else [])
    cum, k2 = 0.0, 0
    for x in sorted(pages, reverse=True):
        cum += x; k2 += 1
        if cum >= (1 - TOL / 2) * A:
            break
    res["ormass_unforced_pages"] = k2
    res["weights"] = p
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+", type=Path)
    ap.add_argument("--ps", type=int, default=32)
    ap.add_argument("--archived-max-tokens", type=int, default=9000)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    arch = _load_archived()
    rows, mism = [], []
    for path in a.samples:
        z = np.load(path, allow_pickle=True)
        for i in range(int(z["n"])):
            q, K, V = z[f"q{i}"], z[f"K{i}"], z[f"V{i}"]
            scale = float(z[f"scale{i}"]); layer = int(z[f"layer{i}"]); step = int(z[f"step{i}"])
            hq, hkv = q.shape[0], K.shape[0]
            G = hq // hkv
            for h in range(hq):
                r = check_head(q[h], K[h // G], V[h // G], scale, a.ps)
                rec = {"file": path.name, "layer": layer, "step": step, "head": h, "T": int(K.shape[1])}
                for key in ("box_k", "sph_k", "ormass_k", "orscore_k"):
                    harness = int(z[f"{key}{i}"][h])
                    rec[key] = r[key]; rec[key + "_harness"] = harness
                    if harness != r[key]:
                        mism.append({**rec, "field": key})
                harness_err = float(z[f"box_err{i}"][h])
                rec["box_err"] = r["box_err"]; rec["box_err_harness"] = harness_err
                if abs(harness_err - r["box_err"]) > 1e-9 + 1e-6 * abs(r["box_err"]):
                    mism.append({**rec, "field": "box_err"})
                rec["box_bound_valid"] = r["box_bound_valid"]; rec["sph_bound_valid"] = r["sph_bound_valid"]
                rec["cert_holds_independent"] = bool(r["box_err"] <= r["box_cert"] * (1 + 1e-6) + 1e-12)
                if K.shape[1] <= a.archived_max_tokens:
                    ev = arch.evaluate_one(r["weights"], V[h // G], a.ps, TOL)
                    rec["archived_oracle_mass_pages"] = ev["oracle_mass_pages"]
                    rec["archived_value_oracle_pages"] = ev["oracle_exact_score_value_pages"]
                    rec["archived_greedy_hindsight_pages"] = ev["greedy_hindsight_pages"]
                    if ev["oracle_mass_pages"] != r["ormass_unforced_pages"]:
                        mism.append({**rec, "field": "archived_oracle_mass_vs_independent_unforced",
                                     "independent": r["ormass_unforced_pages"]})
                rows.append(rec)
    out = {"version": "1.1", "date": "2026-11-04", "heads_checked": len(rows),
           "mismatches": mism, "n_mismatches": len(mism),
           "all_box_bounds_valid": all(r["box_bound_valid"] for r in rows),
           "all_sph_bounds_valid": all(r["sph_bound_valid"] for r in rows),
           "all_certificates_hold_independent": all(r["cert_holds_independent"] for r in rows),
           "archived_oracle_heads": sum("archived_oracle_mass_pages" in r for r in rows),
           "rows": rows}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "mismatches")}, indent=1))
    if mism:
        print("FIRST MISMATCHES:", json.dumps(mism[:5], default=float))


if __name__ == "__main__":
    main()
