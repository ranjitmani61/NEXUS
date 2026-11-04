"""Unit/property tests for tests/g5r_core.py (G5-R v1.0, 2026-11-04).

These tests check the mathematics on synthetic tensors. They do NOT establish
anything about real models, hardware or performance.
Run: PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/test_g5r_core.py
"""
import math

import numpy as np
import pytest
import torch

import g5r_core as C

torch.set_default_dtype(torch.float64)


def _case(seed, hq=4, hkv=2, T=200, d=16, dv=8, ps=16, structure="random"):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(hkv, T, d, generator=g)
    V = torch.randn(hkv, T, dv, generator=g)
    q = torch.randn(hq, d, generator=g)
    if structure == "clustered":
        K[:, :ps] = q[::hq // hkv][:, None] * 2.0 + 0.01 * torch.randn(hkv, ps, d, generator=g)
    if structure == "uniform":
        K = torch.zeros_like(K)
    return q, K, V, 1 / math.sqrt(d), ps


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("structure", ["random", "clustered", "uniform"])
def test_certificates_hold_and_dominance(seed, structure):
    q, K, V, sc, ps = _case(seed, structure=structure, T=150 + seed)
    ref = C.dense_reference(q, K, V, sc, ps)
    meta = C.build_meta(K, V, ps)
    g = ref.g
    box = C.greedy_mass_certificate(ref, V, C.box_bound(q, meta, g, sc))
    sph = C.greedy_mass_certificate(ref, V, C.sphere_bound(q, meta, g, sc))
    val = C.value_certificate(ref, V, meta, C.box_bound(q, meta, g, sc))
    orm, ors = C.oracle_mass(ref), C.oracle_score(ref)
    for r in (box, sph, val):
        assert not r["viol"].any(), "certificate violated"
        assert (r["err"] <= C.TOL + 1e-12).all()
    assert box["bound_violations"] == 0 and sph["bound_violations"] == 0
    # Dominance (contract §7): any valid-bound mass-certificate method >= OR-SCORE >= OR-MASS
    assert (box["k"] >= ors).all() and (sph["k"] >= ors).all()
    assert (ors >= orm).all()


def test_oracle_score_is_exact_minimum_bruteforce():
    """Brute-force all subsets on tiny cases to confirm OR-SCORE is the true minimum."""
    import itertools
    for seed in range(60):
        q, K, V, sc, ps = _case(seed, hq=2, hkv=1, T=4 * 7 + 3, d=6, dv=3, ps=4)
        ref = C.dense_reference(q, K, V, sc, ps)
        ors = C.oracle_score(ref)
        s = ref.scores.amax(-1)
        a, a_part, m = C._page_mass_stats(ref, s)
        b = ps * torch.exp(m - s[:, None])
        c = C.TOL / (2 - C.TOL)
        for h in range(2):
            best = None
            for k in range(ref.full + 1):
                for S in itertools.combinations(range(ref.full), k):
                    inS = torch.zeros(ref.full, dtype=torch.bool); inS[list(S)] = True
                    if b[h][~inS].sum() <= c * (a_part[h] + a[h][inS].sum()) * (1 + 1e-12):
                        best = k; break
                if best is not None:
                    break
            assert best == int(ors[h])


def test_full_visit_is_exact():
    q, K, V, sc, ps = _case(3, structure="uniform")
    ref = C.dense_reference(q, K, V, sc, ps)
    meta = C.build_meta(K, V, ps)
    r = C.greedy_mass_certificate(ref, V, C.box_bound(q, meta, ref.g, sc))
    # uniform keys -> flat attention -> needs (nearly) all pages; output error then 0
    assert (r["k"] >= ref.full - 1).all()


def test_clustered_needs_few_pages():
    q, K, V, sc, ps = _case(5, structure="clustered", d=16)
    K[:, :ps] *= 4.0
    ref = C.dense_reference(q, K, V, sc, ps)
    meta = C.build_meta(K, V, ps)
    r = C.greedy_mass_certificate(ref, V, C.box_bound(q, meta, ref.g, sc))
    assert r["k"].min() <= 2


def test_stale_metadata_can_produce_false_certificate():
    """Adversarial: metadata built before keys were overwritten -> invalid bound."""
    # v1.0 of this test used random keys: the certificate never stopped early, so all pages were
    # visited and no false certificate could occur (test-construction error, corrected 2026-11-04).
    # Correct witness: page 0 legitimately dominant at metadata-build time -> screening stops early;
    # page 3 keys are then overwritten (missed metadata update) to dominate even more.
    q, K, V, sc, ps = _case(7, T=128, structure="clustered")
    K[:, :ps] *= 4.0
    meta = C.build_meta(K, V, ps)
    K2 = K.clone()
    K2[:, ps * 3: ps * 4] = q[::2][:, None] * 12.0   # page 3 becomes dominant AFTER metadata build
    ref = C.dense_reference(q, K2, V, sc, ps)
    r = C.greedy_mass_certificate(ref, V, C.box_bound(q, meta, ref.g, sc))
    assert r["bound_violations"] > 0
    assert r["viol"].any(), "expected the stale-metadata certificate to be false"


def test_quest_budget_monotone_error_zero_at_full():
    q, K, V, sc, ps = _case(11, T=256)
    ref = C.dense_reference(q, K, V, sc, 16)
    m16 = C.build_meta(K, V, 16)
    r_all = C.quest_select(ref, V, m16, q, sc, budget=256)
    assert torch.allclose(r_all["out"], ref.out, atol=1e-12)
    r_small = C.quest_select(ref, V, m16, q, sc, budget=32)
    assert (r_small["err"] >= 0).all()


def test_value_certificate_against_archived_oracle_formula():
    """max_box_fraction bisection here == archived g5_trace_oracle.max_box_fraction."""
    import importlib.util, pathlib
    p = pathlib.Path(__file__).parent / "_archived_g5_trace_oracle.py"
    if not p.exists():
        pytest.skip("archived oracle copy not present")
    spec = importlib.util.spec_from_file_location("arch", p); arch = importlib.util.module_from_spec(spec); spec.loader.exec_module(arch)
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.random(30); d = rng.random(30) * 3; z = rng.random() + 0.1
        ref_val = arch.max_box_fraction(u, d, z)
        lo, hi = 0.0, d.max()
        for _ in range(80):
            mid = (lo + hi) / 2
            if (u * np.maximum(d - mid, 0)).sum() > mid * z: lo = mid
            else: hi = mid
        assert abs(hi - ref_val) < 1e-9


def test_log_domain_certificate_survives_extreme_loose_bound():
    """Regression for the real-model underflow failure (Qwen2.5 layer 0, 2026-11-04):
    one page with an enormous (but valid) upper bound must not drive U to 0."""
    q, K, V, sc, ps = _case(13, T=96, d=8)
    ref = C.dense_reference(q, K, V, sc, ps)
    meta = C.build_meta(K, V, ps)
    ub = C.box_bound(q, meta, ref.g, sc).clone()
    ub[:, 0] += 900.0          # valid (only looser) bound, > float64 exp range above the rest
    r = C.greedy_mass_certificate(ref, V, ub)
    assert not r["viol"].any()
    assert (r["cert"][r["k"] < ref.full] > 0).all(), "certificate must not collapse to 0 while pages remain"
    v = C.value_certificate(ref, V, meta, ub)
    assert not v["viol"].any()
