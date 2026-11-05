"""Regression for float64 mass-certificate roundoff in real-oracle diagnostics."""
from __future__ import annotations
import numpy as np
from g5_trace_oracle import evaluate_one


def test_page_mass_roundoff_never_creates_negative_certificate():
    rng = np.random.default_rng(20261105)
    observed_roundoff = False
    for trial in range(512):
        w = rng.dirichlet(np.ones(65))
        w = w / w.sum()  # evaluate_one also normalizes before computing page masses
        page_mass = np.array([w[i:i+8].sum() for i in range(0,65,8)])
        total_from_pages = float(np.cumsum(page_mass)[-1])
        if total_from_pages > 1.0:
            observed_roundoff = True
        r = evaluate_one(w, np.zeros((65,4)), page_size=8, tolerance=1e-12)
        assert r['oracle_mass_certificate'] >= 0.0, f'trial={trial} total={total_from_pages}'
        assert r['oracle_mass_certificate'] <= 1e-12
    assert observed_roundoff, 'negative uncorrected certificate condition was not exercised'


def test_selected_all_pages_has_zero_mass_certificate():
    w = np.array([0.13, 0.11, 0.23, 0.19, 0.17, 0.17],dtype=np.float64)
    v = np.array([[0.1,0.2]]*6)
    out = evaluate_one(w, v, page_size=1,tolerance=1e-14)
    assert out['oracle_mass_pages'] == 6
    assert out['oracle_mass_certificate'] >= 0.0
    assert out['oracle_mass_certificate'] <= 1e-14
