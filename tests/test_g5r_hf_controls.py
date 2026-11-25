"""Controls in the live analysis path (tests/g5r_hf.py). G5-R, added 2026-11-05.
Run: PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/test_g5r_hf_controls.py
"""
import pytest
import torch

import g5r_hf as H


def _case(seed=7, hq=4, hkv=2, T=70, d=16):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(hq, d, generator=g, dtype=torch.float64)
    K = torch.randn(hkv, T, d, generator=g, dtype=torch.float64)
    V = torch.randn(hkv, T, d, generator=g, dtype=torch.float64)
    return q, K, V, 0.25


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-9, 1e-6])
def test_dense64p_is_exactly_scaled_dense64(eps):
    q, K, V, sc = _case()
    H.CTL.method = "dense64"
    base, _ = H._live_output(0, q, K, V, sc)
    H.CTL.method = f"dense64p:{eps}"
    pert, st = H._live_output(0, q, K, V, sc)
    H.CTL.method = ""
    assert st["pages_frac"] == 1.0
    assert torch.equal(pert, base * (1.0 + eps))
    if eps > 0:
        assert not torch.equal(pert, base)


def test_existing_dense64_unchanged_by_control():
    q, K, V, sc = _case(seed=3)
    H.CTL.method = "dense64"
    a, _ = H._live_output(0, q, K, V, sc)
    H.CTL.method = ""
    ref = H.C.dense_reference(q, K, V, sc, H.CTL.ps)
    assert torch.equal(a, ref.out)
