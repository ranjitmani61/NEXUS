"""Correctness of the TIMED attention paths (tests/g5r_fast_attn.py) against the
float64 analysis core (tests/g5r_core.py). G5-R v1.0, 2026-11-04.
Run: PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/test_g5r_fast_attn.py
"""
import math

import pytest
import torch

import g5r_core as C
import g5r_fast_attn as F


class _Mod(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = 1
        self.is_causal = True


def _case(seed, hq=4, hkv=2, T=330, d=16, structure="random"):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(1, hkv, T, d, generator=g)
    V = torch.randn(1, hkv, T, d, generator=g)
    q = torch.randn(1, hq, 1, d, generator=g)
    if structure == "clustered":
        for kv in range(hkv):
            K[0, kv, 64:96] = q[0, kv * (hq // hkv), 0] * 3.0 + 0.01 * torch.randn(32, d, generator=g)
    return q, K, V, 1 / math.sqrt(d)


def _core(q, K, V, sc, ps):
    q64, K64, V64 = q[0, :, 0].double(), K[0].double(), V[0].double()
    ref = C.dense_reference(q64, K64, V64, sc, ps)
    return ref, C.build_meta(K64, V64, ps), q64, V64


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("structure", ["random", "clustered"])
def test_dense_and_quest_match_core(seed, structure):
    q, K, V, sc = _case(seed, structure=structure)
    F.reset("dense")
    o, _ = F.fast_attention(_Mod(5), q, K, V, None, scaling=sc)
    ref, _, q64, V64 = _core(q, K, V, sc, 16)
    assert torch.allclose(o[0, 0].double(), ref.out, atol=1e-5)
    for B in (64, 128):
        F.reset(f"quest:{B}")
        o, _ = F.fast_attention(_Mod(5), q, K, V, None, scaling=sc)
        qr = C.quest_select(ref, V64, C.build_meta(K[0].double(), V64, 16), q64, sc, B)
        assert torch.allclose(o[0, 0].double(), qr["out"], atol=1e-5)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("structure", ["random", "clustered"])
@pytest.mark.parametrize("meth,kind", [("nx_box", "box"), ("nx_sph", "sph")])
def test_nx_matches_core_selection(seed, structure, meth, kind):
    q, K, V, sc = _case(seed, structure=structure)
    F.reset(meth)
    o, _ = F.fast_attention(_Mod(5), q, K, V, None, scaling=sc)
    ref, meta, q64, V64 = _core(q, K, V, sc, 32)
    ub = C.box_bound(q64, meta, ref.g, sc) if kind == "box" else C.sphere_bound(q64, meta, ref.g, sc)
    r = C.greedy_mass_certificate(ref, V64, ub)
    # fast path either reproduces the certified selection or (if any head needs all pages) runs dense
    if (r["k"] == ref.full).any():
        assert torch.allclose(o[0, 0].double(), ref.out, atol=1e-5)
    else:
        assert torch.allclose(o[0, 0].double(), r["out"], atol=1e-5)
    assert (o[0, 0].double() - ref.out).norm(dim=-1).max() / ref.vmax.min() <= C.TOL + 1e-6


def test_clustered_case_actually_screens():
    # All query heads aligned with the dominant page: the timed path falls back to dense for the
    # whole layer if ANY head fails (granularity documented in g5r_fast_attn), so a test of
    # screening must make every head screenable.
    q, K, V, sc = _case(1, structure="clustered")
    q[0, :, 0] = q[0, 0, 0]
    for kv in range(K.shape[1]):
        K[0, kv, 64:96] = q[0, 0, 0] * 9.0
    F.reset("nx_box")
    F.fast_attention(_Mod(5), q, K, V, None, scaling=sc)
    assert F.ST.pages_read < F.ST.pages_total, "screening should skip pages on a dominant-page case"
