"""Tests for cycle-2 instrumentation (tests/c2_common.py). Added 2026-11-10.
Run: PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/test_c2_common.py

These test instrumentation correctness only (synthetic modules); no model evidence.
"""
import time

import pytest
import torch

import c2_common as C


@pytest.mark.parametrize("name,cls", [
    ("model.layers.0.self_attn.q_proj", "attn_proj"), ("model.layers.3.self_attn.o_proj", "attn_proj"),
    ("model.layers.0.mlp.gate_proj", "mlp"), ("model.layers.0.mlp.down_proj", "mlp"),
    ("model.layers.0.shared_mlp.input_linear", "mlp"), ("model.layers.9.shared_mlp.output_linear", "mlp"),
    ("lm_head", "lm_head"), ("model.something.proj", "linear_other")])
def test_classify(name, cls):
    assert C.classify(name) == cls


class _Toy(torch.nn.Module):
    def __init__(self, d=64, v=100, tie=True):
        super().__init__()
        self.embed = torch.nn.Embedding(v, d)
        self.q_proj = torch.nn.Linear(d, d, bias=True)
        self.down_proj = torch.nn.Linear(d, d, bias=False)
        self.lm_head = torch.nn.Linear(d, v, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        h = self.q_proj(h)
        time.sleep(0.002)  # non-Linear work, must land in "other"
        h = self.down_proj(h)
        return self.lm_head(h)


def test_weight_bytes_counts_tied_once_and_bias():
    m = _Toy(d=64, v=100)
    b = C.linear_weight_bytes(m)
    assert b["attn_proj"] == (64 * 64 + 64) * 4
    assert b["mlp"] == 64 * 64 * 4
    assert b["lm_head"] == 100 * 64 * 4


def test_weight_bytes_quantized_dynamic():
    m = torch.ao.quantization.quantize_dynamic(_Toy(d=64, v=100, tie=False), {torch.nn.Linear}, dtype=torch.qint8)
    b = C.linear_weight_bytes(m)
    assert b["mlp"] == 64 * 64  # int8 weight, no bias
    assert b["attn_proj"] == 64 * 64 + 64 * 4  # int8 weight + fp32 bias
    assert b["lm_head"] == 100 * 64


@pytest.mark.parametrize("quant", [False, True])
def test_timer_attributes_linear_time_and_leaves_rest(quant):
    m = _Toy(tie=not quant)
    if quant:
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    t = C.LinearTimer(m)
    x = torch.tensor([[1, 2, 3]])
    t0 = time.perf_counter()
    with torch.no_grad():
        m(x)
    total = time.perf_counter() - t0
    s = sum(t.acc.values())
    assert set(t.calls) == {"attn_proj", "mlp", "lm_head"}
    assert all(v == 1 for v in t.calls.values())
    assert 0 < s < total
    assert total - s >= 0.0019  # the sleep is outside every Linear
    t.remove()
    t.reset()
    with torch.no_grad():
        m(x)
    assert sum(t.acc.values()) == 0.0  # hooks really removed


def test_triad_small():
    r = C.triad_gbps(n_bytes=1 << 22, reps=3, threads=1)
    assert r["check_ok"] and r["gbps_median"] > 0


class _Wrapped(torch.nn.Module):
    """HF-like layout: body under `model`, head as `lm_head`."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.q_proj = torch.nn.Linear(16, 16)
        self.model.down_proj = torch.nn.Linear(16, 16)
        self.lm_head = torch.nn.Linear(16, 32, bias=False)


@pytest.mark.parametrize("variant,lm_quant", [("int8dyn", True), ("int8dyn_pc", True), ("int8dyn_pc_nolm", False)])
def test_quantize_variants(variant, lm_quant):
    base = _Wrapped()
    q = C.quantize_variant(base, variant)
    assert C.is_linear(q.model.q_proj) and not isinstance(q.model.q_proj, torch.nn.Linear)
    assert isinstance(q.lm_head, torch.nn.Linear) != lm_quant
    assert isinstance(base.model.q_proj, torch.nn.Linear)  # base untouched
    if variant != "int8dyn":
        assert q.model.q_proj.weight().qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric)
    x = torch.randn(2, 16)
    with torch.no_grad():
        ref = base.lm_head(base.model.down_proj(base.model.q_proj(x)))
        out = q.lm_head(q.model.down_proj(q.model.q_proj(x)))
    assert torch.allclose(ref, out, atol=0.1)
