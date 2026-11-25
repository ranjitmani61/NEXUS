"""Tests for C3 removal-based attribution (tests/c3_gpu_breakdown.py). Added 2026-11-10.
Synthetic modules only: they test the instrumentation logic, not any model or hardware claim.
Run: PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/test_c3_gpu_breakdown.py
"""
import numpy as np
import pytest
import torch

import c3_gpu_breakdown as G


@pytest.fixture(autouse=True)
def _float32_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.gate_proj = torch.nn.Linear(8, 16)
        self.down_proj = torch.nn.Linear(16, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, 32, bias=False)

    def forward(self, x):
        h = self.norm(x + self.q_proj(x))
        h = h + self.down_proj(torch.relu(self.gate_proj(h)))
        return self.lm_head(h)


@pytest.mark.parametrize("a,s", [("full", set()), ("no_mlp", {"mlp"}), ("no_attn_core", {"attn_core"}),
                                 ("no_all", {"attn_core", "attn_proj", "mlp", "lm_head"})])
def test_removed_set(a, s):
    assert G.removed_set(a) == s


def test_ablator_stubs_and_exact_restore():
    torch.manual_seed(0)
    m = _Toy().eval()
    x = torch.randn(1, 1, 8)
    ab = G.Ablator(m)
    assert {k: len(v) for k, v in ab.mods.items()} == {"attn_proj": 1, "mlp": 2, "lm_head": 1}
    with torch.no_grad():
        ref = m(x).clone()
        ab.set({"lm_head"})
        assert torch.equal(m(x), torch.zeros(1, 1, 32))
        ab.set({"mlp"})  # set() restores first, so only MLP is removed now
        assert torch.equal(m.gate_proj(x[..., :8]), torch.zeros(1, 1, 16))
        assert not torch.equal(m(x), torch.zeros(1, 1, 32))
        ab.set({"attn_core", "attn_proj"})
        assert G.STATE["remove_attn"] is True
        ab.restore()
        assert G.STATE["remove_attn"] is False
        assert all("forward" not in mod.__dict__ for v in ab.mods.values() for mod in v)
        assert torch.equal(m(x), ref)


def test_weight_bytes_classes_and_bias():
    b = G.weight_bytes(_Toy())
    assert b["attn_proj"] == (8 * 8 + 8) * 4
    assert b["mlp"] == (8 * 16 + 16 + 16 * 8 + 8) * 4
    assert b["lm_head"] == 8 * 32 * 4


def test_quality_identical_and_shifted():
    rng = np.random.default_rng(0)
    ref = []
    for _ in range(3):
        lp = torch.log_softmax(torch.randn(5, 11), -1)
        ref.append((lp, rng.random(5)))
    q = G.quality_vs_ref(ref, ref)
    assert q["ppl_ratio"] == 1.0 and q["top1_agree"] == 1.0 and q["matched_quality"]
    worse = [(lp, nll + 0.02) for lp, nll in ref]  # +0.02 nats/token -> PPL ratio e^0.02 > 1.01
    q2 = G.quality_vs_ref(ref, worse)
    assert abs(q2["ppl_ratio"] - np.exp(0.02)) < 1e-9 and not q2["matched_quality"]


def test_attention_stub_shape():
    G.register_attention()
    from transformers import AttentionInterface
    f = AttentionInterface()["c3_sdpa"]
    q = torch.randn(1, 4, 1, 16); k = torch.randn(1, 2, 7, 16); v = torch.randn(1, 2, 7, 16)

    class _M(torch.nn.Module):
        num_key_value_groups = 2
        is_causal = True
        training = False
    G.STATE["remove_attn"] = True
    try:
        o, _ = f(_M(), q, k, v, None, scaling=0.25, dropout=0.0)
    finally:
        G.STATE["remove_attn"] = False
    assert o.shape == (1, 1, 4, 16) and torch.count_nonzero(o) == 0
    o2, _ = f(_M(), q, k, v, None, scaling=0.25, dropout=0.0)
    assert o2.shape == (1, 1, 4, 16) and torch.count_nonzero(o2) > 0


def test_mask_registered_for_custom_attention():
    """Regression for defect C3-D1: without a registered mask, StaticCache decode is wrong."""
    G.register_attention()
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    assert "c3_sdpa" in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping


def test_granite_capture_patch_guard():
    """C3-D2: the mamba-mask patch is applied only when the model has no Mamba layers."""
    from types import SimpleNamespace
    inner = SimpleNamespace(_update_mamba_mask=lambda am, cp: "orig")
    ok = SimpleNamespace(config=SimpleNamespace(model_type="granitemoehybrid", layer_types=["attention"] * 3), model=inner)
    assert G.patch_granite_capture(ok) is True
    assert ok.model._update_mamba_mask(None, None) is None
    bad = SimpleNamespace(config=SimpleNamespace(model_type="granitemoehybrid", layer_types=["attention", "mamba"]),
                          model=SimpleNamespace(_update_mamba_mask=lambda am, cp: "orig"))
    with pytest.raises(AssertionError):
        G.patch_granite_capture(bad)
    other = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"), model=SimpleNamespace())
    assert G.patch_granite_capture(other) is False


def test_capture_is_no_grad():
    """C3-D4: graph capture must run without autograd (as eager timing does)."""
    import inspect
    src = inspect.getsource(G.Decoder.capture)
    assert "@torch.no_grad()" in src and "assert not torch.is_grad_enabled()" in src


def test_kv_layer_count_full_attention():
    """C3-D5: Qwen2 configs label layers 'full_attention'; they must count as KV layers."""
    import c3_analyze as A
    q = {"num_hidden_layers": 24, "layer_types": ["full_attention"] * 24, "num_key_value_heads": 2, "hidden_size": 896, "num_attention_heads": 14}
    g = {"num_hidden_layers": 40, "layer_types": ["attention"] * 40, "num_key_value_heads": 8, "hidden_size": 2560, "num_attention_heads": 40}
    h = {"num_hidden_layers": 4, "layer_types": ["mamba", "attention", "mamba", "attention"], "num_key_value_heads": 1, "hidden_size": 64, "num_attention_heads": 1}
    assert A.kv_bytes_per_slot(q) == 2 * 24 * 2 * 64 * 2 == 12288
    assert A.kv_bytes_per_slot(g) == 2 * 40 * 8 * 64 * 2 == 81920
    assert A.kv_bytes_per_slot(h) == 2 * 2 * 1 * 64 * 2
