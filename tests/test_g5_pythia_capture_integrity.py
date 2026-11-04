"""Fail-closed tests for Pythia capture instrumentation; NO pretrained weights used.

These are synthetic integration fixtures. They do not clear G5.
"""
from types import SimpleNamespace
import math

import pytest
import torch

from g5_pythia_capture import (
    REVISION, capture_for_text, context_reconstruction_error,
    verify_checkpoint_revision,
)


def make_attention_fixture(t=65, h=2, d=4, seed=71):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn((1, t, h, d), generator=g, dtype=torch.float64)
    logits = torch.randn((1, h, t, t), generator=g, dtype=torch.float64)
    future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    a = logits.masked_fill(future[None, None], -1e12).softmax(-1)
    expected = torch.einsum('bhqk,bkhd->bqhd', a, v).reshape(1, t, h*d)
    return a, v, expected


def test_unchecked_query_false_acceptance_in_legacy_three_position_check():
    a, v, expected = make_attention_fixture()
    t = a.shape[-1]
    tampered = a.clone()
    q = t - 2  # Measured query, not in original [first, middle, last] check.
    tampered[:, :, q, :] = 0
    tampered[:, :, q, 0] = 1
    assert context_reconstruction_error(tampered, v, expected, [0, t//2, t-1]) < 1e-12
    assert context_reconstruction_error(tampered, v, expected, [q, t-1]) > 0.1


@pytest.mark.parametrize('mutation', ['future', 'negative', 'nan', 'unnormalized'])
def test_invalid_attention_rows_fail_closed(mutation):
    a, v, expected = make_attention_fixture()
    q = 50
    bad = a.clone()
    if mutation == 'future':
        bad[0, 0, q, q+1] = 0.25
    elif mutation == 'negative':
        bad[0, 0, q, 0] = -0.05
    elif mutation == 'nan':
        bad[0, 0, q, 0] = math.nan
    else:
        bad[0, 0, q, :q+1] *= 0.3
    with pytest.raises(ValueError, match='Invalid causal'):
        context_reconstruction_error(bad, v, expected, [q])


def test_pinned_revision_is_mandatory():
    verify_checkpoint_revision(REVISION, REVISION)
    with pytest.raises(RuntimeError, match='missing or differs'):
        verify_checkpoint_revision(REVISION, None)
    with pytest.raises(RuntimeError, match='missing or differs'):
        verify_checkpoint_revision(REVISION, '0' * 40)
    with pytest.raises(ValueError, match='audited'):
        verify_checkpoint_revision('0' * 40, '0' * 40)


class FakeAttention(torch.nn.Module):
    def __init__(self, corrupted=False):
        super().__init__()
        self.query_key_value = torch.nn.Linear(8, 24, bias=False)
        self.dense = torch.nn.Linear(8, 8, bias=False)
        self.corrupted = corrupted

    def forward(self, x):
        b, t, _ = x.shape
        qkv = self.query_key_value(x).reshape(b, t, 2, 3, 4)
        q = qkv[:, :, :, 0, :]
        k = qkv[:, :, :, 1, :]
        v = qkv[:, :, :, 2, :]
        logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / 2
        future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        a = logits.masked_fill(future[None, None], -1e10).softmax(-1)
        ctx = torch.einsum('bhqk,bkhd->bqhd', a, v).reshape(b, t, 8)
        self.dense(ctx)  # Pre-hook records actual, unmodified attention context.
        if self.corrupted:
            a = a.clone()
            a[:, :, t-2, :] = 0
            a[:, :, t-2, 0] = 1
        return a


class FakeModel(torch.nn.Module):
    def __init__(self, corrupted=False):
        super().__init__()
        torch.manual_seed(1)
        self.embed = torch.nn.Embedding(128, 8)
        layer = torch.nn.Module()
        layer.attention = FakeAttention(corrupted=corrupted)
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList([layer])
        self.config = SimpleNamespace(
            model_type='gpt_neox', num_attention_heads=2,
            max_position_embeddings=128,
        )

    def forward(self, ids, output_attentions, use_cache, return_dict):
        assert output_attentions and not use_cache and return_dict
        x = self.embed(ids)
        a = self.gpt_neox.layers[0].attention(x)
        return SimpleNamespace(attentions=(a,))


class FakeTokenizer:
    def __call__(self, text, return_tensors, add_special_tokens):
        assert return_tensors == 'pt' and not add_special_tokens
        return {'input_ids': torch.arange(len(text)).reshape(1, -1)}


def test_full_capture_catches_corruption_outside_legacy_three_queries():
    model = FakeModel(corrupted=True)
    with pytest.raises(AssertionError, match='reconstruction failed'):
        capture_for_text(model, FakeTokenizer(), 'x'*65, max_tokens=65, last_queries=5)


def test_full_capture_accepts_consistent_fixture_but_not_g5_evidence():
    model = FakeModel(corrupted=False)
    rows, checks, t = capture_for_text(
        model, FakeTokenizer(), 'x'*65, max_tokens=65, last_queries=5,
    )
    assert t == 65
    assert len(rows) == 2 * 5
    assert checks[0]['positions_checked'] == 7  # first, middle + last five
    assert checks[0]['max_preprojection_absolute_error'] < 1e-6
