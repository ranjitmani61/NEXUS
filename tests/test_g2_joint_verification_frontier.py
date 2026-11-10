"""Falsification tests for the G2 conditional speculative-verification model."""
import math

import numpy as np
import pytest

from g2_joint_verification_frontier import (
    ANCHORS, expected_output_tokens, target_cost_units, modeled_speedup,
    max_beta_for_two_x, min_acceptance_rate_for_two_x, simulated_output_tokens, serial_attention,
    fused_attention,
)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_all_accepted_gives_k_plus_one(k):
    assert expected_output_tokens(k, 1) == k + 1


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_none_accepted_gives_one(k):
    assert expected_output_tokens(k, 0) == 1


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_acceptance_length_monotone(k):
    vals = [expected_output_tokens(k, p) for p in (0, .5, .7, .9, .99, 1)]
    assert vals == sorted(vals)


@pytest.mark.parametrize("p", [0.5, 0.7, 0.9, 0.99999999])
def test_stable_series_close_to_direct_sum(p):
    for k in (1, 4, 32):
        assert expected_output_tokens(k, p) == pytest.approx(sum(p**j for j in range(k+1)), rel=1e-10)


@pytest.mark.parametrize("k,p", [(2,.5),(4,.7),(4,.9),(8,.95)])
def test_monte_carlo_acceptance_matches_math(k, p):
    observed=simulated_output_tokens(k,p,20261110,60000)
    assert abs(observed-expected_output_tokens(k,p)) < 0.035


def test_stochastic_repeatable_and_seed_sensitive():
    a=simulated_output_tokens(8,.75,123,4000)
    assert a==simulated_output_tokens(8,.75,123,4000)
    assert a!=simulated_output_tokens(8,.75,124,4000)


@pytest.mark.parametrize("key", list(ANCHORS))
def test_anchor_attn_only_no_reuse_has_no_2x_for_perfect_draft(key):
    f=ANCHORS[key]['f']
    for k in (1,2,4,8,100):
        # With k+1 real target queries, even p=1, delta=h=0, no 2x when f>1/2.
        assert modeled_speedup(f,k,1,1,0) < 2


@pytest.mark.parametrize("key",list(ANCHORS))
def test_family_ceiling_asymptote_one_over_f(key):
    f=ANCHORS[key]['f']
    speed=modeled_speedup(f,1000000,1,1,0)
    assert speed==pytest.approx(1/f,rel=1e-5)


@pytest.mark.parametrize("key", list(ANCHORS))
def test_beta_threshold_inverts_speedup(key):
    f=ANCHORS[key]['f']
    for k,p,delta,h in [(4,.9,.05,0),(8,.95,.05,.05),(1,1,0,0)]:
        beta=max_beta_for_two_x(f,k,p,delta,h)
        assert modeled_speedup(f,k,p,beta,delta,h)==pytest.approx(2)
        assert modeled_speedup(f,k,p,beta+.05,delta,h)<2


def test_no_reuse_beta1_and_one_bonus_cost_accounted():
    f=.7
    assert target_cost_units(f,4,1,0)==pytest.approx(1+4*f)
    # Incorrect free-bonus model 1+3*f would artificially inflate speed.
    assert modeled_speedup(f,4,1,1,0)==pytest.approx(5/(1+4*f))


def test_some_operating_points_impossible_even_perfect_KV_reuse():
    f=.7052066730549075
    assert max_beta_for_two_x(f,4,.6,.05,0)<0
    assert modeled_speedup(f,4,.6,0,.05)<2


@pytest.mark.parametrize("f",[0,.2,.5861178531211508,.7052066730549075,1])
def test_model_monotonic_with_beta_and_overheads(f):
    assert modeled_speedup(f,4,.9,.1,.05)<modeled_speedup(f,4,.9,.1,0)
    if f>0: assert modeled_speedup(f,4,.9,.8,.05)<modeled_speedup(f,4,.9,.2,.05)


@pytest.mark.parametrize("k",[1,2,4,8])
def test_exact_historical_prefix_attention_single_head(k):
    rng=np.random.default_rng(20261110+k)
    keys=rng.standard_normal((211,32),dtype=np.float32)
    values=rng.standard_normal((211,32),dtype=np.float32)
    q=rng.standard_normal((k,32),dtype=np.float32)
    ref=serial_attention(q,keys,values)
    fused=fused_attention(q,keys,values)
    np.testing.assert_allclose(fused,ref,rtol=3e-4,atol=3e-4)


@pytest.mark.parametrize("num_tokens",[1,17,49,128])
def test_attention_output_finite_and_shape(num_tokens):
    rng=np.random.default_rng(num_tokens)
    keys=rng.standard_normal((num_tokens,64),dtype=np.float32)
    values=rng.standard_normal((num_tokens,64),dtype=np.float32)
    q=rng.standard_normal((5,64),dtype=np.float32)
    out=fused_attention(q,keys,values)
    assert out.shape==(5,64)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("wrong",[
    (-.1,4,.9,0,.05,0),(1.1,4,.9,0,.05,0),
    (.7,4,-.1,0,.05,0),(.7,4,1.01,0,.05,0),
    (.7,0,.9,0,.05,0),(.7,4,.9,-.01,.05,0),
    (.7,4,.9,0,-.01,0),(.7,4,.9,0,.05,-.01),
])
def test_malformed_parameters_rejected(wrong):
    with pytest.raises(ValueError):modeled_speedup(*wrong)


def test_k_bool_and_f_nan_rejected():
    with pytest.raises(ValueError):expected_output_tokens(True,.9)
    with pytest.raises(ValueError):modeled_speedup(float('nan'),4,.9,0,.05)


def test_beta_no_coalescing_works_even_if_perfect_draft_has_no_gain():
    assert modeled_speedup(.705,8,1,1,.1)<1/.705


def test_f_zero_is_degenerate_and_beta_threshold_not_applicable():
    assert max_beta_for_two_x(0,4,.9,.05) is None
    assert modeled_speedup(0,4,.9,0,.05)==modeled_speedup(0,4,.9,1,.05)


@pytest.mark.parametrize("key", list(ANCHORS))
@pytest.mark.parametrize("beta", [0.0, .1, .25, .5, .75, 1.0])
def test_inverse_acceptance_threshold(key, beta):
    f=ANCHORS[key]["f"]
    p=min_acceptance_rate_for_two_x(f,4,beta,.05)
    if p is None:
        assert modeled_speedup(f,4,1.0,beta,.05)<2
    else:
        assert 0<=p<=1
        assert modeled_speedup(f,4,p,beta,.05)==pytest.approx(2.0,rel=1e-9)
        if p>.001:
            assert modeled_speedup(f,4,p-.0001,beta,.05)<2


def test_posthoc_k_stationary_matches_reference():
    from g2_joint_kstation_posthoc import kv_stationary
    rng=np.random.default_rng(123)
    q=rng.normal(size=(4,16)).astype(np.float32)
    k=rng.normal(size=(87,16)).astype(np.float32)
    v=rng.normal(size=(87,16)).astype(np.float32)
    np.testing.assert_allclose(serial_attention(q,k,v),kv_stationary(q,k,v),rtol=3e-4,atol=3e-4)
