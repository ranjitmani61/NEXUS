"""Reproducible offline unit/property tests. Does not test pretrained-model behavior."""
import itertools
import numpy as np
import pytest
from g5_trace_oracle import evaluate_one, max_box_fraction, synthetic_fixtures


def test_box_fraction_bruteforce_random():
    rng=np.random.default_rng(1241)
    for case in range(250):
        n=int(rng.integers(1,8))
        U=rng.uniform(0,4,n)
        d=rng.uniform(0,5,n)
        z=float(rng.uniform(.1,10))
        exact=max(float((x@d)/(z+x.sum())) for bits in itertools.product([0,1],repeat=n)
                  for x in [U*np.array(bits)])
        approx=max_box_fraction(U,d,z)
        assert abs(exact-approx) < 1e-11


def test_synthetic_fixtures():
    for name,w,v in synthetic_fixtures():
        r=evaluate_one(w,v,32,.02)
        assert r['oracle_exact_score_value_true_error'] <= r['oracle_exact_score_value_certificate'] + 2e-10
        assert r['oracle_exact_score_value_true_error'] <= .020000001
        assert r['oracle_mass_certificate'] <= .020000001
        assert 1 <= r['oracle_exact_score_value_pages'] <= r['pages']
        assert 1 <= r['oracle_mass_pages'] <= r['pages']


def test_random_adversarial_lengths_and_zeros():
    rng=np.random.default_rng(221)
    for case in range(120):
        n=int(rng.integers(1,150))
        d=int(rng.integers(1,13))
        x=rng.exponential(size=n)
        if case%4 == 0: x[:]=1
        if case%6 == 0: x[0]=0
        if case%11 == 0: x[:]=0; x[-1]=1
        v=rng.normal(size=(n,d))
        if case%3 == 0: v[:]=0
        r=evaluate_one(x,v,int(rng.integers(1,32)),float(rng.uniform(.001,.5)))
        assert r['oracle_exact_score_value_true_error'] <= r['oracle_exact_score_value_certificate'] + 2e-10


def test_ideal_constant_values():
    n=256; d=6
    w=np.ones(n)/n
    v=np.tile(np.arange(d,dtype=float),(n,1))
    r=evaluate_one(w,v,8,.02)
    assert r['oracle_exact_score_value_pages']==1
    assert r['greedy_hindsight_pages']==1
    assert r['oracle_mass_pages'] > 25
    assert r['oracle_exact_score_value_true_error'] < 1e-12


def test_zero_values():
    r=evaluate_one(np.ones(32),np.zeros((32,4)),8,.02)
    assert r['oracle_exact_score_value_pages']==1
    assert r['oracle_exact_score_value_true_error']==0


@pytest.mark.parametrize('w,v',[
    (np.zeros(4),np.ones((4,2))),
    (np.array([1,0,np.nan,0]),np.ones((4,2))),
    (np.array([1,0,0,0]),np.ones((3,2))),
    (np.array([1,-1,2,0]),np.ones((4,2))),
])
def test_reject_invalid(w,v):
    with pytest.raises(ValueError):
        evaluate_one(w,v)
