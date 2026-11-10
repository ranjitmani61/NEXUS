"""Property, identity, and negative-control tests for G2 FFN verification proxy."""
import unittest
import numpy as np
from g2_ffn_verification_cost import (
    expected_length, verifier_speed, beta_threshold, rho_budget_for_2x,
    swiglu_serial, swiglu_batch, frontier, N_PROPOSALS, P_VALUES, C2_ATTENTION_SHARES)

class TestMath(unittest.TestCase):
    def test_length_closed_form(self):
        for p in [0,.1,.5,.8,.95,.999,1]:
            for k in [1,2,4,8]:
                a=expected_length(p,k)
                b=(1-p**(k+1))/(1-p) if p!=1 else k+1
                self.assertAlmostEqual(a,b,places=10)
    def test_invalid_length(self):
        for p,k in [(-.2,2),(1.1,2),(.5,-1)]:
            with self.assertRaises(ValueError):expected_length(p,k)
    def test_speed_zero_lag(self):
        self.assertAlmostEqual(verifier_speed(1,4,.5,1,0,0,0),5)
    def test_ref_old_assumption(self):
        f=.7052066730549075;p=.9;k=4
        v=beta_threshold(p,k,f,1)
        self.assertAlmostEqual(v,.30046156410023434,places=12)
    def test_rho_monotone(self):
        for f in [.2,.4,.7,.9]:
            self.assertLess(verifier_speed(.9,4,f,2,.25),verifier_speed(.9,4,f,1,.25))
    def test_beta_monotone(self):
        for f in [.2,.4,.7,.9]:
            self.assertLess(verifier_speed(.9,4,f,1,1),verifier_speed(.9,4,f,1,0))
    def test_threshold_equality(self):
        for f in [.2,.4,.7,.9]:
            for rho in [.5,1,1.5,2.5]:
                v=beta_threshold(.9,4,f,rho)
                if v>=0:self.assertAlmostEqual(verifier_speed(.9,4,f,rho,v),2.0,places=11)
    def test_rho_equality(self):
        for f in [.2,.4,.7,.9]:
            r=rho_budget_for_2x(.95,8,f,.1)
            if r>0:self.assertAlmostEqual(verifier_speed(.95,8,f,r,.1),2,places=11)
    def test_negative_control_no_reuse_draft(self):
        self.assertLess(verifier_speed(.8,4,.7,3,1),1.0)
    def test_negative_control_no_free_bonus(self):
        self.assertAlmostEqual(expected_length(0,4),1)
    def test_invalid_cost(self):
        for vals in [(.9,4,-.1,1,0),(.9,4,.5,-1,0),(.9,4,.5,1,-1),(.9,4,.5,1,0,-.1)]:
            with self.assertRaises(ValueError):verifier_speed(*vals)
    def test_invalid_threshold(self):
        for f,k in [(0,4),(.4,0)]:
            with self.assertRaises(ValueError):beta_threshold(.9,k,f,1)
    def test_invalid_rho_budget(self):
        with self.assertRaises(ValueError):rho_budget_for_2x(.9,4,1)
    def test_bound_behavior(self):
        for rho in [.5,1,2]:
            for f in [.1,.7]:
                threshold=beta_threshold(.9,4,f,rho)
                if threshold>=0:
                    self.assertGreaterEqual(verifier_speed(.9,4,f,rho,threshold),2-1e-12)
                    self.assertLess(verifier_speed(.9,4,f,rho,threshold+.01),2)
    def test_zero_draft_still_not_free_work(self):
        self.assertLess(verifier_speed(1,1,.8,10,0,0,0),2)

class TestKernel(unittest.TestCase):
    def test_numerical_equivalence(self):
        for seed in range(5):
            rng=np.random.default_rng(seed)
            for din,dout in [(7,11),(16,8),(32,64)]:
                for n in [1,2,3,5,9]:
                    wg=rng.normal(size=(din,dout)).astype('f')/np.sqrt(din)
                    wu=rng.normal(size=(din,dout)).astype('f')/np.sqrt(din)
                    wd=rng.normal(size=(dout,din)).astype('f')/np.sqrt(dout)
                    x=rng.normal(size=(n,din)).astype('f')
                    a=swiglu_serial(x,wg,wu,wd); b=swiglu_batch(x,wg,wu,wd)
                    self.assertTrue(np.allclose(a,b,rtol=2e-5,atol=2e-5), (seed,din,dout,n))
    def test_wrong_shape(self):
        rng=np.random.default_rng(123)
        x=rng.normal(size=(2,3)).astype('f'); w=rng.normal(size=(3,5)).astype('f')
        with self.assertRaises(ValueError):swiglu_batch(x,w,w,w)
    def test_all_queries_counted(self):
        self.assertEqual(N_PROPOSALS,(1,2,4,8))
    def test_matrix_total(self):
        # 4 host anchors x 4 proposal lengths x 4 acceptance assumptions
        self.assertEqual(len(C2_ATTENTION_SHARES)*len(N_PROPOSALS)*len(P_VALUES),64)

if __name__=='__main__':unittest.main()

class TestComponentBreakdown(unittest.TestCase):
    def test_no_scale_equals_old_model(self):
        for f in [.2,.4,.7]:
            a=verifier_speed(.9,4,f,1,0,scaled_share=.1)
            b=verifier_speed(.9,4,f,1,0)
            self.assertAlmostEqual(a,b)
    def test_component_only_improves_vs_all(self):
        for f in [.2,.4,.7]:
            g=.1; rho=2.2
            self.assertGreater(beta_threshold(.9,4,f,rho,scaled_share=g),beta_threshold(.9,4,f,rho))
            self.assertGreater(verifier_speed(.9,4,f,rho,.25,scaled_share=g),verifier_speed(.9,4,f,rho,.25))
    def test_component_threshold_exact(self):
        f=.3797642644870673;g=.2590455626976019;rho=2.196
        beta=beta_threshold(.9,4,f,rho,scaled_share=g)
        self.assertAlmostEqual(beta,.3539903285397069,places=10)
        self.assertAlmostEqual(verifier_speed(.9,4,f,rho,beta,scaled_share=g),2,places=11)
    def test_component_invalid(self):
        with self.assertRaises(ValueError):beta_threshold(.9,4,.8,2,scaled_share=.3)
    def test_component_zero(self):
        f=.4
        self.assertAlmostEqual(verifier_speed(.9,4,f,1,.2),verifier_speed(.9,4,f,123,.2,scaled_share=0))
    def test_rho_budget_component(self):
        f=.38;g=.25
        rho=rho_budget_for_2x(.95,4,f,.2,scaled_share=g)
        self.assertAlmostEqual(verifier_speed(.95,4,f,rho,.2,scaled_share=g),2,places=10)
    def test_frontier_has_mlp_only_on_c2(self):
        from g2_ffn_verification_cost import C2_MLP_SHARES
        fake=[{"label": "Qwen2.5-0.5B", "cases":[{"k":k,"rho_batch_vs_single":2.2} for k in N_PROPOSALS]},
              {"label": "Granite-4.0-micro-base", "cases":[{"k":k,"rho_batch_vs_single":2.2} for k in N_PROPOSALS]}]
        rows=frontier(fake)
        self.assertEqual(len(rows),64)
        self.assertEqual(sum(x['beta_threshold_if_only_mlp_scaled'] is not None for x in rows),32)
        self.assertEqual(set(C2_MLP_SHARES),{'Granite host C / 16K / b1','Qwen host C / 32K / b1'})

class TestScorecard(unittest.TestCase):
    def test_scorecard_immutable_shape(self):
        from g2_ffn_verifier_scorecard import audit
        fake={"matrix_cells":64,"assumed_seed":1,"models":[{"label":"X","cases":[{"k":4,"n_queries":5,"rho_batch_vs_single":2,
              "batch_vs_serial_time":.4,"max_abs_output_error":1e-8}]}],"frontier":[]}
        second={**fake,"assumed_seed":2}
        x=audit(fake,second)
        self.assertEqual(len(x['models'][0]['cases']),1)
    def test_scorecard_refuses_same_seed(self):
        from g2_ffn_verifier_scorecard import audit
        fake={"matrix_cells":64,"assumed_seed":1,"models":[],"frontier":[]}
        with self.assertRaises(AssertionError):audit(fake,fake)
