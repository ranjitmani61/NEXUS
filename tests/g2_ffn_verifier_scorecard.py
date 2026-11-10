#!/usr/bin/env python3
"""Deterministic audit/scorecard of the two unpooled FFN microbenchmark runs."""
import json, math, sys
from pathlib import Path

def audit(a,b):
    assert a['matrix_cells']==b['matrix_cells']==64
    assert a['assumed_seed']!=b['assumed_seed']
    models=[]
    for m1,m2 in zip(a['models'],b['models']):
        assert m1['label']==m2['label']
        cases=[]
        for c1,c2 in zip(m1['cases'],m2['cases']):
            assert c1['k']==c2['k']
            rho1=c1['rho_batch_vs_single']; rho2=c2['rho_batch_vs_single']
            assert rho1>0 and rho2>0
            cases.append({'k':c1['k'],'n_queries':c1['n_queries'],
                          'rho_seed1':rho1,'rho_seed2':rho2,
                          'fractional_difference':rho2/rho1-1,
                          'batch_vs_serial_seed1':c1['batch_vs_serial_time'],
                          'batch_vs_serial_seed2':c2['batch_vs_serial_time'],
                          'max_numerical_error_seed1':c1['max_abs_output_error'],
                          'max_numerical_error_seed2':c2['max_abs_output_error']})
        models.append({'model_shape':m1['label'],'cases':cases})
    sel=[]
    for x in a['frontier']:
        if x['k']==4 and x['p_assumed']==.9:sel.append(x)
    for x in sel:
        assert x['beta_threshold_if_all_nonattention_scaled']<x['beta_threshold_old_rho1']
        if x['beta_threshold_if_only_mlp_scaled'] is not None:
            assert x['beta_threshold_if_only_mlp_scaled']>x['beta_threshold_if_all_nonattention_scaled']
    return {'date':'2026-11-10','study':'G2 component-resolved verifier FFN cost sensitivity',
            'source_kind':'local synthetic FP32 FFN kernel timings and external C2 timing shares',
            'models':models,'frontier_scenarios':len(a['frontier']),
            'rho1_idealized_beta0_2x_count':None, # computed below
            'all_nonattention_scaled_beta0_2x_count':sum(x['reachable_at_beta0_all_nonattention_scaled'] for x in a['frontier']),
            'component_mlp_scaled_scenarios':sum(x['beta_threshold_if_only_mlp_scaled'] is not None for x in a['frontier']),
            'component_mlp_scaled_beta0_2x_count':sum(x['reachable_at_beta0_only_mlp_scaled'] or False for x in a['frontier']),
            'selected_k4_p09':sel,
            'gate':'G1 not novel; G2 sensitivity only; G5 not run on real speculative target; no G6 authorization'}

def main():
    root=Path(__file__).resolve().parent.parent
    a=json.loads((root/'results/G2_ffn_verifier_cost.json').read_text())
    b=json.loads((root/'results/G2_ffn_verifier_seed2.json').read_text())
    ans=audit(a,b)
    from g2_ffn_verification_cost import verifier_speed
    ans['rho1_idealized_beta0_2x_count']=sum(verifier_speed(x['p_assumed'],x['k'],x['measured_attention_share_external'],1,0)>=2 for x in a['frontier'])
    p=root/'results/G2_ffn_verifier_scorecard.json'
    p.write_text(json.dumps(ans,indent=2,sort_keys=True)+'\n')
    print('2 measured model-shaped proxies; 64 matrix cells, prior idealized beta0 >=2:',ans['rho1_idealized_beta0_2x_count'])
    print('all-nonattn-scaled beta0 >=2:',ans['all_nonattention_scaled_beta0_2x_count'])
    print('mlp-only beta0 >=2:',ans['component_mlp_scaled_beta0_2x_count'],'/',ans['component_mlp_scaled_scenarios'])

if __name__=='__main__':main()
