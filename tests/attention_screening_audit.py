#!/usr/bin/env python3
"""Project 01 - Auditable synthetic page-screening experiment, v1.0 (2026-11-03).

Scientific intent: audit a KNOWN algorithm, not invent a processor.
Requires numpy. Run from any directory:
  OPENBLAS_NUM_THREADS=1 python tests/attention_screening_audit.py \
      --output results/attention_screening_audit.json
All results are synthetic mathematical/numerical software results, NOT chip metrics.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

D, B, L = 64, 128, 32
TOL = 0.02
SEEDS = list(range(20261103, 20261115))  # 12 predetermined seeds
REGIMES = ('clustered', 'unstructured', 'uniform', 'needle')
BOUNDS = ('sphere', 'box')


def build_example(seed: int, regime: str, blocks=B, length=L, dim=D):
    r = np.random.default_rng(seed)
    q = r.standard_normal(dim)
    v = r.standard_normal((blocks, length, dim))
    if regime == 'clustered':
        centers = r.standard_normal((blocks, dim))
        centers[min(11, blocks - 1)] = 1.5*q + r.normal(0, 0.15, dim)
        k = centers[:, None, :] + r.normal(0, 0.04, (blocks, length, dim))
    elif regime == 'unstructured':
        k = r.standard_normal((blocks, length, dim))
    elif regime == 'uniform':
        k = np.zeros((blocks, length, dim))
    elif regime == 'needle':
        k = r.standard_normal((blocks, length, dim))
        k[-1, -1] = 2.0*q  # a rare, strong alignment embedded in an ordinary block
    else:
        raise ValueError(regime)
    return q, k, v


def bound_metadata(k, v, q, bound):
    blocks, length, dim = k.shape
    scale = np.sqrt(dim)
    vnorm_upper = float(np.max(np.linalg.norm(v, axis=2)))
    if bound == 'sphere':
        centers = k.mean(axis=1)
        radii = np.linalg.norm(k-centers[:, None, :], axis=2).max(axis=1)
        upper = (centers @ q + np.linalg.norm(q)*radii)/scale
        floats_per_page = dim + 2  # center[d], radius, value-norm bound
    elif bound == 'box':
        lows = k.min(axis=1)
        highs = k.max(axis=1)
        upper = (highs @ np.maximum(q, 0) + lows @ np.minimum(q, 0))/scale
        floats_per_page = 2*dim + 1  # minimum[d], maximum[d], value-norm bound
    else:
        raise ValueError(bound)
    return upper, vnorm_upper, blocks*floats_per_page*4


def exact_attention(q, k, v):
    scores = np.einsum('bld,d->bl', k, q)/np.sqrt(len(q))
    exp = np.exp(scores - scores.max())
    return np.einsum('bl,bld->d', exp, v)/np.sum(exp), scores


def upper_logsumexp(xs):
    if len(xs) == 0:
        return -np.inf
    peak = np.max(xs)
    return float(peak + np.log(np.sum(np.exp(xs - peak))))


def screen(q, k, v, *, bound, tol=TOL):
    blocks, length, dim = k.shape
    upper, vmax, meta_bytes = bound_metadata(k, v, q, bound)
    # All block bounds rely ONLY on precomputed metadata, not per-query full K/V loads.
    # This code stores K/V to model what would be fetched if a block is selected.
    selected = []
    numerator = np.zeros(dim)
    normalizer = 0.0
    score_shift = -np.inf
    cert = 2.0
    for b in np.argsort(-upper):
        bs = (k[b] @ q)/np.sqrt(dim)
        next_shift = max(score_shift, float(bs.max()))
        shrink = 0 if score_shift == -np.inf else np.exp(score_shift-next_shift)
        w = np.exp(bs - next_shift)
        numerator = shrink*numerator + w @ v[b]
        normalizer = shrink*normalizer + np.sum(w)
        score_shift = next_shift
        selected.append(int(b))
        remaining = np.ones(blocks, dtype=bool)
        remaining[selected] = False
        # Each of L keys in an unvisited page has softmax score <= page upper.
        log_u = upper_logsumexp(upper[remaining] + np.log(length))
        log_z = score_shift + np.log(normalizer)
        # 2*U/(Z+U) computed stably, in the log domain.
        if log_u == -np.inf:
            cert = 0.0
        elif log_z >= log_u:
            t = np.exp(log_u-log_z)
            cert = float(2*t/(1+t))
        else:
            t = np.exp(log_z-log_u)
            cert = float(2/(1+t))
        if cert <= tol:
            break
    approximation = numerator/normalizer
    dense_bytes = blocks*length*dim*2*4  # both K and V are float32-equivalent
    fetched_bytes = len(selected)*length*dim*2*4
    # Costs exclude sorting, math operations, caching/transactions and silicon cycles.
    return {
        'bound': bound,
        'selected_blocks': len(selected),
        'selected_fraction': len(selected)/blocks,
        'certificate': cert,
        'normalized_error': None,  # added by caller with dense reference
        'vnorm_bound': vmax,
        'metadata_read_bytes_per_query': meta_bytes,
        'fetched_KV_bytes_per_query': fetched_bytes,
        'dense_KV_bytes_per_query': dense_bytes,
        'modeled_read_ratio': (fetched_bytes+meta_bytes)/dense_bytes,
        # pessimistic initial one-time scanning K+V, amortized across Q queries
        'modeled_read_ratio_Q1_with_build': (fetched_bytes+meta_bytes+dense_bytes)/dense_bytes,
        'modeled_read_ratio_Q128_with_build': (fetched_bytes+meta_bytes+dense_bytes/128)/dense_bytes,
        '_approx': approximation,
        '_score_upper': upper,
    }


def run_audit():
    observations = []
    cert_violations = []
    score_bound_violations = []
    for regime in REGIMES:
        for seed in SEEDS:
            q, k, v = build_example(seed, regime)
            truth, scores = exact_attention(q,k,v)
            for method in BOUNDS:
                result = screen(q,k,v,bound=method)
                value_err = np.linalg.norm(result['_approx']-truth)/result['vnorm_bound']
                result['normalized_error'] = float(value_err)
                ub = result.pop('_score_upper')
                result.pop('_approx')
                if np.any(scores.max(axis=1) > ub + 1e-10):
                    score_bound_violations.append((regime, seed, method))
                if value_err > result['certificate'] + 1e-10:
                    cert_violations.append((regime, seed, method))
                if result['certificate'] > TOL+1e-10:
                    cert_violations.append(('stopping', regime, seed, method))
                observations.append({'regime':regime, 'seed':seed, **result})
    aggregate = []
    for regime in REGIMES:
        for method in BOUNDS:
            group = [r for r in observations if r['regime']==regime and r['bound']==method]
            aggregate.append({
                'regime':regime,
                'method':method,
                'n':len(group),
                'median_selected_blocks':float(np.median([r['selected_blocks'] for r in group])),
                'min_selected_blocks':int(min(r['selected_blocks'] for r in group)),
                'max_selected_blocks':int(max(r['selected_blocks'] for r in group)),
                'median_traffic_ratio_without_build':float(np.median([r['modeled_read_ratio'] for r in group])),
                'median_traffic_ratio_with_build_Q1':float(np.median([r['modeled_read_ratio_Q1_with_build'] for r in group])),
                'median_traffic_ratio_with_build_Q128':float(np.median([r['modeled_read_ratio_Q128_with_build'] for r in group])),
                'max_normalized_output_error':float(max(r['normalized_error'] for r in group)),
                'max_certificate':float(max(r['certificate'] for r in group)),
            })
    return {
        'meta':{
            'project':'Project 01', 'version':'1.0', 'date':'2026-11-03',
            'description':'Auditing known sparse-attention screening, NOT novel hardware',
            'scope':'Synthetic single-head, single-query decode; CPU Python, no hardware timings',
            'dimensions':D,'blocks':B,'tokens_per_block':L,'tol':TOL,
            'seeds':SEEDS,'regimes':REGIMES,'bounds':BOUNDS,
            'dtype':'float64 simulation, float32-EQUIVALENT modeled memory bytes',
            'python':platform.python_version(), 'numpy':np.__version__,
            'limits':[
                'No model traces, inference accuracy, model quality, latency, energy or chip benchmarks',
                'Construction modeled as one-time full K/V scan; excludes arithmetic, memory transactions and cache hierarchy',
                'Top-K sparse methods have different quality contracts; bound alternative is only Quest-inspired, not a Quest reproduction',
                'Certificate is exact-arithmetic in derivation, not a verified finite-precision guarantee',
                'Assumes a cached global upper bound on V vector norms and exact K/V float values',
                'No precision/quantization kernels or multi-head/model outputs',
            ],
        },
        'aggregate':aggregate,
        'tests':{
            'scenarios':len(observations),
            'score_bound_violations':score_bound_violations,
            'certificate_violations':cert_violations,
            'pass':not score_bound_violations and not cert_violations,
        },
        'observations':observations,
    }


def small_property_tests():
    # Boundary and adversarial checks over smaller random sizes.
    tests = 0
    for dim in (1,3,8,64):
        for blocks in (2,7):
            for length in (1,5,16):
                for case in ('clustered','unstructured','uniform','needle'):
                    q,k,v = build_example(12345+dim+blocks+length,case,blocks,length,dim)
                    truth, scores = exact_attention(q,k,v)
                    for method in BOUNDS:
                        result = screen(q,k,v,bound=method,tol=0.02)
                        actual=np.linalg.norm(result['_approx']-truth)/result['vnorm_bound']
                        assert actual <= result['certificate']+1e-10, (dim,blocks,length,case,method,actual,result['certificate'])
                        assert np.all(scores.max(axis=1) <= result['_score_upper']+1e-10)
                        assert result['certificate'] <= 0.0200000001
                        tests += 1
    return tests


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/attention_screening_audit.json')
    args=parser.parse_args()
    run=run_audit()
    run['tests']['additional_boundary_cases']=small_property_tests()
    p=Path(args.output)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(run,indent=2)+'\n',encoding='utf-8')
    print('Saved',p.resolve())
    print('Tests:',run['tests'])
    for r in run['aggregate']:
        print('{:12s} {:6s} blocks={:5.1f}x/128 traffic={:7.2%} Q1-build={:7.2%} Q128-build={:7.2%} maxerr={:.7f}'.format(
            r['regime'], r['method'],r['median_selected_blocks'],r['median_traffic_ratio_without_build'],
            r['median_traffic_ratio_with_build_Q1'],r['median_traffic_ratio_with_build_Q128'],r['max_normalized_output_error']))
    if not run['tests']['pass']:
        sys.exit(2)

if __name__ == '__main__':
    main()