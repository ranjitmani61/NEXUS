#!/usr/bin/env python3
"""Project 01 G3: guarded mixed-precision *screening* of exact stored-operand attention.

No claim of hardware implementation, novel processor, or formal end-to-end proof.
The low-precision (FP16 or BF16) tensors are stored inputs; score bounds use FP32
and a conventional gamma error envelope, while final attention uses FP64.

The observed output error is relative to exact attention on QUANTIZED operands.
The conservative 2*omitted_softmax_mass result is conditional on correct
score intervals; code's FP64 endpoint construction is not rigorously directed
rounding and must not be called a formal hardware certificate.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import ml_dtypes
import numpy as np

FORMATS = {'fp16': np.dtype('float16'), 'bf16': np.dtype(ml_dtypes.bfloat16)}
TOLERANCE = 0.02
U32 = 2**-24


def stored(x: np.ndarray, fmt: str) -> np.ndarray:
    return np.asarray(x, dtype=FORMATS[fmt])


def gen_case(kind: str, seed: int, *, pages=64, size=16, d=64, queries=32, fmt='fp16'):
    rng = np.random.default_rng(seed)
    n = pages*size
    V = rng.normal(size=(n, 16)).astype('float64')
    if kind == 'clustered':
        base = np.ones(d, dtype='float64')
        K = rng.normal(0, 0.025, (n, d)) - 5*base
        K[:size] = rng.normal(0, 0.025, (size, d)) + 5*base
        Q = 1 + rng.normal(0, 0.015, (queries, d))
    elif kind == 'random':
        K = rng.normal(size=(n, d))
        Q = rng.normal(size=(queries, d))
    elif kind == 'uniform':
        K = rng.normal(0, 0.0001, size=(n,d))
        Q = rng.normal(size=(queries,d))
    elif kind == 'needle':
        K = rng.normal(0, 0.2, (n,d))
        K[0] = np.ones(d)*5
        Q = 1 + rng.normal(0, 0.015, (queries,d))
    elif kind == 'cancellation':
        assert pages == 2 and size == 1 and d == 3
        middle, huge = (16, 32768) if fmt == 'fp16' else (8, 2048)
        K = np.array([[0,10,0], [huge,middle,-huge]],dtype='float64')
        Q = np.ones((queries,d))
        V = np.array([[1.,0.,0.],[-1.,0.,0.]])
    else:
        raise ValueError(kind)
    return stored(K,fmt), stored(Q,fmt), V


class PageCache:
    def __init__(self, K, V, page_size):
        assert K.ndim == 2 and V.ndim == 2 and K.shape[0]==V.shape[0] and K.shape[0]%page_size==0
        self.K = K
        self.V = np.asarray(V,dtype='float64')
        self.size = page_size
        self.pages = len(K)//page_size
        self.d = K.shape[1]
        self.version=0
        # Reference arrays are intentionally preconverted for matched reference measurement.
        self.K64 = K.astype('float64')


class PageMetadata:
    def __init__(self, cache: PageCache):
        # Low-precision extrema from stored inputs; no key reconstruction error
        reshaped = cache.K.reshape(cache.pages,cache.size,cache.d)
        self.lo = np.min(reshaped,axis=1).astype('float32')
        self.hi = np.max(reshaped,axis=1).astype('float32')
        self.version = cache.version


def dense(cache: PageCache, q_stored: np.ndarray, page=None):
    q = q_stored.astype('float64')
    sl = slice(None) if page is None else slice(page*cache.size,(page+1)*cache.size)
    score = (cache.K64[sl] @ q)/math.sqrt(cache.d)
    m = max(score)
    w = np.exp(score-m)
    w /= np.sum(w)
    return w @ cache.V[sl]


def guarded_page_score_intervals(cache: PageCache, meta: PageMetadata, q_stored):
    """Return (lower,upper,reason) for stored-operand, normalized token scores.

    All bounds are conditional on standard FP32 floating-point assumptions.
    No directed rounding is provided for interval-endpoint construction.
    """
    if meta.version!=cache.version: return None,None,'stale_version'
    if len(q_stored)!=cache.d: return None,None,'bad_shape'
    q = q_stored.astype('float32')
    if not (np.isfinite(q).all() and np.isfinite(meta.lo).all() and np.isfinite(meta.hi).all()):
        return None,None,'nonfinite'
    if cache.d <= 0 or 2*cache.d*U32 >= 1: return None,None,'gamma_domain'
    max_mag = max(float(np.max(np.abs(q))),float(np.max(np.abs(meta.lo))),float(np.max(np.abs(meta.hi))))
    if max_mag > 1e10: return None,None,'range_rejection'
    low_terms = np.where(q[None,:]>=0,meta.lo,meta.hi) * q[None,:]
    high_terms = np.where(q[None,:]>=0,meta.hi,meta.lo) * q[None,:]
    if not (np.isfinite(low_terms).all() and np.isfinite(high_terms).all()):
        return None,None,'nonfinite_products'
    if (np.any((low_terms!=0)&(np.abs(low_terms)<np.finfo(np.float32).tiny)) or
            np.any((high_terms!=0)&(np.abs(high_terms)<np.finfo(np.float32).tiny))):
        return None,None,'subnormal_products'
    low_est = low_terms.sum(axis=1,dtype=np.float32).astype('float64')
    high_est = high_terms.sum(axis=1,dtype=np.float32).astype('float64')
    gamma = (2*cache.d*U32)/(1-2*cache.d*U32)
    lo_mag = np.abs(low_terms.astype('float64')).sum(axis=1)
    hi_mag = np.abs(high_terms.astype('float64')).sum(axis=1)
    padlo = 32*np.finfo(np.float64).eps*(1+np.abs(low_est)+gamma*lo_mag)
    padhi = 32*np.finfo(np.float64).eps*(1+np.abs(high_est)+gamma*hi_mag)
    lbs = (low_est - (gamma*lo_mag + padlo))/math.sqrt(cache.d)
    ubs = (high_est + (gamma*hi_mag + padhi))/math.sqrt(cache.d)
    if not (np.isfinite(lbs).all() and np.isfinite(ubs).all()):
        return None,None,'nonfinite_bounds'
    return lbs,ubs,'ok'


def guarded_admission(cache: PageCache, meta: PageMetadata, q_stored, *, eps=TOLERANCE):
    """Screen at most one page using metadata score intervals.

    Conservative 2 * omitted attention mass is an error bound normalized by
    max ||V_i||, provided metadata intervals enclose all exact stored scores.
    """
    if not 0 < eps < 2: raise ValueError('invalid epsilon')
    lbs,ubs,reason=guarded_page_score_intervals(cache,meta,q_stored)
    if reason!='ok': return False,None,None,reason
    selected = int(np.argmax(lbs))
    mask = np.ones(cache.pages,dtype=bool)
    mask[selected] = False
    other = ubs[mask]
    if len(other)==0: return True,selected,0.,'only_page'
    pivot=max(lbs[selected],float(np.max(other)))
    skip = float(np.exp(other-pivot).sum())
    keep = math.exp(float(lbs[selected]-pivot))
    if keep+skip<=0: return False,None,None,'numerical_rejection'
    mass_bound=float(2*skip/(keep+skip))
    if mass_bound<=eps:
        return True,selected,mass_bound,'admitted'
    return False,None,mass_bound,'dense_fallback'


def query(cache, meta, q, *, eps=TOLERANCE):
    yes,pg,bound,reason=guarded_admission(cache,meta,q,eps=eps)
    out=dense(cache,q,page=pg if yes else None)
    return out, yes, bound, reason


def check_intervals(cache,meta,q):
    """Check exact score min/max against all page intervals with same method."""
    qq=q.astype('float32')
    d=cache.d
    low=np.where(qq[None,:]>=0,meta.lo,meta.hi).astype('float64')*qq.astype('float64')
    high=np.where(qq[None,:]>=0,meta.hi,meta.lo).astype('float64')*qq.astype('float64')
    exact_low=low.sum(axis=1)/math.sqrt(d)
    exact_high=high.sum(axis=1)/math.sqrt(d)
    actual=(cache.K64@q.astype('float64')).reshape(cache.pages,cache.size)/math.sqrt(d)
    return bool(np.all(actual >= exact_low[:,None]-1e-10) and np.all(actual <= exact_high[:,None]+1e-10))


def timed(fn, runs=1):
    start=time.perf_counter_ns()
    for _ in range(runs): fn()
    return (time.perf_counter_ns()-start)/1e6


def benchmark(fmt='fp16', seed=1001, cases=('clustered','random','uniform','needle'), queries=32, pages=64, page_size=16,d=64):
    result=[]
    for kind in cases:
        K,Q,V=gen_case(kind,seed,pages=pages,size=page_size,d=d,queries=queries,fmt=fmt)
        cache=PageCache(K,V,page_size)
        build_ms=timed(lambda:PageMetadata(cache))
        meta=PageMetadata(cache)
        # Warm both and precompile BLAS dispatch before recording separate loops.
        for q in Q[:3]: dense(cache,q); query(cache,meta,q)
        outputs=[];admissions=0;maxerror=0.; maxovershoot=0.; counts={}
        refms=[]; candidate_ms=[]
        for q in Q:
            # Adjacent timing interleaves reference and candidate to reduce drift.
            t0=time.perf_counter_ns(); ref=dense(cache,q); refms.append((time.perf_counter_ns()-t0)/1e6)
            t0=time.perf_counter_ns(); out,yes,bound,reason=query(cache,meta,q)
            candidate_ms.append((time.perf_counter_ns()-t0)/1e6)
            counts[reason]=counts.get(reason,0)+1
            admissions+=int(yes)
            vmax=max(1e-30,float(np.max(np.linalg.norm(cache.V,axis=1))))
            error=float(np.linalg.norm(ref-out)/vmax)
            maxerror=max(maxerror,error)
            if yes:
                maxovershoot=max(maxovershoot,error-bound)
                if error>bound+1e-9: raise AssertionError(f'unsafe admission {fmt} {kind} {error=} {bound=}')
            if error>TOLERANCE+1e-9: raise AssertionError(f'output violation {fmt} {kind}')
            outputs.append(bool(check_intervals(cache,meta,q)))
        if not all(outputs): raise AssertionError('bad exact metadata extrema')
        ref_total=sum(refms)
        cand_total=sum(candidate_ms)+build_ms
        result.append({'format':fmt,'case':kind,'seed':seed,'queries':queries,'pages':pages,
                       'page_size':page_size,'dimension':d,'admitted':admissions,
                       'fraction_admitted':admissions/queries,'max_normalized_error':maxerror,
                       'max_bound_overshoot':maxovershoot,'interval_exact_check_pass':all(outputs),
                       'reason_counts':counts,
                       'reference_total_ms':ref_total,'candidate_query_ms':sum(candidate_ms),
                       'metadata_build_ms':build_ms,'candidate_total_ms':cand_total,
                       'total_ratio_to_dense':cand_total/ref_total if ref_total else None,
                       'query_ratio_to_dense':sum(candidate_ms)/ref_total if ref_total else None,
                       'reference_median_ms':statistics.median(refms),
                       'candidate_median_ms':statistics.median(candidate_ms),
                       'metadata_bytes':meta.lo.nbytes+meta.hi.nbytes,
                       'stored_key_bytes':cache.K.nbytes})
    return result


def test_counterexample():
    cases=[]
    for fmt in FORMATS:
        K,Q,V=gen_case('cancellation',seed=0,pages=2,size=1,d=3,queries=1,fmt=fmt)
        cache=PageCache(K,V,1);meta=PageMetadata(cache)
        out,yes,bound,reason=query(cache,meta,Q[0])
        reference=dense(cache,Q[0]); err=float(np.linalg.norm(out-reference)/max(np.linalg.norm(V,axis=1)))
        assert err<1e-10, (fmt,err)
        cases.append({'format':fmt,'admitted':yes,'reason':reason,'error':err,'bound':bound})
    return cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/g3_guarded_mixed_precision_rep1.json')
    ap.add_argument('--queries',type=int,default=32)
    args=ap.parse_args()
    rows=[]
    for fmt in FORMATS:
        for seed in (1001,1031):
            rows.extend(benchmark(fmt=fmt,seed=seed,queries=args.queries))
    report={'version':'1.0','date':'2026-11-03','gate':'G3 IN PROGRESS',
            'measurement':'local CPU NumPy wall-time; one BLAS thread recommended',
            'accuracy_reference':'FP64 exact softmax over stored low-precision operands',
            'warning':'No pretrained models, no RTL/hardware, no formal interval endpoints; quantization error not covered.',
            'python':platform.python_version(),'numpy':np.__version__,
            'ml_dtypes':ml_dtypes.__version__,'cpu':platform.processor(),
            'openblas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),
            'witnesses':test_counterexample(),'cases':rows}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    for row in rows:
        print(row['format'],row['case'],row['seed'],'admit',row['admitted'],'/',row['queries'],
              'total ratio',round(row['total_ratio_to_dense'],3),'max error',round(row['max_normalized_error'],8))
    print('RESULTS',path)

if __name__=='__main__':main()
