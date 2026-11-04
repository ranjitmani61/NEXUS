#!/usr/bin/env python3
"""G4 oracle-headroom diagnostic (not trained LLM, not hardware benchmark).

Run from project root with BLAS pinned to one thread:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g4_oracle_headroom_audit.py

The oracle reads the entire query-key score vector and has no implementable
speed advantage: it is used ONLY to estimate idealized page-selection headroom.
"""
from __future__ import annotations
import argparse
import json
import math
import platform
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import g3_g4_autoregressive_trace_audit as old

VERSION = '1.0'
N = 512
DECODE = 20
SEEDS = (20261103, 20261123)
REGIMES = ('iid_tokens', 'periodic_tokens', 'piecewise_tokens', 'rare_marker_tokens')
EVAL_HEADS = (0, 2)
PAGE_SIZE = old.PAGE_LEN  # 32
TOL = old.EPSILON   # .02 (mass certificate <= 0.02)


def mass_oracle(scores: np.ndarray, n: int, page_size: int=PAGE_SIZE, tol: float=TOL):
    """Infeasible exact-info bound: minimal pages to retain >=(1-tol/2) mass.

    True scores already known (all keys read); sorts pages by exact exp mass.
    This gives the minimum pages needed for the *mass-based* 2m certificate;
    it is NOT a general information-theoretic limit on approximate attention.
    """
    stable = np.exp(scores - scores.max())
    p = stable / stable.sum()
    page_masses = np.array([p[i:min(i+page_size,n)].sum() for i in range(0,n,page_size)])
    masses = np.sort(page_masses)[::-1]
    retain_n = int(np.searchsorted(np.cumsum(masses), 1-tol/2-1e-12, side='left')+1)
    assert 1 <= retain_n <= len(masses)
    return retain_n, page_masses



def hindsight_value_oracle(scores, values, dense, vmax, page_size=PAGE_SIZE, tol=TOL):
    """Optimistic prefix-of-pages heuristic based on exact QK and V values.

    Never an implementable method: all QK and V rows are needed up front, and
    minimizing error over arbitrary subsets is NOT solved by this greedy order.
    It can identify when the mass-only certificate is conservative.
    """
    n = len(scores)
    sh = scores.max()
    w = np.exp(scores-sh)
    page_ids = np.arange(n)//page_size
    page_masses = np.bincount(page_ids, weights=w)
    order = np.argsort(-page_masses, kind='stable')
    yz = np.zeros(values.shape[1],dtype=np.float64)
    z = 0.0
    for step,p in enumerate(order,1):
        sl=slice(int(p)*page_size, min((int(p)+1)*page_size,n))
        yz+=w[sl]@values[sl]
        z+=w[sl].sum()
        if np.linalg.norm(yz/z-dense)/vmax <= tol+1e-12:
            return step
    raise AssertionError('full output should pass')


def make_rows() -> list[dict]:
    rows = []
    # Maintain old module N to satisfy its internally fixed N x N future mask.
    old.N = N
    old.DECODE = DECODE
    for seed in SEEDS:
        embeddings, weights = old.make_weights(seed)
        for regime in REGIMES:
            ids = old.tokens_for(regime, seed, n=N)
            traces = old.generate_traces(ids, embeddings, weights)
            for layer, (q_all,k_all,v_all) in enumerate(traces):
                for head in EVAL_HEADS:
                    cache = old.OnlinePages(old.HEAD_DIM)
                    kseq = k_all[:,head,:]
                    vseq = v_all[:,head,:]
                    qseq = q_all[:,head,:]
                    for t in range(N-DECODE):
                        cache.append(kseq[t],vseq[t])
                    for t in range(N-DECODE,N):
                        cache.append(kseq[t],vseq[t])
                        q = qseq[t]
                        dense, scores = cache.dense(q)
                        oracle_pages, page_masses = mass_oracle(scores, cache.total_tokens)
                        values = np.concatenate([np.asarray(v) for v in cache.values])
                        hindsight_pages = hindsight_value_oracle(scores,values,dense,cache.vmax)
                        # Ideal exact-score per page upper bound: infeasible without
                        # all-key dot products; real method uses a geometric upper.
                        ideal_upper = np.array([scores[i:i+len(p)].max() for i,p in
                                                zip(range(0,cache.total_tokens,PAGE_SIZE),cache.keys)])
                        geometric_upper = cache.fresh_upper(q)
                        ideal = old.screen(q,cache,ideal_upper,tol=TOL)
                        real = old.screen(q,cache,geometric_upper,tol=TOL)
                        assert np.all(ideal_upper <= geometric_upper + 1e-8)
                        for v in (ideal,real):
                            err = float(np.linalg.norm(v['result']-dense)/cache.vmax)
                            if err > v['certificate'] + 1e-9:
                                raise AssertionError(f'output certificate failure: {err} > {v["certificate"]}')
                        if not (oracle_pages <= ideal['selected_pages'] <= real['selected_pages']):
                            # Greedy page-max screening is not guaranteed to be monotone
                            # in the number of pages relative to geometric bounds because
                            # each changes the traversal order. Record rather than assume.
                            # Oracle <= each is the only universal ordering required.
                            if oracle_pages > min(ideal['selected_pages'],real['selected_pages']):
                                raise AssertionError('Oracle lower-bound check failed')
                        n_pages = cache.num_pages
                        rows.append({
                            'seed':seed, 'regime':regime, 'layer':layer,'head':head,'query_index':t,
                            'tokens':cache.total_tokens,'pages':n_pages,
                            'oracle_mass_pages':oracle_pages,
                            'hindsight_value_pages':hindsight_pages,
                            'ideal_score_max_pages':ideal['selected_pages'],
                            'geometric_pages':real['selected_pages'],
                            'oracle_fraction':oracle_pages/n_pages,
                            'hindsight_value_fraction':hindsight_pages/n_pages,
                            'ideal_score_max_fraction':ideal['selected_pages']/n_pages,
                            'geometric_fraction':real['selected_pages']/n_pages,
                            'attention_entropy_bits':float(-np.sum(page_masses*np.log2(np.maximum(page_masses,1e-300)))),
                            'largest_page_mass':float(max(page_masses)),
                            'ideal_output_certificate':ideal['certificate'],
                            'geometric_output_certificate':real['certificate'],
                            'ideal_output_error':float(np.linalg.norm(ideal['result']-dense)/cache.vmax),
                            'geometric_output_error':float(np.linalg.norm(real['result']-dense)/cache.vmax)
                        })
    return rows


def constructed_controls() -> list[dict]:
    """Exact analytic oracle on 128 pages of 32 tokens.

    No vector embeddings are assumed here. The output is not a model trace.
    """
    n=128*32
    examples = [
       ('uniform',np.zeros(n)),
       ('one_high_scoring_page',np.concatenate([np.full(32,18.0),np.zeros(n-32)])),
       ('rare_high_scoring_single_token',np.concatenate([np.array([18.0]),np.zeros(n-1)])),
       ('medium_peak_page',np.concatenate([np.full(32,8.0),np.zeros(n-32)])),
    ]
    output=[]
    for name,scores in examples:
        cnt, masses = mass_oracle(scores,len(scores))
        output.append({'name':name,'tokens':n,'pages':n//PAGE_SIZE,
                       'oracle_mass_pages':cnt,'oracle_fraction':cnt/(n//PAGE_SIZE),
                       'largest_page_mass':float(max(masses))})
    return output


def aggregate(rows):
    n=len(rows)
    def s(key):return np.array([x[key] for x in rows])
    def by(key):
        return {regime:{'count':int(sum(r['regime']==regime for r in rows)),
                        'oracle_mass_median_fraction':float(np.median([r['oracle_fraction'] for r in rows if r['regime']==regime])),
                        'ideal_max_median_fraction':float(np.median([r['ideal_score_max_fraction'] for r in rows if r['regime']==regime])),
                        'geometric_median_fraction':float(np.median([r['geometric_fraction'] for r in rows if r['regime']==regime])),
                        'oracle_mass_full_read_pct':float(np.mean([r['oracle_mass_pages']==r['pages'] for r in rows if r['regime']==regime])*100),
                        'hindsight_value_median_fraction':float(np.median([r['hindsight_value_fraction'] for r in rows if r['regime']==regime])),
                        'real_geometric_full_read_pct':float(np.mean([r['geometric_pages']==r['pages'] for r in rows if r['regime']==regime])*100),
                        } for regime in REGIMES}
    return {
        'queries':n,
        'oracle_mass_median_fraction':float(np.median(s('oracle_fraction'))),
        'hindsight_value_median_fraction':float(np.median(s('hindsight_value_fraction'))),
        'hindsight_value_full_read_pct':float(np.mean(s('hindsight_value_pages')==s('pages'))*100),
        'oracle_mass_full_read_pct':float(np.mean(s('oracle_mass_pages')==s('pages'))*100),
        'ideal_score_max_median_fraction':float(np.median(s('ideal_score_max_fraction'))),
        'geometric_median_fraction':float(np.median(s('geometric_fraction'))),
        'geometric_full_read_pct':float(np.mean(s('geometric_pages')==s('pages'))*100),
        'ideal_score_max_full_read_pct':float(np.mean(s('ideal_score_max_pages')==s('pages'))*100),
        'max_geometric_error':float(np.max(s('geometric_output_error'))),
        'max_geometric_certificate':float(np.max(s('geometric_output_certificate'))),
        'by_regime':by('regime')
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/g4_oracle_headroom_audit.json')
    args=p.parse_args()
    rows=make_rows()
    r={'metadata':{
       'date':'2026-11-03','version':VERSION,'gate':'G4 exploratory oracle bound diagnostic',
       'warning':'UNTRAINED toy transformer, NOT pretrained, NOT physical memory access, NOT end-to-end speedup',
       'model':'2 layer 4 head RoPE causal transformer with RANDOM parameters',
       'N':N,'DECODE':DECODE,'page_size':PAGE_SIZE,'tol_normalized':TOL,
       'seeds':SEEDS,'regimes':REGIMES,'heads':EVAL_HEADS,
       'python':platform.python_version(),'numpy':np.__version__,
       'interpretation':'Oracle mass uses full QK and exact page masses; unavailable at reduced data-movement cost. Its page count is only a floor for 2m mass certificates, NOT all possible approximations. Hindsight_value uses full V as well, sorting pages by true mass and stopping at observed error <=tol; this is an infeasible heuristic, not a certified lower bound or optimal solution.',
       'timing':'no timing, PPA, model quality, energy or production inference measured',
       'statistical':'same seed and model weights used for many within-run correlated decode steps; descriptive statistics only'
    },'aggregate':aggregate(rows),'constructed_controls':constructed_controls(),'rows':rows}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(r,indent=2)+'\n')
    print('saved',output,'rows',len(rows))
    print(json.dumps(r['aggregate'],indent=2))
    print('constructed',json.dumps(r['constructed_controls'],indent=2))

if __name__=='__main__':
    main()