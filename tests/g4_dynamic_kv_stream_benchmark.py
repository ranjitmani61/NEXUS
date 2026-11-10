#!/usr/bin/env python3
"""G4: streaming KV cache + conservative metadata tests, v1.3.

Experimental diagnostic, NOT novel, NOT a real LLM benchmark or accelerator model.
Full reference attention is exact float64 NumPy; one-page screening bound is
sufficient in real arithmetic, only numerically verified in float64.
"""
from __future__ import annotations
import argparse, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np

PAGE=32
DIM=64
INITIAL=4096
APPENDS=128
TOL=.02
SEEDS=(1029,1031)
REGIMES=('clustered','random','uniform','needle')
REPEATS=3


def now_ms(): return time.perf_counter_ns()/1e6

def median_timing(func,repeats=REPEATS):
    values=[]; result=None
    for _ in range(repeats):
        t=now_ms(); result=func(); values.append(now_ms()-t)
    return float(np.median(values)), result

class Cache:
    def __init__(self, keys, values, capacity):
        assert keys.shape == values.shape and keys.ndim == 2
        self.d=keys.shape[1]; self.n=keys.shape[0]; self.version=0
        self.keys=np.empty((capacity,self.d),dtype=np.float64)
        self.values=np.empty((capacity,self.d),dtype=np.float64)
        self.keys[:self.n]=keys
        self.values[:self.n]=values
    def append(self,k,v):
        if self.n>=len(self.keys): raise ValueError('cache full')
        self.keys[self.n]=k; self.values[self.n]=v
        self.n+=1; self.version+=1

class Metadata:
    def __init__(self,cache,mode):
        if mode not in ('fixed','box'): raise ValueError(mode)
        self.mode=mode; self.d=cache.d
        capacity=(len(cache.keys)+PAGE-1)//PAGE
        self.centers=np.zeros((capacity,self.d),dtype=np.float64)
        self.radius=np.zeros(capacity,dtype=np.float64)
        self.minimum=np.zeros((capacity,self.d),dtype=np.float64)
        self.maximum=np.zeros((capacity,self.d),dtype=np.float64)
        self.count=np.zeros(capacity,dtype=np.int64)
        self.n=0; self.version=0
        # Vectorized bulk construction; charge only work necessary for this mode.
        # A prior baseline used a Python per-token initial scan; that overstated costs.
        full=cache.n//PAGE
        if full:
            pages=cache.keys[:full*PAGE].reshape(full,PAGE,self.d)
            self.count[:full]=PAGE
            if self.mode=='fixed':
                self.centers[:full]=pages[:,0,:]
                self.radius[:full]=np.linalg.norm(pages-pages[:,0:1,:],axis=-1).max(axis=1)
            else:
                self.minimum[:full]=pages.min(axis=1)
                self.maximum[:full]=pages.max(axis=1)
            self.n=full*PAGE
        for idx in range(full*PAGE,cache.n): self._ingest(cache.keys[idx])
        self.version=cache.version
    def _ingest(self,k):
        page=self.n//PAGE
        if self.count[page]==0:
            if self.mode=='fixed': self.centers[page]=k
            else:
                self.minimum[page]=k
                self.maximum[page]=k
        elif self.mode=='fixed':
            self.radius[page]=max(self.radius[page],float(np.linalg.norm(k-self.centers[page])))
        else:
            self.minimum[page]=np.minimum(self.minimum[page],k)
            self.maximum[page]=np.maximum(self.maximum[page],k)
        self.count[page]+=1; self.n+=1
    def append(self,cache):
        if self.n!=cache.n-1 or cache.version!=self.version+1:
            raise ValueError('out-of-order cache update')
        self._ingest(cache.keys[cache.n-1]); self.version=cache.version
    def boundaries(self,q):
        npages=(self.n+PAGE-1)//PAGE
        if self.mode=='fixed':
            center=self.centers[:npages] @ q
            fuzz=np.linalg.norm(q)*self.radius[:npages]
            low=(center-fuzz)/math.sqrt(self.d)
            high=(center+fuzz)/math.sqrt(self.d)
        else:
            product_a=self.minimum[:npages]*q
            product_b=self.maximum[:npages]*q
            low=np.minimum(product_a,product_b).sum(axis=1)/math.sqrt(self.d)
            high=np.maximum(product_a,product_b).sum(axis=1)/math.sqrt(self.d)
        return low,high,self.count[:npages]


def dense(q,cache):
    score=cache.keys[:cache.n]@q/math.sqrt(cache.d)
    w=np.exp(score-np.max(score))
    return (w@cache.values[:cache.n])/w.sum()


def certified_one_page(q,cache,meta,tol=TOL):
    if not np.isfinite(q).all(): raise ValueError('non-finite query')
    if meta.n!=cache.n or meta.version!=cache.version:
        return dense(q,cache),cache.n//PAGE+(cache.n%PAGE!=0),'stale_dense_fallback',0.0
    low,high,count=meta.boundaries(q)
    if (count<=0).any() or count.sum()!=cache.n:
        raise ValueError('corrupt page counts')
    best=int(np.argmax(low+np.log(count)))
    sh=float(np.max(high))
    upper_mass=count*np.exp(high-sh)
    omitted=float(np.sum(np.delete(upper_mass,best)))
    lower_mass=float(count[best]*np.exp(low[best]-sh))
    cert=float(2*omitted/(omitted+lower_mass)) if omitted+lower_mass else 2.
    if cert<=tol:
        start=best*PAGE; end=start+int(count[best]);
        ks=cache.keys[start:end]; vs=cache.values[start:end]
        sc=ks@q/math.sqrt(cache.d)
        w=np.exp(sc-sc.max())
        return (w@vs)/w.sum(),1,'certified_one_page',cert
    return dense(q,cache),len(count),'bound_dense_fallback',0.0


def scenario(seed,kind):
    rng=np.random.default_rng(seed)
    q0=rng.normal(size=DIM)
    alln=INITIAL+APPENDS
    if kind=='clustered':
        per_page_center=rng.normal(size=(math.ceil(alln/PAGE),DIM))
        per_page_center[11]=1.5*q0+rng.normal(0,.15,size=DIM)
        keys=np.repeat(per_page_center,PAGE,axis=0)[:alln]+rng.normal(0,.04,size=(alln,DIM))
    elif kind=='random':keys=rng.normal(size=(alln,DIM))
    elif kind=='uniform':keys=np.zeros((alln,DIM))
    elif kind=='needle':
        keys=rng.normal(size=(alln,DIM))
        keys[-1]=2*q0
    else:raise ValueError(kind)
    values=rng.normal(size=(alln,DIM))
    queries=q0[None,:]+rng.normal(0,.025,size=(APPENDS,DIM))
    return np.ascontiguousarray(keys),np.ascontiguousarray(values),queries


def run():
    rows=[]; builds=[]
    max_abs_err=0.
    for regime in REGIMES:
        for seed in SEEDS:
            k,v,qs=scenario(seed,regime)
            cache=Cache(k[:INITIAL],v[:INITIAL],len(k))
            metas={}; builds_by_mode={}
            for mode in ('fixed','box'):
                build_ms, m=median_timing(lambda:Metadata(cache,mode),3)
                metas[mode]=m
                builds_by_mode[mode]=build_ms
                builds.append({'regime':regime,'seed':seed,'mode':mode,'initial_build_ms':build_ms})
            vnorm=np.linalg.norm(v,axis=1) # offline verification; not online metadata
            current_vmax=float(vnorm[:INITIAL].max())
            for step,q in enumerate(qs):
                cache.append(k[INITIAL+step],v[INITIAL+step])
                current_vmax=max(current_vmax,float(vnorm[INITIAL+step]))
                u_times={}
                for mode, m in metas.items():
                    t=now_ms(); m.append(cache); u_times[mode]=now_ms()-t
                # Interleave order to avoid only candidate benefiting from one order.
                if step%2==0:
                    dense_ms, truth=median_timing(lambda:dense(q,cache))
                out_by_mode={}
                for mode,m in metas.items():
                    timing,res=median_timing(lambda:certified_one_page(q,cache,m))
                    out_by_mode[mode]=(timing,res)
                if step%2==1:
                    dense_ms,truth=median_timing(lambda:dense(q,cache))
                for mode,m in metas.items():
                    # Check that every actual key score is between its page bounds.
                    low,high,c=m.boundaries(q)
                    for page in range(len(c)):
                        scores=cache.keys[page*PAGE:page*PAGE+int(c[page])]@q/math.sqrt(DIM)
                        if min(scores) < low[page]-5e-12 or max(scores)>high[page]+5e-12:
                            raise AssertionError(f'bound violation {regime} {seed} {step} {mode} {page}')
                    hy_ms,(approx,pages,status,cert)=out_by_mode[mode]
                    error=float(np.linalg.norm(approx-truth)/current_vmax)
                    max_abs_err=max(max_abs_err,error)
                    if error>cert+2e-10 or cert>TOL+2e-10:
                        raise AssertionError(f'certificate violation {regime} {seed} {step} {mode}: {error} {cert}')
                    rows.append({'regime':regime,'seed':seed,'step':step,'tokens':cache.n,
                        'mode':mode,'pages_read':pages,'admitted':status=='certified_one_page',
                        'status':status,'certificate':cert,'true_normalized_error':error,
                        'metadata_update_ms':u_times[mode],
                        'candidate_ms':hy_ms,'dense_ms':dense_ms})
    summary=[]
    for regime in REGIMES:
        for mode in ('fixed','box'):
            r=[x for x in rows if x['regime']==regime and x['mode']==mode]
            c=np.array([x['candidate_ms'] for x in r]); d=np.array([x['dense_ms'] for x in r]);
            u=np.array([x['metadata_update_ms'] for x in r])
            b=[x['initial_build_ms'] for x in builds if x['regime']==regime and x['mode']==mode]
            summary.append({'regime':regime,'mode':mode,'queries':len(r),
                'admission_rate':float(np.mean([x['admitted'] for x in r])),
                'p50_candidate_ms':float(np.median(c)),
                'p50_dense_ms':float(np.median(d)),
                'p95_candidate_ms':float(np.percentile(c,95)),
                'p95_dense_ms':float(np.percentile(d,95)),
                'median_candidate_over_dense':float(np.median(c/d)),
                'median_update_us':float(np.median(u)*1000),
                'total_latency_ratio_including_metadata_update':float((c.sum()+u.sum())/d.sum()),
                'total_latency_ratio_including_initial_build':float((c.sum()+u.sum()+sum(b))/d.sum()),
                'max_normalized_error':float(max(x['true_normalized_error'] for x in r))})
    return {'metadata':{'date':'2026-11-03','version':'1.3','gate':'G4','dtype':'float64',
        'kind':'measured NumPy CPU, synthetic dynamic KV cache; not pretrained/hardware',
        'initial_tokens':INITIAL,'appended_tokens':APPENDS,'page_size':PAGE,'dimension':DIM,
        'seeds':list(SEEDS),'regimes':REGIMES,'tolerance':TOL,'timing_repeats':REPEATS,
        'python':platform.python_version(),'numpy':np.__version__,
        'blas_threads':{x:os.getenv(x) for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'source':'QUEST 2024 already uses online page min/max, so novelty not claimed',
        'limitations':['CPU NumPy, not optimized baseline or accelerator','synthetic q/K/V',
            'float64 empirical checks are not a finite-precision proof',
            'vectorized initial build + incremental metadata overhead separately charged',
            'full K/V append memory store common to both paths, not timed',
            'offline dense correctness oracle and V norm scan excluded from candidate timing',
            'timing noise, cache warmup, Python overhead, and page selection control flow',
            'no measure of energy, DRAM bytes, memory bus contention or real inference']},
        'summary':summary,'max_observed_normalized_error':max_abs_err,'rows':rows,'builds':builds}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/g4_dynamic_kv_stream.json'))
    a=parser.parse_args()
    output=run()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['summary'],indent=2))
    print('saved',a.output.resolve())

if __name__=='__main__':main()
