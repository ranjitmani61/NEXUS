#!/usr/bin/env python3
"""G3 snapshot coherency experiment. NOT production concurrency safety or hardware proof.

This module deliberately preserves the old PageCache to reproduce its failure,
then tests an immutable-copy, unique-version snapshot API. Every update copies
all K/V and rebuilds metadata: intentionally correctness-first, not optimized.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import itertools
import json
import math
import platform
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_guarded_mixed_precision_audit import (PageCache, PageMetadata, dense,
    guarded_admission, query, FORMATS)

_ids = itertools.count(1)

def frozen_copy(array, dtype=None):
    array = np.asarray(array, dtype=dtype)
    out = np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)
    assert not out.flags.writeable
    return out

@dataclass(frozen=True)
class FrozenSnapshot:
    K: np.ndarray
    V: np.ndarray
    K64: np.ndarray
    size: int
    pages: int
    d: int
    version: int

    @staticmethod
    def from_data(K, V, page_size):
        K = np.asarray(K)
        V = np.asarray(V)
        if not isinstance(page_size, int) or page_size <= 0:
            raise ValueError('page_size must be a positive integer')
        if K.ndim != 2 or V.ndim != 2 or K.shape[0] == 0 or K.shape[0] != V.shape[0] or K.shape[1] == 0 or K.shape[0]%page_size:
            raise ValueError('invalid key/value shapes')
        if not np.issubdtype(K.dtype, np.floating):
            # bf16 is a special dtype that does not register np.floating
            if K.dtype not in FORMATS.values(): raise ValueError('unsupported key dtype')
        if not np.all(np.isfinite(K.astype('float64'))) or not np.all(np.isfinite(V)):
            raise ValueError('nonfinite keys or values')
        k = frozen_copy(K)
        v = frozen_copy(V, dtype=np.float64)
        return FrozenSnapshot(k,v,frozen_copy(k.astype('float64')),page_size,len(k)//page_size,k.shape[1],next(_ids))

    def replace_page(self, page, K, V):
        if not isinstance(page,int) or not 0 <= page < self.pages: raise ValueError('invalid page')
        K = np.asarray(K,dtype=self.K.dtype); V = np.asarray(V,dtype='float64')
        if K.shape != (self.size,self.d) or V.shape != (self.size,self.V.shape[1]):
            raise ValueError('replacement shapes differ')
        nk = np.array(self.K,copy=True); nv = np.array(self.V,copy=True)
        sl = slice(page*self.size,(page+1)*self.size)
        nk[sl] = K; nv[sl] = V
        return FrozenSnapshot.from_data(nk,nv,self.size)

    def append_page(self, K, V):
        K = np.asarray(K,dtype=self.K.dtype); V = np.asarray(V,dtype='float64')
        if K.shape != (self.size,self.d) or V.shape != (self.size,self.V.shape[1]):
            raise ValueError('append shapes differ')
        return FrozenSnapshot.from_data(np.concatenate((self.K,K)),np.concatenate((self.V,V)),self.size)

@dataclass(frozen=True)
class FrozenMetadata:
    lo: np.ndarray
    hi: np.ndarray
    version: int

def build_meta(snapshot):
    meta = PageMetadata(snapshot)
    return FrozenMetadata(frozen_copy(meta.lo),frozen_copy(meta.hi),meta.version)

def safe_snapshot_query(snapshot, metadata, q, *, eps=0.02):
    """Validate the stored-Q contract before invoking the preexisting screening path.

    Invalid inputs fail closed. This is not a formal concurrency or FP proof.
    """
    if not isinstance(snapshot, FrozenSnapshot) or not isinstance(metadata, FrozenMetadata):
        raise TypeError('trusted snapshot and metadata types required')
    q = np.asarray(q)
    if q.ndim != 1 or q.shape != (snapshot.d,) or q.dtype != snapshot.K.dtype:
        raise ValueError('query must have the stored key dtype and exact 1D shape')
    if not np.all(np.isfinite(q.astype('float64'))):
        raise ValueError('nonfinite query is not a valid attention input')
    return query(snapshot,metadata,q,eps=eps)

def witness_old_mutable_api(fmt):
    d, page_size = 16,4
    K = np.vstack((np.full((page_size,d),5),np.full((page_size,d),-5))).astype(FORMATS[fmt])
    V = np.vstack((np.zeros((page_size,1)),np.ones((page_size,1))))
    q = np.ones(d,dtype=FORMATS[fmt])
    cache = PageCache(K,V,page_size);meta = PageMetadata(cache)
    before,_,_,_ = query(cache,meta,q)
    cache.K[page_size:] = 20
    cache.K64[page_size:] = 20 # prior code has a separate mutable FP64 copy
    approximate,yes,bound,reason = query(cache,meta,q)
    exact = dense(cache,q)
    error = float(np.linalg.norm(approximate-exact) / np.max(np.linalg.norm(cache.V,axis=1)))
    return dict(format=fmt,admitted=bool(yes),reported_bound=float(bound),actual_normalized_error=error,
                mutation_without_version_change=True,reason=reason,version=int(cache.version),
                before_output=before.tolist(),approximation=approximate.tolist(),reference=exact.tolist())

def check_snapshot(fmt):
    d, page_size = 16,4
    src = np.vstack((np.full((page_size,d),5),np.full((page_size,d),-5))).astype(FORMATS[fmt])
    src_v = np.vstack((np.zeros((page_size,1)),np.ones((page_size,1))))
    q = np.ones(d,dtype=FORMATS[fmt]); s = FrozenSnapshot.from_data(src,src_v,page_size)
    meta = build_meta(s)
    assert not s.K.flags.writeable and not s.K64.flags.writeable and not s.V.flags.writeable
    assert not meta.lo.flags.writeable and not meta.hi.flags.writeable
    for obj in (s.K,s.K64,s.V,meta.lo,meta.hi):
        try:
            obj.flags.writeable = True
            raise AssertionError('immutable bytes backing unexpectedly made writeable')
        except ValueError: pass
    src[:] = 99; src_v[:] = 99 # input aliases must not affect sealed snapshot
    assert np.all(s.K[:page_size].astype('float64')==5)
    out,yes,bound,_=safe_snapshot_query(s,meta,q)
    assert yes and np.linalg.norm(out-dense(s,q)) <= bound + 1e-12
    s2 = s.replace_page(1, np.full((page_size,d),20),np.ones((page_size,1)))
    assert s2.version != s.version
    oldmeta_admit,_,_,reason=guarded_admission(s2,meta,q)
    assert not oldmeta_admit and reason == 'stale_version'
    out_old,_,_,_ = safe_snapshot_query(s,meta,q)
    assert np.linalg.norm(out_old-dense(s,q)) < 1e-12
    meta2 = build_meta(s2)
    out2,yes2,bound2,_=safe_snapshot_query(s2,meta2,q)
    err2=float(np.linalg.norm(out2-dense(s2,q)))
    assert err2<=0.02+1e-12
    assert (not yes2) or err2<=bound2 +1e-12
    s3 = s2.append_page(np.full((page_size,d),-1), np.full((page_size,1),0.5))
    assert s3.pages==3 and s3.version not in (s.version,s2.version)
    assert guarded_admission(s3,meta2,q)[3]=='stale_version'
    return dict(format=fmt,old_version=s.version,new_version=s2.version,appended_version=s3.version,
                stale_metadata_rejected=True,actual_post_update_error=err2,
                new_metadata_admitted=bool(yes2),new_metadata_bound=(float(bound2) if bound2 is not None else None))

def stress(seed,fmt,updates=64):
    rng=np.random.default_rng(seed);size=8;d=32;pages=16
    K=rng.normal(size=(pages*size,d)).astype(FORMATS[fmt]);V=rng.normal(size=(pages*size,8))
    q=rng.normal(size=d).astype(FORMATS[fmt]);s=FrozenSnapshot.from_data(K,V,size)
    admissions=0; maxerr=0.; stale_rejects=0
    for i in range(updates):
        m=build_meta(s)
        out,yes,bound,_ = safe_snapshot_query(s,m,q)
        error=float(np.linalg.norm(out-dense(s,q))/max(1e-30,np.max(np.linalg.norm(s.V,axis=1))))
        assert error<=0.02+1e-10
        assert not yes or error<=bound+1e-10
        maxerr=max(maxerr,error); admissions+=int(yes)
        page=i%pages
        newK=rng.normal(size=(size,d)).astype(FORMATS[fmt]);newV=rng.normal(size=(size,8))
        nxt=s.replace_page(page,newK,newV)
        assert guarded_admission(nxt,m,q)[3]=='stale_version'
        stale_rejects+=1; s=nxt
    return dict(format=fmt,seed=seed,updates=updates,stale_rejects=stale_rejects,admissions=admissions,max_error=maxerr)

def copy_cost(seed=77,rounds=12):
    rng=np.random.default_rng(seed);size=16;d=64;pages=64
    K=rng.normal(size=(pages*size,d)).astype('float16');V=rng.normal(size=(pages*size,16))
    s=FrozenSnapshot.from_data(K,V,size)
    updK=np.ones((size,d),dtype='float16');updV=np.ones((size,16))
    durations=[]
    for _ in range(rounds):
        t=time.perf_counter_ns()
        s2=s.replace_page(0,updK,updV);m=build_meta(s2)
        durations.append((time.perf_counter_ns()-t)/1e6)
        s=s2
    return dict(rounds=rounds,pages=pages,page_size=size,dimension=d,
                median_copy_and_rebuild_ms=float(np.median(durations)),
                first_ms=float(durations[0]),input_bytes=int(K.nbytes+V.nbytes),
                qualification='local NumPy CPU; full memory copy + metadata rebuild, not optimized')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/g3_snapshot_consistency_audit.json');args=ap.parse_args()
    report=dict(version='1.0',date='2026-11-03',gate='G3 IN PROGRESS',
                notes='Untracked mutation safety counterexample; snapshot fix is functional but O(N) copy and no concurrent hardware verification.',
                python=platform.python_version(),numpy=np.__version__,
                witnesses=[witness_old_mutable_api(x) for x in FORMATS],
                snapshots=[check_snapshot(x) for x in FORMATS],
                stresses=[stress(s,f) for f in FORMATS for s in (11,29,43)],
                copy_cost=copy_cost())
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
