#!/usr/bin/env python3
"""POST-HOC 2026-11-10 ablation: K-contiguous score GEMM vs Q-contiguous score GEMM.

Not pre-registered. Does NOT replace original G2 primary benchmark or measure GPU.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np
from g2_joint_verification_frontier import SEED, serial_attention, fused_attention, _softmax_fp32


def kv_stationary(q, k, v):
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or k.shape != v.shape or q.shape[1] != k.shape[1]:
        raise ValueError("invalid Q,K,V shapes")
    return _softmax_fp32((k @ q.T).T * (q.shape[1] ** -0.5)) @ v


def run(repeats: int = 13, seed: int = SEED):
    rng=np.random.default_rng(seed)
    rows=[]
    for t in (4096,8192,16384):
        k=rng.standard_normal((t,64),dtype=np.float32)
        v=rng.standard_normal((t,64),dtype=np.float32)
        q8=rng.standard_normal((9,64),dtype=np.float32)
        single_times=[]
        for _ in range(5): serial_attention(q8[:1],k,v)
        for _ in range(repeats):
            start=time.perf_counter_ns()
            serial_attention(q8[:1],k,v)
            single_times.append((time.perf_counter_ns()-start)/1e6)
        median_single=statistics.median(single_times)
        for n in (2,3,5,9):
            q=q8[:n]
            reference=serial_attention(q,k,v)
            out_q=fused_attention(q,k,v)
            out_k=kv_stationary(q,k,v)
            for out in (out_q,out_k):
                if not np.allclose(reference,out,rtol=3e-4,atol=3e-4):
                    raise AssertionError("numerical mismatch")
            durations={"serial":[],"q_stationary":[],"kv_stationary":[]}
            funcs={"serial":lambda:serial_attention(q,k,v),
                   "q_stationary":lambda:fused_attention(q,k,v),
                   "kv_stationary":lambda:kv_stationary(q,k,v)}
            for f in funcs.values():
                for _ in range(5):f()
            for rep in range(repeats):
                names=list(funcs)
                names=names[rep%3:]+names[:rep%3]
                for name in names:
                    start=time.perf_counter_ns()
                    funcs[name]()
                    durations[name].append((time.perf_counter_ns()-start)/1e6)
            med={name:statistics.median(xs) for name,xs in durations.items()}
            rows.append({"T":t,"n":n,"k_equivalent_proposals":n-1,"single_query_median_ms":median_single,"median_ms":med,
                         "q_beta":(med["q_stationary"]/median_single-1)/(n-1),
                         "kv_beta":(med["kv_stationary"]/median_single-1)/(n-1),
                         "q_vs_serial":med["q_stationary"]/med["serial"],
                         "kv_vs_serial":med["kv_stationary"]/med["serial"],
                         "q_maxabs":float(np.max(np.abs(out_q-reference))),
                         "kv_maxabs":float(np.max(np.abs(out_k-reference)))})
    return {"method":"explicitly post-hoc KV-stationary alternative (not selected in pre-registration)",
            "environment":{"python":platform.python_version(),"numpy":np.__version__,
                           "machine":platform.machine(),"OPENBLAS_NUM_THREADS":os.getenv("OPENBLAS_NUM_THREADS")},
            "seed":seed,"repeats":repeats,"rows":rows}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    data=run()
    path=Path(args.output);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"rows":len(data["rows"]),"max_error":max(max(r['q_maxabs'],r['kv_maxabs']) for r in data['rows'])}))
