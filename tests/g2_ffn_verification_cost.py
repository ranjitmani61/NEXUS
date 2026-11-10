#!/usr/bin/env python3
"""G2 proxy: multi-query SwiGLU matrix costs, not pretrained inference or novelty.

Reproducible single-thread invocation:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 python tests/g2_ffn_verification_cost.py --output results/G2_ffn_verifier_cost.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np

# Published config shapes: see source URLs in accompanying contract; no weights used.
SHAPES = {"Qwen2.5-0.5B": (896, 4864), "Granite-4.0-micro-base": (2560, 8192)}
C2_ATTENTION_SHARES = {"Granite host B / 16K / b1": .7052066730549075,
                       "Granite host C / 16K / b1": .3797642644870673,
                       "Qwen host A / 32K / b4": .5861178531211508,
                       "Qwen host C / 32K / b1": .49043314352149003}
# MLP shares measured on C2 host C; absent for G5-R host A/B.
# https://github.com/ranjitmani61/NEXUS/tree/922dc8f3b5639f4d5c6297da8f698f9b4580a149/results/c2
C2_MLP_SHARES = {"Granite host C / 16K / b1": .2590455626976019,
                 "Qwen host C / 32K / b1": .14626117892390306}
N_PROPOSALS = (1, 2, 4, 8)
P_VALUES = (0.7, 0.8, 0.9, 0.95)
DRAFT_PER_PROPOSAL = .05
SEED = 20261110


def expected_length(p: float, k: int) -> float:
    if not (0 <= p <= 1) or k < 0:
        raise ValueError("invalid p or k")
    return sum(p**j for j in range(k + 1))


def scaled_share_of_nonattention(f: float, scaled_share:float|None)->float:
    # Default reproduces the explicitly pessimistic all-nonattention proxy.
    g=1-f if scaled_share is None else scaled_share
    if not (0<=f<=1 and 0<=g<=1-f):
        raise ValueError("invalid time-share decomposition")
    return g


def verifier_speed(p: float, k: int, f: float, rho: float, beta: float,
                   delta: float = DRAFT_PER_PROPOSAL, h: float = 0,
                   scaled_share:float|None = None) -> float:
    g=scaled_share_of_nonattention(f, scaled_share)
    if not (beta>=0 and rho>0 and delta>=0 and h>=0):
        raise ValueError("invalid cost")
    # If g is MLP only, the rest of target nonattention work is optimistically
    # treated as constant at one-token latency. Not a bound on actual hardware.
    return expected_length(p,k)/(1 + g*(rho-1)+f*beta*k+delta*k+h)


def beta_threshold(p:float,k:int,f:float,rho:float,
                   delta:float=DRAFT_PER_PROPOSAL,h:float=0,
                   scaled_share:float|None=None)->float:
    g=scaled_share_of_nonattention(f,scaled_share)
    if f<=0 or k<=0 or rho<=0 or delta<0 or h<0:
        raise ValueError("invalid threshold parameters")
    return (expected_length(p,k)/2 - 1 - g*(rho-1) - delta*k-h)/(f*k)


def rho_budget_for_2x(p:float,k:int,f:float,beta:float=0.0,
                      delta:float=DRAFT_PER_PROPOSAL,h:float=0.0,
                      scaled_share:float|None=None)->float:
    g=scaled_share_of_nonattention(f,scaled_share)
    if g<=0 or k<=0 or beta<0 or delta<0 or h<0:
        raise ValueError("need positive scaled time share and k")
    return 1 + (expected_length(p,k)/2 - 1-f*beta*k-delta*k-h)/g

def swiglu_serial(x:np.ndarray, wg:np.ndarray, wu:np.ndarray, wd:np.ndarray) -> np.ndarray:
    out=[]
    for row in x:
        gate=row @ wg
        up=row @ wu
        silu=gate/(1+np.exp(-gate))
        out.append((silu*up) @ wd)
    return np.stack(out,axis=0)


def swiglu_batch(x:np.ndarray, wg:np.ndarray, wu:np.ndarray, wd:np.ndarray) -> np.ndarray:
    gate=x @ wg
    up=x @ wu
    return ((gate/(1+np.exp(-gate)))*up) @ wd


def q(vals, percentile):
    return float(np.quantile(vals,percentile))


def timed(fn, *, repeats:int, warmup:int):
    for _ in range(warmup):
        fn()
    durations=[]
    for _ in range(repeats):
        t=time.perf_counter_ns(); fn(); durations.append((time.perf_counter_ns()-t)/1e6)
    return durations


def measure_shape(label:str, d:int, h:int, *, seed:int=SEED, repeats:int=9, warmup:int=2):
    rng=np.random.default_rng(seed+d)
    # Qwen ~52MB, Granite ~252MB float32; synthetic weights only.
    wg=rng.standard_normal((d,h),dtype=np.float32)*(0.08/np.sqrt(d))
    wu=rng.standard_normal((d,h),dtype=np.float32)*(0.08/np.sqrt(d))
    wd=rng.standard_normal((h,d),dtype=np.float32)*(0.08/np.sqrt(h))
    x=rng.standard_normal((max(N_PROPOSALS)+1,d),dtype=np.float32)
    cases=[]
    for k in N_PROPOSALS:
        n=k+1; xx=x[:n]
        a=swiglu_serial(xx,wg,wu,wd); b=swiglu_batch(xx,wg,wu,wd)
        error=float(np.max(np.abs(a-b))); tol=1e-5
        if not np.allclose(a,b,rtol=1e-5,atol=tol):
            raise AssertionError(f"mismatch {label}, k={k}, max error={error}")
        # The single-token baseline uses exactly the same row and weights.
        sample=lambda:swiglu_serial(xx[:1],wg,wu,wd)
        batched=lambda:swiglu_batch(xx,wg,wu,wd)
        serial=lambda:swiglu_serial(xx,wg,wu,wd)
        # Interleave sample and batch to reduce drift confounding.
        for _ in range(warmup):sample();batched();serial()
        single_ms=[]; batched_ms=[]; serial_ms=[]
        for rep in range(repeats):
            funcs=[("single",sample),("batch",batched),("serial",serial)]
            if rep%2: funcs.reverse()
            times={}
            for key,func in funcs:
                st=time.perf_counter_ns();func();times[key]=(time.perf_counter_ns()-st)/1e6
            single_ms.append(times["single"]);batched_ms.append(times["batch"]);serial_ms.append(times["serial"])
        # Ratio of medians, bootstrap-style no false statistical inference.
        rho=statistics.median(batched_ms)/statistics.median(single_ms)
        cases.append({"k":k,"n_queries":n,"weights_bytes":int(3*d*h*4),
                      "nonattention_dense_flops":int(6*d*h*n),
                      "weight_operational_intensity_flops_per_byte_ideal":float(6*d*h*n/(3*d*h*4)),
                      "max_abs_output_error":error,
                      "single_ms":single_ms,"batch_ms":batched_ms,"serial_ms":serial_ms,
                      "single_median_ms":statistics.median(single_ms),"batch_median_ms":statistics.median(batched_ms),
                      "serial_median_ms":statistics.median(serial_ms),
                      "rho_batch_vs_single":rho,
                      "batch_vs_serial_time":statistics.median(batched_ms)/statistics.median(serial_ms),
                      "rho_paired_p10_p90":[q(np.array(batched_ms)/np.array(single_ms),.1),q(np.array(batched_ms)/np.array(single_ms),.9)]})
    return {"label":label,"hidden":d,"intermediate":h,"cases":cases}


def frontier(measured):
    rows=[]
    shape_by_name={x["label"]:x for x in measured}
    for host, f in C2_ATTENTION_SHARES.items():
        shape=shape_by_name["Granite-4.0-micro-base" if "Granite" in host else "Qwen2.5-0.5B"]
        g=C2_MLP_SHARES.get(host)
        for c in shape["cases"]:
            k=c["k"]
            for p in P_VALUES:
                rho=c["rho_batch_vs_single"]
                all_nonav=beta_threshold(p,k,f,rho)
                # Old optimistic (rho=1) and pessimistic (all nonattention
                # scales as FFN). Only host C supports measured-MLP-share
                # sensitivity; it remains a cross-host extrapolation.
                row={"host_anchor":host,"measured_attention_share_external":f,
                     "measured_mlp_share_external":g,"k":k,"p_assumed":p,
                     "rho_ffn_microbenchmark":rho,
                     "beta_threshold_old_rho1":beta_threshold(p,k,f,1),
                     "beta_threshold_if_all_nonattention_scaled":all_nonav,
                     "speed_beta0_all_nonattention_scaled":verifier_speed(p,k,f,rho,0),
                     "reachable_at_beta0_all_nonattention_scaled":verifier_speed(p,k,f,rho,0)>=2}
                if g is not None:
                    row.update({"beta_threshold_if_only_mlp_scaled":beta_threshold(p,k,f,rho,scaled_share=g),
                                "speed_beta0_only_mlp_scaled":verifier_speed(p,k,f,rho,0,scaled_share=g),
                                "speed_beta025_only_mlp_scaled":verifier_speed(p,k,f,rho,.25,scaled_share=g),
                                "reachable_at_beta0_only_mlp_scaled":verifier_speed(p,k,f,rho,0,scaled_share=g)>=2})
                else:
                    row.update({"beta_threshold_if_only_mlp_scaled":None,
                                "speed_beta0_only_mlp_scaled":None,
                                "speed_beta025_only_mlp_scaled":None,
                                "reachable_at_beta0_only_mlp_scaled":None})
                rows.append(row)
    return rows

def run(repeats=9,warmup=2, seed=SEED):
    models=[measure_shape(name,*dims,seed=seed,repeats=repeats,warmup=warmup) for name,dims in SHAPES.items()]
    rows=frontier(models)
    return {"version":"1.0","date":"2026-11-10","kind":"synthetic shape proxy, NOT full model",
            "assumed_seed":seed,"repeats":repeats,"warmup":warmup,"dtype":"float32",
            "env":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform(),
                   "openblas_threads":os.environ.get("OPENBLAS_NUM_THREADS"),"cpu_count":os.cpu_count()},
            "models":models,"frontier":rows,"matrix_cells":len(rows),
            "limitations":["No real weights, no target verifier, no drafter, no causal attention, no actual acceptance.",
                           "All f anchors from different CPUs/implementations; projected speedups are NOT measured.",
                           "FFN is only a proxy for nonattention, not a lower bound on wallclock time.",
                           "Only MLP-scaled sensitivity charges rho to the observed MLP share on C2 host C; remaining components are optimistically constant.",
                           "All-nonattention-scaled sensitivity is not a measured end-to-end bound and can overstate costs.",
                           "Granite full model may differ in kernel mix; GEMM layout/cache effects dominate."]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--repeats',type=int,default=9);parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    if args.repeats<3 or args.warmup<0:parser.error('repeats>=3, warmup>=0')
    res=run(repeats=args.repeats,warmup=args.warmup,seed=args.seed)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    for m in res['models']:
        print(m['label'],[(c['k'],round(c['rho_batch_vs_single'],3),round(c['batch_vs_serial_time'],3)) for c in m['cases']])
    print(f"{res['matrix_cells']} conditional frontier rows saved: {path}")

if __name__=='__main__': main()
