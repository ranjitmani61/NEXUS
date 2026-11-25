#!/usr/bin/env python3
"""G5 diagnostic attention-trace evaluator (version 1.0, 2026-11-03).

Inputs are real attention probabilities (or explicitly synthetic test fixtures) and
per-token values. This is an *offline oracle diagnostic* and cannot establish
achievable KV traffic reduction or latency: the probabilities require computing
full attention and the values require reading the full V cache.

Equation: ||output_subset - output_dense|| / max_i ||v_i||_2 <= tol.
Exact-real arithmetic certificates are evaluated numerically in float64.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np

VERSION = "1.0"
DATE = "2026-11-03"


def max_box_fraction(upper_masses: np.ndarray, distances: np.ndarray, z: float) -> float:
    """Upper bound max_{0<=u<=U} (u @ distances)/(z+sum(u)); real arithmetic.

    The root t solves sum_j U_j max(d_j-t,0) = t*z.
    Raises on invalid inputs; bisection computed in float64, not a formal FP proof.
    """
    u = np.asarray(upper_masses, dtype=np.float64)
    d = np.asarray(distances, dtype=np.float64)
    if u.shape != d.shape or u.ndim != 1:
        raise ValueError("U and d must be vectors with matching shape")
    if not (np.isfinite(z) and z > 0):
        raise ValueError("Selected probability mass must be > 0")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(d)) or np.any(u < 0) or np.any(d < 0):
        raise ValueError("Bounds and distances must be finite and nonnegative")
    if not u.size or np.max(d) == 0:
        return 0.0
    lo, hi = 0.0, float(np.max(d))
    for _ in range(64):
        t = 0.5 * (hi + lo)
        if float(u @ np.maximum(d-t, 0.0)) > t * z:
            lo = t
        else:
            hi = t
    return float(np.nextafter(hi, np.inf))


def evaluate_one(weights: np.ndarray, values: np.ndarray, page_size: int = 32,
                 tolerance: float = .02) -> dict:
    """Compare exact-attention offline mass, exact-score value and hindsight tests.

    Returns descriptive page counts and certificate checks, not hardware speedups.
    A page can be incomplete (causal prefix). One query/head at a time.
    """
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if w.ndim != 1 or v.ndim != 2 or v.shape[0] != w.size or w.size == 0:
        raise ValueError("expected weights [tokens] and values [tokens, dim]")
    if not (isinstance(page_size, int) and page_size > 0):
        raise ValueError("page_size must be a positive integer")
    if not (0 < tolerance < 1):
        raise ValueError("tolerance must be in (0,1)")
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(v)) or np.any(w < 0):
        raise ValueError("attention and values must be finite; weights nonnegative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("attention has zero mass")
    w = w / total
    n = w.size
    vmax = float(np.linalg.norm(v, axis=1).max())
    scale = vmax if vmax > 0 else 1.0
    dense = w @ v
    indices = [np.arange(i, min(i+page_size, n)) for i in range(0, n, page_size)]
    num_pages = len(indices)
    page_mass = np.array([w[ix].sum() for ix in indices])
    page_max = np.array([w[ix].max() for ix in indices])
    page_sum_bound = np.array([len(ix) * page_max[j] for j, ix in enumerate(indices)])
    centers = np.array([v[ix].mean(axis=0) for ix in indices])
    radii = np.array([np.linalg.norm(v[ix]-centers[j],axis=1).max()
                      for j, ix in enumerate(indices)])

    # Global oracle: sorting pages by *actual exact mass*, impossible to do cheaply.
    mass_order = np.argsort(-page_mass, kind="stable")
    cum = np.cumsum(page_mass[mass_order])
    mass_needed = int(np.searchsorted(cum, 1.0-tolerance/2.0, side="left")+1)
    mass_needed = min(mass_needed, num_pages)
    mass_read_mass = float(cum[mass_needed-1])
    mass_error_certificate = min(1.0, 2.0*(1.0-mass_read_mass))

    # Score-max oracle: selecting pages in descending max exact attention weight.
    # Per-page mass upper bounds U_j = page_size_j * max_i p_i.
    score_order = np.argsort(-page_max,kind="stable")
    z=0.0
    numerator=np.zeros(v.shape[1],dtype=np.float64)
    chosen=[]
    value_cert=float('inf')
    chosen_output=np.zeros_like(numerator)
    for j in score_order:
        j=int(j)
        ix=indices[j]
        z+=float(w[ix].sum())
        numerator+=w[ix] @ v[ix]
        chosen.append(j)
        chosen_output = numerator / z
        others=np.array([k for k in range(num_pages) if k not in chosen],dtype=int)
        if others.size:
            dist=np.linalg.norm(centers[others]-chosen_output,axis=1)+radii[others]
            value_cert=max_box_fraction(page_sum_bound[others], dist, z) / scale
        else:
            value_cert=0.0
        if value_cert <= tolerance:
            break
    value_error=float(np.linalg.norm(chosen_output-dense)/scale)
    if value_error > value_cert + 2e-10:
        raise AssertionError(f"Invalid numerical value certificate {value_error=} {value_cert=}")
    if value_error > tolerance + 2e-10:
        raise AssertionError("Value method exceeds error tolerance")
    if mass_error_certificate > tolerance + 2e-10:
        raise AssertionError("Mass oracle violated declared tolerance")

    # Greedy hindsight: knows the full output and exact page masses. Not a lower bound
    # on the optimal page count, not executable, not a certificate.
    h_z=0.0
    h_n=np.zeros(v.shape[1],dtype=np.float64)
    hindsight_n=num_pages
    hindsight_error=0.0
    for count, j in enumerate(mass_order,1):
        ix=indices[int(j)]
        h_z+=float(w[ix].sum())
        h_n+=w[ix] @ v[ix]
        err=float(np.linalg.norm(h_n/h_z-dense)/scale)
        if err <= tolerance:
            hindsight_n=count
            hindsight_error=err
            break

    return {
        "tokens":int(n),"dim":int(v.shape[1]),"pages":num_pages,
        "oracle_mass_pages":mass_needed,
        "oracle_mass_fraction":mass_needed/num_pages,
        "oracle_mass_certificate":float(mass_error_certificate),
        "oracle_exact_score_value_pages":len(chosen),
        "oracle_exact_score_value_fraction":len(chosen)/num_pages,
        "oracle_exact_score_value_certificate":float(value_cert),
        "oracle_exact_score_value_true_error":float(value_error),
        "greedy_hindsight_pages":int(hindsight_n),
        "greedy_hindsight_fraction":float(hindsight_n/num_pages),
        "greedy_hindsight_true_error":float(hindsight_error),
        "max_single_page_mass":float(page_mass.max()),
        "max_value_norm":float(vmax),
        "guarantee_type":"exact arithmetic sufficient error bound; numerical verification only",
        "oracle_note":"All methods use complete softmax weights and V; not executable sparsity"
    }


def summarize(rows:list[dict]) -> dict:
    if not rows:
        raise ValueError("No rows to summarize")
    metrics=("oracle_mass_fraction","oracle_exact_score_value_fraction",
             "greedy_hindsight_fraction","oracle_exact_score_value_true_error")
    result={"queries":len(rows)}
    for m in metrics:
        a=np.array([r[m] for r in rows],dtype=float)
        result[f"median_{m}"]=float(np.median(a))
        result[f"min_{m}"]=float(a.min())
        result[f"max_{m}"]=float(a.max())
    result["fraction_value_oracle_fewer_pages_than_mass_oracle"] = float(np.mean(
      [r["oracle_exact_score_value_pages"] < r["oracle_mass_pages"] for r in rows]))
    result["all_value_certificates_verified_numerically"]=True
    return result


def synthetic_fixtures(seed:int=4102) -> list:
    rng=np.random.default_rng(seed)
    n,d=1024,16
    w=rng.dirichlet(np.ones(n))
    base=rng.normal(size=d)
    return [
      ("uniform_probs_random_v", np.ones(n)/n, rng.normal(size=(n,d))),
      ("random_probs_random_v",w,rng.normal(size=(n,d))),
      ("random_probs_constant_v",w,np.tile(base,(n,1))),
      ("spike_probs_random_v",np.r_[0.95,np.ones(n-1)*(0.05/(n-1))],rng.normal(size=(n,d))),
      ("random_probs_near_constant_v",w,base+0.001*rng.normal(size=(n,d))),
    ]


def self_test_results() -> dict:
    rows=[]
    for name,w,v in synthetic_fixtures():
        result=evaluate_one(w,v,32,.02)
        result["regime"]=name
        rows.append(result)
    return {"metadata":{
          "date":DATE,"version":VERSION,"model":"NONE — mathematically synthetic fixtures",
          "gate":"G5 harness readiness, NOT G5 representative benchmark",
          "torch_required":False,"transformers_required":False,
          "python":platform.python_version(),"numpy":np.__version__,
          "page_size":32,"tolerance":.02,
          "limitations":"All oracle steps require exact attention weights and V; cannot establish achievable savings"
       },"summary":summarize(rows),"rows":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true',help='run synthetic correctness-only fixtures')
    parser.add_argument('--output',default='results/g5_trace_pipeline_smoke.json')
    args=parser.parse_args()
    if not args.smoke:
        parser.error('this standalone tool supports --smoke only; use g5_pythia_capture.py for pretrained')
    res=self_test_results()
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res['summary'],indent=2))
    print('saved',target)

if __name__=='__main__':
    main()
