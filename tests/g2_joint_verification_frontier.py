#!/usr/bin/env python3
"""NEXUS G2: optimistic speculative-verification + KV-reuse frontier.

This is NOT a speculative decoder, trained model, or new processor.
All measured timings are local single-head CPU microbenchmarks only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

VERSION = "1.0"
SEED = 20261110
ANCHORS = {
    "qwen05_b4_T32000_hostA": {"model": "Qwen2.5-0.5B", "batch": 4, "context": 32000, "f": 0.5861178531211508, "host": "G5-R host A"},
    "granite4micro_b1_T16384_hostB": {"model": "Granite-4.0-micro-base", "batch": 1, "context": 16384, "f": 0.7052066730549075, "host": "G5-R host B"},
}
SOURCE = "https://github.com/ranjitmani61/NEXUS/blob/8510d89980816e06a511ab0d665188cbe7da3d6c/results/g5r/summary.json"


def _fraction(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite in [0,1]")
    return float(value)


def _nonnegative(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return float(value)


def _integer_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be integer >=1")
    return k


def expected_output_tokens(k: int, p: float) -> float:
    """Constant per-position conditional acceptance, incl. one fallback or bonus token."""
    k = _integer_k(k)
    p = _fraction("p", p)
    if p == 1.0:
        return float(k + 1)
    return -math.expm1((k + 1) * math.log(p)) / (1.0 - p) if p > 0 else 1.0


def target_cost_units(f: float, k: int, beta: float, delta: float, h: float = 0) -> float:
    """Optimistic: 1 nonattention weight pass + k additional attention queries."""
    f = _fraction("f", f)
    k = _integer_k(k)
    beta = _nonnegative("beta", beta)
    delta = _nonnegative("delta", delta)
    h = _nonnegative("h", h)
    return 1.0 + f * beta * k + delta * k + h


def modeled_speedup(f: float, k: int, p: float, beta: float, delta: float, h: float = 0) -> float:
    return expected_output_tokens(k, p) / target_cost_units(f, k, beta, delta, h)


def max_beta_for_two_x(f: float, k: int, p: float, delta: float, h: float = 0) -> float | None:
    """Largest beta allowing S>=2. None when f=0 (beta irrelevant)."""
    f = _fraction("f", f)
    k = _integer_k(k)
    delta = _nonnegative("delta", delta)
    h = _nonnegative("h", h)
    length = expected_output_tokens(k, p)
    return (length / 2.0 - 1.0 - delta * k - h) / (f * k) if f > 0 else None


def min_acceptance_rate_for_two_x(f: float, k: int, beta: float, delta: float, h: float = 0) -> float | None:
    """Inverse required p for 2x within the optimistic independent-accept model."""
    target_cost_units(f, k, beta, delta, h)  # validate inputs
    if modeled_speedup(f, k, 1.0, beta, delta, h) < 2.0 - 1e-12:
        return None
    if modeled_speedup(f, k, 0.0, beta, delta, h) >= 2.0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(75):
        mid = (lo + hi) / 2.0
        if modeled_speedup(f, k, mid, beta, delta, h) >= 2.0:
            hi = mid
        else:
            lo = mid
    return hi


def simulated_output_tokens(k: int, p: float, seed: int, n: int) -> float:
    """Empirically check independent-acceptance accounting; not LLM model accuracy."""
    k = _integer_k(k)
    p = _fraction("p", p)
    if n <= 0:
        raise ValueError("n>0")
    rng = np.random.default_rng(seed)
    successes = (rng.random((n, k)) < p)
    # First rejection emits fallback; all accepted emits bonus.
    return float(np.mean(1 + np.cumprod(successes, axis=1).sum(axis=1)))


def _softmax_fp32(scores: np.ndarray) -> np.ndarray:
    z = scores - np.max(scores, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def serial_attention(q: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Reference: independent one-query dense historical-prefix attention."""
    if q.ndim != 2 or keys.ndim != 2 or values.ndim != 2 or q.shape[1] != keys.shape[1] or keys.shape != values.shape:
        raise ValueError("expected Q[n,d], K[T,d], V[T,d]")
    scale = float(q.shape[1]) ** -0.5
    outputs = []
    for row in q:
        scores = (keys @ row) * scale
        weights = _softmax_fp32(scores)
        outputs.append(weights @ values)
    return np.stack(outputs, axis=0)


def fused_attention(q: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Known dense GEMM form; no new algorithm, exact same historical prefix."""
    if q.ndim != 2 or keys.ndim != 2 or values.ndim != 2 or q.shape[1] != keys.shape[1] or keys.shape != values.shape:
        raise ValueError("expected Q[n,d], K[T,d], V[T,d]")
    scores = (q @ keys.T) * (float(q.shape[1]) ** -0.5)
    return _softmax_fp32(scores) @ values


def _timer(func, args, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        func(*args)
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        func(*args)
        values.append((time.perf_counter_ns() - t0) * 1e-6)
    return values


def cpu_attention_microbenchmark(seed: int = SEED, sizes: tuple[int, ...] = (4096, 8192, 16384),
                                 query_counts: tuple[int, ...] = (1, 2, 4, 8),
                                 dim: int = 64, warmup: int = 5, repeats: int = 13) -> list[dict]:
    if repeats < 3 or warmup < 0:
        raise ValueError("need at least 3 repeats, warmup>=0")
    result = []
    rng = np.random.default_rng(seed)
    for t in sizes:
        keys = rng.standard_normal((t, dim), dtype=np.float32)
        values = rng.standard_normal((t, dim), dtype=np.float32)
        queries = rng.standard_normal((max(query_counts), dim), dtype=np.float32)
        base = _timer(serial_attention, (queries[:1], keys, values), repeats, warmup)
        single = statistics.median(base)
        for n in query_counts:
            q = queries[:n]
            serial = serial_attention(q, keys, values)
            fused = fused_attention(q, keys, values)
            maxabs = float(np.max(np.abs(serial - fused)))
            allclose = bool(np.allclose(serial, fused, atol=3e-4, rtol=3e-4))
            if not allclose:
                raise RuntimeError(f"attention equivalence failed: T={t}, n={n}, maxabs={maxabs}")
            # Each trial measures both variants in alternating order; timings are descriptive.
            serial_ms, fused_ms = [], []
            for z in range(repeats):
                if z % 2:
                    pair = [(serial_attention, serial_ms), (fused_attention, fused_ms)]
                else:
                    pair = [(fused_attention, fused_ms), (serial_attention, serial_ms)]
                for fun, dest in pair:
                    start = time.perf_counter_ns()
                    fun(q, keys, values)
                    dest.append((time.perf_counter_ns() - start) / 1e6)
            median_fused = statistics.median(fused_ms)
            result.append({
                "T": t, "dim": dim, "n": n, "seed": seed,
                "single_query_median_ms": single,
                "serial_median_ms": statistics.median(serial_ms),
                "fused_median_ms": median_fused,
                "serial_p10_p90_ms": [float(x) for x in np.quantile(serial_ms, [.1, .9])],
                "fused_p10_p90_ms": [float(x) for x in np.quantile(fused_ms, [.1, .9])],
                "fused_vs_serial_latency_ratio": median_fused / statistics.median(serial_ms),
                "fused_attn_cost_ratio_to_one_query": median_fused / single,
                "fused_incremental_beta": (median_fused / single - 1.0) / (n - 1) if n > 1 else None,
                "max_absolute_output_difference": maxabs,
                "numerically_equivalent": allclose,
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()

    scenarios = []
    for name, anchor in ANCHORS.items():
        for k in (1, 2, 4, 8):
            for p in (0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
                for delta in (0, 0.05, 0.10):
                    for h in (0, 0.05):
                        threshold = max_beta_for_two_x(anchor["f"], k, p, delta, h)
                        scenarios.append({"anchor": name, "k": k, "p": p, "delta": delta, "h": h,
                                          "E_committed": expected_output_tokens(k, p),
                                          "beta_max_for_2x": threshold,
                                          "2x_possible_even_with_beta_0": threshold is not None and threshold >= -1e-12,
                                          "speedup_beta0": modeled_speedup(anchor["f"], k, p, 0, delta, h),
                                          "speedup_beta1": modeled_speedup(anchor["f"], k, p, 1, delta, h)})
    artifacts = {
        "name": "G2 speculative verification and shared-KV speed frontier",
        "version": VERSION, "created_date": "2026-11-10",
        "status": "conditional analytical screening; real-model G5 NOT RUN",
        "source_g5r": SOURCE,
        "anchors": ANCHORS,
        "model_note": "k draft proposals, k+1 target queries, identical p assumption, nonattention target work charged only once; output quality and beta transfer NOT verified",
        "scenario_count": len(scenarios), "scenario_matrix": scenarios,
        "cpu_benchmark": [] if args.skip_timing else cpu_attention_microbenchmark(),
        "cpu_environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "gate_decision": {"G1": "existing speculative + shared-KV mechanisms known; no novelty established",
                          "G2": "conditional analytic feasibility only",
                          "G5": "NOT RUN", "G6_plus": "NOT AUTHORIZED"},
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifacts, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "scenarios": len(scenarios),
                      "bench_rows": len(artifacts["cpu_benchmark"]),
                      "max_abs_output_diff": max((x["max_absolute_output_difference"] for x in artifacts["cpu_benchmark"]), default=None),
                      "gate": artifacts["gate_decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
