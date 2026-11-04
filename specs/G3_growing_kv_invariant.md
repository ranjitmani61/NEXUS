# Project 01 G3 — Incremental KV Certificate Invariants

**Version:** 1.0 | **Date:** 2026-11-03 | **Gate:** G3 in progress  
**Review lenses:** A1 architecture; A2 mathematics; A3 ML systems; A4 runtime; A6 memory; A8 verification; A10 performance. **Independent human review:** none.

## Purpose and assumptions

Audit the previously proposed **known** center/radius screening and query-anchor reuse under **growing** KV pages. This is not a new theorem, circuit, or novel processor design. Intended application: autoregressive causal attention, not full inference validation. Exact-arithmetic mathematical claims are separate from floating-point numerical results.

## State and invariants

For each current page `i` of `n_i` keys in `R^d`:

- `c_i` = page mean of **all currently present keys**.
- `r_i >= max_j ||k_ij - c_i||_2`.
- `g_i >= ||c_i||_2 + r_i`.
- `U_i(a) >= max_j a·k_ij/sqrt(d)` for the stored anchor query `a`.

For any query `q`, the **fresh** page bound is

`F_i(q) = (q·c_i + ||q||_2 r_i)/sqrt(d)`.

The **reused-anchor** page bound is

`R_i(q|a) = U_i(a) + ||q-a||_2 g_i/sqrt(d)`.

Both are upper bounds on all current page scores in exact arithmetic by Cauchy–Schwarz and the triangle inequality.

**Critical concurrency invariant:** Before any new decode query sees a newly appended key, every modified page must have correctly updated `c_i`, `r_i`, `g_i` and `U_i(a)`; alternatively, disable screening and fall back to dense attention until an atomic metadata update is visible. The other closed pages can retain valid metadata. A stale bound on even one page invalidates certification.

## Output certificate

Let `Z` be the exact softmax exponential sum for the visited keys using a stable common score shift; let `M` be the unvisited exponential sum; let `U = sum_i n_i exp(R_i)` be a safe bound so `M <= U`. Let `Vmax >= max_j ||v_j||_2` cover **all current values**, including appended ones. Then the normalized L2 error of partial attention is bounded by

`||partial_output - full_output||_2 / Vmax <= 2 U / (Z + U)`.

Proof: the full output is the convex combination of the visited-value mean and unvisited-value mean with weights `Z/(Z+M)` and `M/(Z+M)`; each mean has norm at most `Vmax`. Thus the difference has norm at most `2 Vmax M/(Z+M)`, increasing in `M` for `M >= 0`. The code computes the expression in the log domain. **Finite precision:** outward rounding of bounds, safe handling of norm/reduction error, and robust control flow are not implemented here; float64 numerical checks do not prove safety in float32/FP16/BF16.

## Negative control

A newly inserted high-scoring key can invalidate a previously safe bound on its page. `tests/g3_dynamic_kv_properties.py` constructs precisely this case. A deliberately stale bound falsely reports error <= 0.000671 while its observed normalized output error is 1.0. When metadata is updated before selection, the observed error becomes 0 in the same case.

## Implementation and reproducibility

```
cd /mnt/data
OPENBLAS_NUM_THREADS=1 python tests/g3_dynamic_kv_properties.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g3_g4_autoregressive_trace_audit.py --output results/g3_g4_toy_trace_audit.json
```

- Incremental pages recompute the single partially filled page after each appended key and value. This is mathematically safe but may be expensive.
- The transformer trace generator is a **randomly initialized, untrained**, two-layer, four-head, teacher-forced causal transformer using RoPE. It has no demonstrated language ability.
- The attention-screening engine performs score and output calculations on **only selected** pages; a separate dense oracle is run for correctness inspection. Its oracle cost is not part of modeled sparse traffic.
- Numerical tests compare every page bound with exact dense scores and actual normalized attention-output error with the stated certificate in float64.
- The cost model is a rough bytes-read accounting, not a CPU performance result or GPU, HBM, energy or physical-design measurement.

**G3 status:** IN PROGRESS. Exact-arithmetic reasoning is straightforward and synthetic growing-KV tests pass, but finite-precision proof, adversarial concurrency, and real pretrained-model inference validation are missing.