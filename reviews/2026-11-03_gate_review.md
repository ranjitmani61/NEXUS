# Gate Review — 2026-11-03

**Project:** 01 — Novel AI processor research  
**Rules:** AGENTS.md v1.0 read before work.  
**Review lenses:** A1, A2, A3, A4, A6, A8, A10, A11. Independent human review: **none**.  
**Version:** 1.0

## Reproduction

Commands:

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 python processor_baseline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/attention_screening_audit.py --output results/attention_screening_audit.json
```

Environment: Python 3.13.5; NumPy 2.3.5; float64 computations, float32-equivalent memory-accounting proxy. Real hardware timing/power not performed.

**Original prototype rerun matched stored record:** clustered 1/128 pages, 2.368% modeled traffic; unstructured 128/128, 101.587%. These numbers reflect *original* sphere-only implementation, with slightly different metadata accounting.

**New verification/audit:** 4 synthetic regimes × 12 predetermined seeds × 2 bound strategies = **96 runs** and **192 additional small boundary/property cases**. No observed violations of either max-score bound or output-error certificate in this test set. This is not a formal proof or verification of finite-precision hardware.

## Audit results

Median (n=12 seeds in each regime and bound); `modeled traffic` includes **all page metadata read per query**, and selected K/V page reads. Base = dense K/V read in float32 equivalent; actual calculations were float64.

| Synthetic regime | Bound | Median selected pages /128 | Traffic (no build) | Traffic (single-query full-scan build) | Traffic (128-query amortized full-scan build) |
|---|---|---:|---:|---:|---:|
| Clustered | Center/radius | 1 | 2.39% | 102.39% | 3.17% |
| Clustered | Axis-wise min/max | 1 | 3.93% | 103.93% | 4.71% |
| Independent Gaussian | Center/radius | 128 | 101.61% | 201.61% | 102.39% |
| Independent Gaussian | Axis-wise min/max | 128 | 103.15% | 203.15% | 103.93% |
| Uniform keys | Center/radius | 127 | 100.83% | 200.83% | 101.61% |
| Uniform keys | Axis-wise min/max | 127 | 102.37% | 202.37% | 103.15% |
| Rare boosted key | Center/radius | 127.5 | 101.22% | 201.22% | 102.00% |
| Rare boosted key | Axis-wise min/max | 128 | 103.15% | 203.15% | 103.93% |

Source of exact run-level numbers: `results/attention_screening_audit.json`.

### Mathematical audit

For every block `b` with keys `k_{b,j}`, define `c_b = mean(k_{b,*})`, `r_b = max_j ||k_{b,j}-c_b||`. Then by Cauchy–Schwarz each score is `<= (q·c_b + ||q|| r_b)/sqrt(d)`. A second, independent bound is `sum_i max(q_i kmin_{b,i}, q_i kmax_{b,i})/sqrt(d)` with the branch selected by sign of `q_i`; that is an axis-aligned box support function.

With visited softmax mass `Z` and unvisited upper mass `U`, omitted true mass `u <= U`, and the known stored/global upper bound `Vmax >= max ||v||`,
`||output_all - output_visited|| / Vmax <= 2u/(Z+u) <= 2U/(Z+U)`.
This inequality holds in **exact arithmetic**. Numerical/software implementation error is **not** included in the certificate. Maintaining block value-norm maxima in metadata is assumed; no per-query scan of V is credited as free.

### Limitations and disconfirming cases

1. No actual model traces or evaluation of model downstream quality. A per-head norm guarantee does not guarantee downstream model quality.
2. Float32-equivalent traffic, but float64 calculations; quantization, rounding and real GPU/ASIC memory transactions not measured.
3. Initial full K/V metadata build read is counted at one full dense-cache scan **only as a conservative scenario**, not a measured necessity. Construction/update CPU operations, sorting, compression/metadata writes and cache/interconnect effects are not fully modeled.
4. Q=128 reuse scenario assumes cache/metadata reuse between queries, unlike evolving autoregressive KV caches without updates; never apply 3.17% to a real decoder without tracking update costs.
5. The favorable clustered data is engineered to make screening trivial; 3 of 4 regimes have **more** traffic than dense in no-build accounting. Do not report the minimum as representative.
6. The min/max method is an **independently formulated bound** inspired by Quest's metadata description; no claim to replicate Quest accuracy, implementation or speedups.
7. Strong software and hardware baselines were identified but not run; absence of hardware access prevents any throughput/energy claims.
8. Source/prior-art review is preliminary and incomplete on patents and contemporary custom hardware.

## Gate decisions

| Gate | Status and rationale |
|---|---|
| G0 — Charter | **PASS for exploratory research only**; see `specs/G0_benchmark_contract.md` |
| G1 — Prior art | **IN PROGRESS**; close match to center/radius bound explicitly discovered; G1 cannot pass for novelty |
| G2 — New mechanism feasibility | **NOT STARTED**; no distinctive architectural mechanism yet |
| G3 — New algorithm | **NOT STARTED**; reference screening is pre-existing; synthetic audit alone does not pass a new design |
| G4 — New design simulation | **NOT STARTED**; baseline model audited, no new design benchmark |
| G5–G12 | **NOT STARTED/BLOCKED by resources** (where hardware/real workloads required) |

## Next authorized action

- Identify a **different and defensible architectural mechanism** through close reading of prior art before committing silicon design effort.
- Evaluate whether memory-side execution, compiler and page-bound arithmetic are viable under explicit bandwidth, metadata/update, precision and scheduling constraints.
- Obtain representative public model KV-cache traces and competitive kernel baselines when suitable runtime hardware is available.
- Avoid changing AGENTS.md or historical results to make unfavorable cases disappear.
