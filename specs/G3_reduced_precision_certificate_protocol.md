# G3 Reduced-Precision Certificate Protocol — v1.0

**Date:** 2026-11-03. **Gate:** G3 IN PROGRESS. **Engineering lenses:** A2 mathematics/numerical analysis, A5 digital arithmetic, A8 verification, A1 architecture, A3 ML systems, A10 performance. This is role-based analysis, NOT an independent human review.

## Objective and falsifiable hypothesis

Determine whether a hypothetical per-page score-bound datapath using serial FP16 or BF16 arithmetic can falsely certify 2%-tolerant attention pruning. Test whether an explicit worst-case dot-product roundoff envelope prevents the constructed false admission.

The existing dynamic KV reference `tests/g4_dynamic_kv_stream_benchmark.py` uses FP64 NumPy bounds, **not** the low-precision serial datapath examined here. This experiment is a design-safety finding, not a regression in that reference.

## Mathematical certificate and counterexample

For two pages, each containing one key, and a query `q` of dimension `d=3`:

- `q=(1,1,1)`; page 0 `k0=(0,10,0)`, `v0=(1,0,0)`;
- FP16 page 1: `k1=(32768,16,-32768)`; BF16 page 1: `k1=(2048,8,-2048)`; `v1=(-1,0,0)`.
- Stored values are exactly representable in their respective formats. True unnormalized dot products are `10,16` (FP16) and `10,8` (BF16), in exact arithmetic over the stored values.
- A **serial multiplication-plus-addition at the stored precision** computes page 1's dot as zero because the middle term is lost at the large intermediate. This is NOT the behavior of all low-precision matrix engines: a FP32-accumulating dot can retain the middle term.
- Normalized scores divide by `sqrt(3)`. Naive metadata then selects page 0 with `cert = 2 exp(-10/sqrt(3)) / (1+exp(-10/sqrt(3))) = 0.006198428...` for both formats, apparently below tolerance `0.02`.
- True normalized output error measured against exact reference softmax over quantized operands is `1.939297819...` (FP16), `0.479263116...` (BF16), both > `0.02` and both > claimed naive certificate.
- The **mass-certificate theorem itself is not refuted**. A hypothesis of that theorem—conservative page bounds—fails because naive FP16/BF16 accumulation supplied invalid bounds.

Independent 70-digit `decimal.Decimal` evaluation of the analytic two-score softmax reproduced both errors to floating-point reporting precision.

## Candidate guard (not fully formalized)

For a length-`d` serial dot in a round-to-nearest format with unit roundoff `u`, excluding all nonzero subnormal/overflow operations, use the standard conservative error envelope:

`|fl(q dot k) - (q dot k)| <= gamma_(2d) * sum_i |q_i*k_i|`, where `gamma_n = n*u/(1-n*u)`, provided `2*d*u < 1`.

We choose `u16=2^-11`, `u_bf16=2^-8`. For every page, replace the naive point score with a low/high score interval from the envelope (with limited extra FP64 computation padding in the implementation). Use **lower** bounds for the admitted page and **upper** bounds for all omitted pages; otherwise perform dense fallback. A toy implementation does this for one key per page. The witness is rejected by the guard in both formats.

The `gamma_(2d)` envelope and its assumptions are documented in Higham & Mary's numerical-analysis literature: https://epubs.siam.org/doi/10.1137/18M1226312 (see also https://tmary.perso.lip6.fr/doc/surveyMixed.pdf ). This is conventional numerical analysis, NOT a novel computing mechanism.

## Commands and actual test coverage

```sh
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/g3_reduced_precision_certificate_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g3_reduced_precision_certificate_audit.py --output results/g3_reduced_precision_audit.json
```

- Nine focused automated test functions passed.
- Two constructed false-certificate witnesses reproduced (FP16, BF16).
- 2,000 random dot-product guard trials: 1,998 evaluable, two conservative fallbacks, zero observed interval violations.
- 1,000 random two-page attention-certificate trials: 499 admitted by the guard, zero detected certified-output violations.
- NaN, infinity, invalid shapes/dtypes and FP16 subnormals lead to conservative fallback; favorable control case still admits selection.
- Exact outputs, counts, software version, and source assumptions in `results/g3_reduced_precision_audit.json`; test logs in `results/g3_reduced_precision_pytest.txt`.

## Limitations and next gate

1. This is a **synthetic two-page** numerical safety test, no real LLM traces, Quest comparison, energy, speed, silicon area, RTL, or physical hardware.
2. The error envelope covers simple serial dot operations only, assuming IEEE-like round-to-nearest and normal intermediates. It does not yet bound metadata radius/norms, softmax `exp`, normalization, value summation, concurrent cache updates or output rounding.
3. Computation of `gamma`, sum of absolute products and interval endpoints in the audit uses FP64 with heuristic padding, not a fully formally proven outward-rounded interval library. **Do not call the code a verified end-to-end implementation.**
4. The guard may cause unacceptable fallback rates in wider dimensions; no runtime overhead was measured. Do not replace FP64 with FP16/BF16 based on these checks.
5. G3 remains IN PROGRESS until the complete prospective numerical datapath, exceptional values, and concurrency are specified and proven or exhaustively checked within stated domain. G5 remains BLOCKED due to absent representative pretrained weights and strong optimized comparators. G6/RTL is not authorized.

## Decision

- **FAIL** the proposed naive serial FP16/BF16 metadata arithmetic for certified screening.
- **PASS** the focused existence proof of an unsafe failure and the toy roundoff-guard smoke/stress checks under stated assumptions.
- **HOLD** G3 overall (IN PROGRESS), G5 BLOCKED, and any claims of architectural novelty or breakthrough.
