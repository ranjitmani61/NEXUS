# G3 — Guarded mixed-precision screening datapath (software experiment) v1.0

**Date:** 2026-11-03. **Goal:** Test whether reduced-precision stored page extrema with FP32 score accumulation plus a conservative floating-point error envelope can avoid the unsafe FP16/BF16 serial accumulation witnessed in the preceding G3 audit, and evaluate its measured execution overhead. **Gate:** G3 IN PROGRESS. **Engineering ACTS:** A2 numerical analysis; A1 architecture; A5 arithmetic mapping; A6 memory; A8 verification; A10 performance; A11 prior art. No independent specialist review.

## Hypothesis and rejection conditions

- **H1 numerical:** Under documented IEEE-style, normal-range round-to-nearest arithmetic assumptions, a conventional `gamma_(2d)` correction around FP32 metadata scores will contain exact attention scores over the **stored quantized operands**. Reject if an exact stored-operand score falls outside a computed interval, or a selected approximation exceeds its reported normalized-error limit.
- **H2 speed:** Even with simple FP32 score guards and FP64 exact selected-page calculations, total measured CPU cost including metadata construction could be below dense reference on highly concentrated synthetic keys. Reject if measured total cost is higher at matched output definition.
- Baseline: dense FP64 NumPy attention over identically quantized K/Q and the same FP64 values; one-thread CPU. This is **not** a competitive optimized production inference baseline.
- Constraint: 2% maximum normalized attention-output error relative to exact stored-operand attention; not the original pre-quantization model. No hardware output claim.

## Datapath and equation

1. Store K and Q in FP16 or BF16. Build per-page coordinatewise min/max extrema. Convert extrema to FP32 for bound arithmetic.
2. For each FP32 query `q`, select sign-aware extrema `x_min` and `x_max`. Compute per-page FP32 reduction `s_min = sum(q_i*x_min_i)` and `s_max = sum(q_i*x_max_i)`.
3. With unit roundoff `u32 = 2^-24`, `gamma_(2d) = (2d*u32)/(1-2d*u32)`, derive conservative floating-point interval estimates `L=(s_min - gamma*sum|q*x_min| - pad)/sqrt(d)` and `U=(s_max + gamma*sum|q*x_max| + pad)/sqrt(d)`, where the implementation uses FP64 for error magnitudes and a small FP64 *heuristic* padding factor.
4. Pick the page with greatest `L`. Bound omitted softmax mass using the other pages' `U`, the selected page's `L`, and page token counts (equal here). The output error normalized by `max_i ||v_i||` is no more than `2 * omitted_probability_mass`, **conditional on valid score intervals and exact final attention**.
5. Admit selected-page FP64 attention only when the bounded normalized error is at most `0.02`; otherwise fall back to FP64 dense attention. Reject stale metadata by version check; reject nonfinite and extreme/subnormal FP32 cases.

This is **not a fully mixed-precision attention processor**: its screening bound arithmetic is FP32 and K/Q storage low precision, while the final attention is FP64. The arithmetic assumptions exclude unsupported device differences, overflow, subnormal behavior, fused-kernel variations, and externally corrupted metadata. The FP64 endpoint construction is **not formally directed-rounded**. Neither formal finite-precision correctness nor hardware correctness is established.

## Executed experiment and reproducibility

```
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/g3_reduced_precision_certificate_audit.py tests/test_g3_guarded_mixed_precision.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  tests/g3_guarded_mixed_precision_audit.py --output results/g3_guarded_mixed_precision_rep1.json
```

Rep2 and rep3 use the same invocation with the output filename changed. Dates, seeds, environment, and all timings are stored in JSON. CPU measurements are short, vary with clock/load, and are only valid as local NumPy/Python reference observations, not hardware performance predictions.

- Two formats (FP16/BF16); two seeds (1001/1031); four synthetic regimes (clustered/random/uniform/needle); 32 queries each; 64 pages × 16 tokens, dimension 64, value dimension 16.
- Three independent wall-clock runs, 16 rows/run = 48 measured case runs = 1,536 query comparisons. All 384 clustered comparisons admitted (32/32 in each case run); all other 1,152 fell back to dense.
- Full pytest set: **76/76** passed, including 9 previous precision-audit tests and 67 new ones. Direct FP32 guard interval checks across dimensions 16/64/128, both formats, multiple query distributions. Two constructed cancellation cases triggered dense fallback.
- Metadata bytes: 32,768 per case; stored K bytes: 131,072 per case. Metadata build time is included in total CPU timing. Reference K is preconverted to FP64 before either timing, and output error verification is performed outside the timed query paths.

| Format | Regime | Total CPU candidate / dense observed across three runs | Admissions |
|---|---|---:|---:|
| FP16 | Clustered | 1.76–2.40× | 192/192 |
| FP16 | Random | 2.62–4.08× | 0/192 |
| FP16 | Uniform | 2.93–4.34× | 0/192 |
| FP16 | Needle | 2.70–3.52× | 0/192 |
| BF16 | Clustered | 1.80–2.27× | 192/192 |
| BF16 | Random | 2.46–3.20× | 0/192 |
| BF16 | Uniform | 2.21–3.26× | 0/192 |
| BF16 | Needle | 2.24–3.05× | 0/192 |

All normalized output discrepancies in these fixtures were below the reported 8-decimal printed resolution (full numerical observations in JSON), not evidence of arbitrary bitwise equality under other workloads. The prior-precision witnesses were rejected by the FP32-screening admission (bound 0.06150 for FP16, 0.47957 for BF16; both exceed 0.02).

## Limits, novelty, and next gate

- No pretrained model, optimized dense/Quest baseline, GPU/TPU, RTL, cycle-level evaluation, energy, or real silicon is measured; G5 remains BLOCKED and G6+ not authorized.
- No proof for all FP32 inputs, formally outward-rounded endpoint arithmetic, serial-vs-parallel reduction semantics, concurrent streaming cache updates, value accumulation at reduced precision, quantization error relative to pre-quantized model, or softmax in reduced precision.
- The page-extrema selection method is established prior art (Quest 2024), and low-precision input/FP32 accumulation is an established engineering technique. No novel mechanism claimed.
- **Decision:** H2 rejected on this CPU implementation: all measured regimes are slower end to end. H1 passes only finite executed checks, not universal proof. Continue G3 only to close finite-precision proof obligations if this method later has real-world utility. In the meantime, attention-screening hardware is not justified.

## External references

- Quest, Tang et al. (2024): https://proceedings.mlr.press/v235/tang24l.html
- Classical floating-point `gamma_n` bound: https://www.cs.utsa.edu/~atc/pub/J42.pdf (surveys Higham's result).
- NVIDIA GPU background (FP16 inputs, sometimes FP32 accumulation): https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/
- TensorRT precision control: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html

All experiment details are inspectable in `tests/g3_guarded_mixed_precision_audit.py`, `tests/test_g3_guarded_mixed_precision.py`, and `results/g3_guarded_mixed_precision_rep{1,2,3}.json`.
