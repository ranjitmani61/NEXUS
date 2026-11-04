# Project 01 — G3 guarded mixed-precision execution review v1.0

**Date:** 2026-11-03 · **Independent reviewer:** none · **ACT lenses:** A1/A2/A5/A6/A8/A10/A11. **AGENTS.md:** read before work, unchanged.

## Gate decision

**G3 IN PROGRESS. Hardware engineering NOT AUTHORIZED. G5 BLOCKED. G12 novelty/breakthrough NOT DEMONSTRATED.**

The new screened precision path fixes the previously constructed *toy* low-precision accumulator failure by moving screening to FP32 with conventional error guards and full dense fallback. Both cancellation witnesses refused admission. Direct interval-inclusion tests and attention output tests had zero detected failures over declared cases; they are **empirical software checks, not a formal hardware safety proof**.

In 1,536 timed query comparisons across FP16/BF16, 4 synthetic regimes, 2 seeds, and 3 independent repetitions, the candidate was **slower than FP64 dense attention in every tested case**. Including metadata construction, total CPU cost ranged from 1.76× to 4.34× dense across cases. Highly clustered K admitted one page of 64 each time; all random, uniform and needle cases fell back to dense. 76 automated tests passed across the new and prior precision audit.

## Material methodological constraints

1. FP64 dense reference is a matching stored-operand CPU baseline, but **not** a production-optimized GPU/CPU attention library; this does not prove performance on actual AI accelerators.
2. The selected-page final attention remains FP64. This is a *mixed-precision screening stage*, not a complete mixed-precision accelerator implementation.
3. Quantization error vs the unquantized model is excluded; only error vs the attention computed from the same stored low-precision Q/K is measured.
4. The standard FP32 error estimate assumes normal, finite round-to-nearest arithmetic and the `gamma_n` model. FP64 endpoint assembly uses heuristic padding, not mathematically certified outward rounding.
5. Pretrained-model traces, optimized Quest comparator, dynamic cache concurrency, eventual RTL/FPGA/ASIC PPA, power, and thermal data are absent.
6. This implementation's latency is dominated by Python/NumPy operations and screening overhead; architecture-level projections require new data, not an extrapolation of these ratios.
7. Metadata version checks do not protect against erroneous bounds written with an unchanged version or against arbitrary memory corruption.

## Next engineering action

**Stop performance-promoting this candidate.** If real pretrained traces become available, rerun a competitive G5 comparison. If numerical correctness is independently valuable, specify an actual FP32/FP64 directed-rounding implementation with known operator semantics and formally verify the interval endpoint logic before G3 passes. Do not start G6 hardware design without evidence of utility and novelty.

Evidence: `specs/G3_guarded_mixed_precision_datapath.md`, `tests/g3_guarded_mixed_precision_audit.py`, `tests/test_g3_guarded_mixed_precision.py`, `results/g3_guarded_mixed_precision_rep{1,2,3}.json`, `results/g3_guarded_mixed_precision_full_pytest.txt`.