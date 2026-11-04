# G4 Executable CPU Admission / Fallback Cost Model — Project 01

- **Date/version:** 2026-11-03, v1.1
- **Status:** local diagnostic, G4 IN PROGRESS; NOT a pretrained model or hardware benchmark
- **ACT review lenses applied:** A1 architecture, A2 applied mathematics, A3 ML systems, A4 runtime, A6 memory/interconnect, A8 verification, A10 performance, A11 prior art. No independent hardware or mathematical reviewer.
- **Objective:** falsify the notion that reduced synthetic KV reads necessarily produce lower execution latency; test a conservative cheap preflight with exact fallback and full preprocessing charges.
- **Baseline:** dense FP64 NumPy full softmax attention on the same CPU, Q/K/V, math precision, and arrays; NOT optimized GPU FlashAttention/Quest.

## Mechanism and proof assumption

Given a static page of L keys with center c_b and max Euclidean radius r_b, the scaled QK score s_i = q·k_i / sqrt(d) lies inside

    lower_b = (q·c_b - ||q|| r_b)/sqrt(d)
    upper_b = (q·c_b + ||q|| r_b)/sqrt(d).

Choose j = argmax upper_b. Before reading any page's full K/V, compute the upper bound on omitted unnormalized mass `U = L sum_{b != j} exp(upper_b - shift)` and a lower bound on selected mass `Zlower = L exp(lower_j - shift)`. If

    2 U / (U + Zlower) <= 0.02,

read and compute attention only over page j. If not, run exact dense attention. Why sufficient: `Zactual >= Zlower`, omitted mass `R <= U`, and with `vmax = max_i ||v_i||` the normalized output error is bounded by `||y_subset-y_dense|| / vmax <= 2 R/(Zactual+R) <= 2 U/(Zlower+U)` in exact arithmetic, for current K/V and the usual renormalized subset output.

The certificate is an output-vector error criterion only, not an end-task quality bound. FP64 is tested numerically; no formally rounded finite-precision guarantee. Metadata MUST be refreshed when K changes. This certificate and fallback are not proved original or patentable; Quest and other prior art already show query-aware page selection. Source: https://proceedings.mlr.press/v235/tang24l.html

## Scope and pre-registered parameters of the local experiment

- NumPy single-thread BLAS CPU reference; float64 throughout; `B=128`, `L=32`, `d=64`, `tol=0.02`.
- Two fixed seeds 1029 and 1031; 20 correlated queries/seed; clustered, random Gaussian, uniform-key, and rare-needle regimes.
- 40 timed queries per regime, 3 repeated measurements per query; measure median and p95; alternate call order to reduce bias. Two complete runs with identical parameters. 160 queries per full run.
- Metadata construction CPU timing: compute key centers and radii from K, take median of five runs; the offline V norm used to verify error is NOT required online and is excluded from admitted runtime charge (corrected from initial v1.0 study).
- Q20 illustrative amortization: total timed inference per static cache + one metadata build per 20 queries, compared to dense for those 20 queries. Real autoregressive decoding appends K/V, so an actual implementation must measure incremental metadata updates and changed-page invariants. Q20 must not be generalized to an evolving KV cache.
- Metadata bytes, cache misses, thermal load, joules and accelerator performance were NOT measured. Python control loop, CPU caching and system noise mean extrapolations to processor design are unsupported.

## Reproduce

    cd /mnt/data
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_g4_hybrid_admission.py tests/test_g5_trace_oracle.py tests/test_g5_capture_shapes.py tests/test_g5_learned_control.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/g4_hybrid_admission_benchmark.py --output results/replication.json

## Decision

G4 local CPU execution comparison is a useful diagnostic and negative control, but not complete against competitive algorithms or representative hardware. G5 remains BLOCKED. The result does not justify starting G6 microarchitecture or claiming G12 innovation.