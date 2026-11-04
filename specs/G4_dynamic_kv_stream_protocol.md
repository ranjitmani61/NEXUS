# G4 Dynamic KV Streaming CPU Experiment — Protocol v1.3

**Date:** 2026-11-03  
**Gate:** G3/G4 IN PROGRESS, G5 BLOCKED  
**Engineering lenses:** A1 architecture, A2 applied mathematics, A3 ML systems, A4 compiler/runtime, A6 memory, A8 verification, A10 systems performance, A11 prior art.  
**Independent expert review:** none. **Novelty claimed:** none.

## Research question
Does the conservative one-page screening/fallback method preserve its output-error certificate while a KV cache grows, and does it beat exact dense attention when initial metadata construction plus per-token metadata updates are charged?

## Hypotheses and matched baseline
1. Every appended key is incorporated before screening; stale metadata forces exact dense attention.
2. The fixed-center and coordinatewise min/max summaries give valid real-arithmetic lower and upper score bounds.
3. In clustered synthetic data only, one page may dominate sufficiently to admit one-page attention within a 2% normalized L2 error bound.
4. In unstructured, uniform, and needle regimes, screening often fails and fallback overhead can cause a net slowdown.

**Baseline:** exact float64 NumPy dense softmax attention on the same active Q/K/V arrays, same CPU, same input data; this is NOT a competitive optimized GPU or LLM system baseline.

## Workload and experiment setup
- 2 seeds: 1029, 1031; 4 conditions: clustered, Gaussian random, uniform keys, late needle.
- Initially 4,096 K/V tokens, 128 subsequent append/query steps, head dimension 64, 32 tokens per page.
- Two page-summary methods: fixed first key and max Euclidean distance, or coordinatewise min/max (known from Quest).
- Each mode computes a sufficient certificate for reading **only one page**, otherwise exact dense attention is used.
- Each query path is timed 3 times and median taken; measurement ordering alternates.
- Initial summary construction is vectorized over full pages and its measured cost is charged once; each incremental summary update is separately timed and charged. K/V insertion common to both paths is excluded from the comparison.
- Output correctness is checked against dense attention offline. Offline V vector norms and dense reference verification are excluded from candidate timing, but normalization uses the maximum over **only currently visible values**.
- Total cost ratio = (sum of candidate inference time + sum of metadata update time + initial metadata build) / sum of dense inference time.

## Math of screening
For page j containing n_j active keys, if l_j <= q·k/sqrt(d) <= u_j for every key in the page, select t maximizing `log(n_j)+l_j`. Define

- `U = sum_{j != t} n_j exp(u_j-s)`; `Z_low = n_t exp(l_t-s)`; `s = max_j u_j`.
- Certificate: `C = 2 U/(U+Z_low)`.
- If `C <= epsilon=0.02`, compute the exact softmax output over page t only. Else run exact dense attention.

Because the omitted mass is at most U and selected mass at least Z_low, in exact arithmetic the normalized L2 output error relative to `Vmax = max_visible ||v_i||_2` is <= C. Fallback produces exact reference output (subject to floating-point rounding).

Fixed-center bounds: `(q·c +/- ||q|| r)/sqrt(d)`, with c the first key and r the largest distance to c so far. Min/max bounds: `sum_i min/max(q_i min_i, q_i max_i)/sqrt(d)`.

**Caution:** This is a sufficient exact-real-arithmetic argument, NOT a finite-precision, BF16, GPU, atomic update or end-task quality proof. The version-counter fallback is a sequential guard; it does not establish thread-safe concurrent readers/writers.

## Reproduce
```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=tests python -m pytest -q \
    tests/test_g4_dynamic_kv_stream.py tests/test_g4_hybrid_admission.py \
    tests/test_g5_trace_oracle.py tests/test_g5_capture_shapes.py tests/test_g5_learned_control.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/g4_dynamic_kv_stream_benchmark.py \
    --output results/g4_dynamic_kv_stream_reproduction.json
```

## Known prior art and limitations
- Quest (Tang et al., ICML 2024) maintains page-level key min/max metadata and performs query-aware KV-page selection; see https://proceedings.mlr.press/v235/tang24l.html and https://github.com/mit-han-lab/Quest . It does not make this exact particular experiment a new processor.
- No real pretrained model, actual GPU, energy, DRAM byte counter, area, ASIC, compiler, silicon cost or optimized FlashAttention/Quest comparison.
- Python/NumPy CPU microbenchmarks are sensitive to runtime, cache, and library implementation.
- Randomly generated keys are not validated as realistic LLM traces; an artificially strong page forms a favorable control.
- Only one head, one growing sequence, no evictions/reuse/concurrency, and no end-to-end model quality.
- Metadata construction can dominate if implemented inefficiently. A prior v1.1 Python token-loop build yielded a misleading negative result that was corrected in v1.2 with vectorized bulk construction. v1.3 also uses current visible Vmax for the offline numerical check.

**Gate expectation:** G3/G4 remain IN PROGRESS even if local tests pass; G5 remains BLOCKED pending representative pretrained traces and a competitive optimized baseline. Do not authorize hardware design or claim novelty from this experiment.