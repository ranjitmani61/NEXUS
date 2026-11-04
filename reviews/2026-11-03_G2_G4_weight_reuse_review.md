# Gate review — G2/G4 weight reuse and latency-throughput tradeoff

**Date:** 2026-11-03 | **Decision:** Existing technique confirmed, limited CPU behavior reproduced. No processor novelty. G5 blocked.

**Protocol:** `AGENTS.md` reread before work; inspected prior G2 byte model and log.
**Applied lenses:** A1/A2/A3/A4/A6/A8/A10/A11. No specialist hardware signoff.

## Verified test outcomes

- Added `tests/g2_weight_reuse_batch.py`, `tests/test_g2_weight_reuse_batch.py`.
- 15/15 focused new unit tests passed, including invalid inputs, algebraic identities and numerical output equivalence.
- 12/12 previously established G2 traffic tests passed without changes.
- Two independent full CPU benchmark runs executed: 2 matrix dimensions × 2 seeds × 6 batches × 7 paired timing repetitions per run. Raw time series preserved in JSON.
- BLAS: NumPy OpenBLAS 0.3.30, one thread. Max difference vs repeated GEMV: 0.000350952 (FP32; tolerance defined in spec).

## Conditional 8B model (not a real-model observation)

At B=32, T=8,192, **perfect shared-weight streaming**, W=16.0 decimal GB and K per request ≈1.074 GB:

- Independent per-request weight+KV traffic summed over the batch: 546.360 decimal GB.
- Shared-weight ideal aggregate traffic: 50.360 decimal GB, of which 34.360 GB KV.
- Conditional byte-volume improvement: **10.849×** relative to unbatched independent streams.
- Pure *peak-bandwidth* batched time lower bound: 10.492 ms vs individual request floor 3.557 ms. It is **incorrect** to call these GPU latency observations.
- Batch token throughput **byte-only upper-bound** improves, yet each synchronous batch needs more total time than one standalone token in the model; actual queueing delay excluded.

## Actual CPU microbenchmark (speedup > 1 means fused faster)

Median across size × seed per run; **each row shows two separate runs**, not confidence intervals.

| Batch | Fused GEMM throughput gain vs B GEMVs, run 1 / run 2 | Fused full-batch completion time vs one standalone GEMV, run 1 / run 2 |
|---:|---:|---:|
| 1 | 0.931× / 0.975× | 1.08× / 1.03× |
| 2 | 0.481× / 0.479× | 4.12× / 4.13× |
| 4 | 0.949× / 0.986× | 4.14× / 4.36× |
| 8 | 1.769× / 1.786× | 4.55× / 4.62× |
| 16 | 3.282× / 3.248× | 4.95× / 5.20× |
| 32 | 5.139× / 5.185× | 6.38× / 6.32× |

**Interpretation:** Performance crossover occurs between batch 4 and batch 8 for this specific one-thread OpenBLAS microbenchmark. B=2 is worse. B=32 wins in aggregate work but loses in per-request completion latency. No queueing or sampling modeled.

## Audit: why this does not establish novelty or GPU performance

- Batched GEMM, continuous batching, and weight reuse are established methods, documented in vLLM and 2024 OSDI material. No distinct new mechanism was defined or demonstrated.
- Distinct FP32 CPU projections do not establish actual BF16 GPU 8B decode behavior; the model uses hypothetical shape, optimistic one-stream weight/KV assumptions, and peak published bandwidth.
- BLAS narrow GEMM inefficiency can make batching at B=2 dramatically slower than separate calls; this is a backend-specific performance cliff, not a law of computing.
- The two runs are not a statistical confidence interval; CPU cache, frequency and background activity are uncontrolled.
- The CPU GEMM benchmark does not move real 16 GB weights, operate KV attention, schedule concurrent requests, or measure physical power/thermal behavior.
- No public/real pretrained checkpoint was loaded. G5 remains blocked; no architecture advancement claimed.

## Gate decisions

- **G1:** Existing known prior art; no new invention identified.
- **G2:** PASS for conditional analytic byte arithmetic only. No physical architecture feasibility demonstrated.
- **G4:** Partial PASS for reproducible, scoped CPU GEMV/GEMM comparison; not an end-to-end simulated/new-hardware comparison.
- **G5:** BLOCKED for real LLM inference with matched-quality optimized baselines.
- **G6–G12:** Not authorized / not established.

**Next discriminatory evidence:** Real model (8B-style BF16/GQA or smaller checkpoint) GPU profiler breakdown by batch, context, cache residency, attention and weight projections, p50/p95 per-token latency, energy and throughput, compared with optimized vLLM/FlashInfer under matched conditions. Without these, stop relabeling known weight reuse as an invention.

**Evidence files:** `results/g2_weight_reuse_batch.json`, `results/g2_weight_reuse_batch_repeat.json`, `specs/G2_weight_reuse_batch_contract.md`. Reproduction commands therein.