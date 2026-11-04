# 2026-11-03 — G3/G4 token-stream coherency and end-to-end cost review

**Version 1.1 | AGENTS.md read before work | Decision: No production or hardware promotion**

## Result (observed, local CPU)
Created `tests/g3_token_stream_kv.py` for partial KV pages and immutable token append; `tests/g4_token_stream_end_to_end.py` for synthetic decode timing. Ran tests and benchmarks, not just inspected code.

**113/113 tests PASSED**, including existing concurrency/precision tests and 19 added tests of partial pages, cache boundaries, append conflicts, input aliasing, reader/writer consistency, FP16/BF16 bounds and exact output checks. No violations detected in the tested cases. This does not imply a formal concurrent-memory-model or floating-point proof.

**Benchmark (two repeated runs; two seeds per run; 64 decode steps per case; ratios of total measured time):**

| Synthetic input | Guarded page admission | Transactional page-screening / preallocated converted-key dense | Versus preallocated same-dtype dense |
|:--|:--|:--|:--|
| Clustered | 64/64 queries for each seed/run | **8.49–10.65× slower** | 4.27–5.83× slower |
| Random | 0/64 | **18.84–21.91× slower** | 10.26–13.43× slower |
| Uniform | 0/64 | **24.87–26.54× slower** | 3.92–4.15× slower |

Breakdown: clustered screening query alone was 0.51–0.64× the runtime of *our slower Python page-by-page dense query*, but with append+screening remained 8.49–10.65× slower than contiguous dense on CPU. Random/uniform screening query was 1.37–1.62× slower than page-by-page dense. This is why a favorable page-selection fraction does **not** establish end-to-end benefit.

Zero detected output errors in these cases; the deliberately clustered case is so extreme that FP64 softmax effectively assigns negligible mass to other pages. This is not representative of pretrained attention. All output comparisons used identical stored FP16 keys and FP64 value tensors. Initial construction and allocator/system costs were excluded equally; candidate initial metadata overhead would only worsen its result if included, but this statement is analytical, not measured here.

**One benchmarking correction:** exploratory v1.0 JSON incorrectly marked `initial_metadata_build_included=True` while initial `TokenStreamKV(...)` construction was outside the timer. v1.1 fixes it to `False`, retains v1.0 reports, and separates append, screening, and paged-dense query times. v1.1 results above supersede v1.0.

## Key engineering risks / boundary of evidence
1. Token-by-token incremental update now works for partial pages; atomic root publication and stale-writer rejection passed tested Python concurrency cases. GPU/DMA readers are not covered by Python's lock.
2. Candidate performs O(P) pointer-tuple reconstruction and metadata stacking for every decode query; per-token updates rewrite the tail page. Actual hardware paged KV implementations need not pay these Python costs.
3. The converted-key dense reference stores a duplicate FP64 K cache (extra memory), while same-dtype dense is also measured. This is a matched-output **but different-memory-cost** comparison.
4. No pretrained checkpoints, real-model traces, production vLLM/PagedAttention benchmark, FlashAttention kernel, GPU, energy, or silicon characterization. No novelty demonstrated.
5. Current mixed-precision screening is empirically guarded; full directed-rounding and hardware proof remain incomplete.

## Decision
- **G3:** IN PROGRESS — token-stream reference coherency checks PASSED under stated Python model; formal finite-precision and device concurrency unproven.
- **G4:** IN PROGRESS — CPU reference results reproducible; candidate measured materially slower than contiguous dense in every synthetic case.
- **G5:** BLOCKED — realistic trained-model validation inaccessible here.
- **G6–G12:** NOT AUTHORIZED; **breakthrough NOT demonstrated**.
- **Immediate next test:** avoid another invented processor claim. When a real pretrained/GPU environment exists, benchmark against production vLLM/FlashAttention with matched quality and include allocator, metadata, token throughput, and power. Until then, stop this Python performance path or work on formally verified arithmetic independent of speedup claims.

## Reproducibility and references
See `specs/G3_G4_token_stream_contract.md` for exact commands, scenarios, dimensions, assumptions, code paths and official prior-art references. Raw data: `results/g4_token_stream_v1_1_rep{1,2}.json`, `results/g3_g4_token_stream_pytest_v1_1.txt`. This report was prepared using engineering review lenses A1, A2, A3, A4, A6, A8, A10, A11; no independent expert certification.
