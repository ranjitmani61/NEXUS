# G4 Dynamic KV Stream — Engineering Gate Review

**Date:** 2026-11-03 · **Version:** 1.3 · **Gate:** G4 IN PROGRESS
**ACT lenses:** A1/A2/A3/A4/A6/A8/A10/A11 (role-based review only; independent specialists not employed).

## Verdict
Two complete synthetic CPU runs show conditional, NOT general, end-to-end improvement. All 53 pytest checks passed. No observed bound/certificate violations; no trained-model evidence or hardware results.

## Measured total cost vs dense attention
Total ratio includes vectorized initial metadata construction and per-append metadata update, across 128 decode steps per seed. Smaller than 1 means lower measured CPU reference time.

| Workload | Fixed center, run 1 / run 2 | Min/max, run 1 / run 2 | Admissions |
|---|---:|---:|---|
| clustered | 0.521 / 0.512 | 0.654 / 0.627 | 100% |
| random | 1.509 / 1.549 | 1.647 / 1.626 | 0% |
| uniform | 1.518 / 1.559 | 1.638 / 1.643 | 0% |
| needle | 1.555 / 1.521 | 1.608 / 1.634 | 0% |

## Verification and failure analysis
- **Unit tests:** 53 passed across dynamic KV property checks and prior G4/G5 harness tests; exact execution output in `results/g4_dynamic_kv_pytest_v1_3.txt`.
- **Numerical audit:** Per-page score bounds verified against all active page keys at every tested step. Each approximate output compared to exact attention using current-visible Vmax; none exceeded its numerical certificate within 2e-10 tolerance.
- **Streaming consistency:** Metadata updates occur after each append and before each tested query. A stale metadata version forces exact dense attention; out-of-order appends are rejected. This does not protect concurrent writes during a query.
- **Negative cases:** Random, uniform, and late-needle inputs triggered dense fallback on every step and were slower after preflight/update costs.
- **Positive control:** Deliberately clustered data caused 100% one-page admission; not representative of trained LLM inference and insufficient evidence of innovation.
- **Correction:** v1.1 used slow Python iteration to build metadata and overstated initial costs; v1.2 vectorized builds and separated only needed data per mode. v1.3 corrected offline error normalization to use the maximum norm among *visible* values, not future values. Earlier raw results remain in `results/` as the correction record.

## Gates
- **G3: IN PROGRESS** — sufficient exact-arithmetic bound and numerically verified implementation, but no formal finite-precision or concurrency proof.
- **G4: IN PROGRESS** — two reproducible local CPU comparisons and adverse controls; not a competitive optimized accelerator baseline.
- **G5: BLOCKED** — no representative pretrained model traces or optimized dense/Quest baseline.
- **G6–G12: NOT AUTHORIZED/PASSED** — nothing here warrants RTL/tapeout or a breakthrough claim.

## Strongest relevant prior art
Quest (Tang et al., ICML 2024): https://proceedings.mlr.press/v235/tang24l.html ; https://github.com/mit-han-lab/Quest . Its KV-page min/max summarization and query-aware selection are directly relevant; novelty **not demonstrated**.

## Reproducibility
Detailed parameters, proof assumptions, cost definitions, commands, and limitations in `specs/G4_dynamic_kv_stream_protocol.md`. Source `tests/g4_dynamic_kv_stream_benchmark.py`, unit tests `tests/test_g4_dynamic_kv_stream.py`, run output `results/g4_dynamic_kv_stream_v1_3_rep1.json` and `rep2.json`.

## Next justified work
Only further if pretrained-model evidence becomes available: collect real trained-model KV traces, compare a competitive optimized dense kernel and Quest under matched model/task quality, measure p50/p95 latency and update costs. If resource inaccessible, document G5 blocker rather than making a new hardware claim.
