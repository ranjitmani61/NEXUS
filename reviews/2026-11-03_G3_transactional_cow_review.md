# Project 01 — Gate G3 transactional KV update review

**Date:** 2026-11-03 · **Version:** 1.0 · **Decision:** G3 IN PROGRESS; no G5/G6 authorization. 

## Direct decision

**Narrow fix confirmed:** Under the supported Python API, the replacement/append transaction ties metadata to immutable key/value pages and publishes an entire cache root atomically, eliminating the previously reproduced stale-version/mutable-array failure path. This is a *reference software design*, not proof of processor correctness or a novelty claim. Full-cache copying on each update is no longer necessary in this implementation.

## Evidence actually produced

- `94 passed in 0.24s` for the three relevant test files (existing plus new), run with `PYTHONPATH=tests OPENBLAS_NUM_THREADS=1`. The new test file contains **9 parameterized test cases at pytest collection level**; the total count is from the actual run, not an extrapolation.
- Tests cover page and metadata immutability, input alias isolation, transaction conflict and rollback, two writers racing, multi-page publication under concurrent reads, optional page append, and numerical comparison with a dense FP64 reference over stored low-precision operands.
- CPU timing with 40 updates per size, **two independent sequential repetitions**, median *update-only* time including metadata upkeep:

| Pages | Prior entire-cache copy, run 1 / run 2 | New page-level COW, run 1 / run 2 | Relative improvement over old reference |
|---:|---:|---:|---:|
| 32 | 0.725 / 0.753 ms | 0.044 / 0.047 ms | 16.35× / 15.88× |
| 128 | 2.938 / 2.919 ms | 0.062 / 0.064 ms | 47.50× / 45.53× |
| 512 | 11.920 / 12.251 ms | 0.077 / 0.081 ms | 154.38× / 151.23× |

Exact values in `results/g3_transactional_kv_benchmark_rep{1,2}.json`; rounded table values above. The measured baseline is a **naive reference full-copy implementation**, NOT a tuned production KV manager. This result must not be presented as an AI inference or hardware speedup. New updates still copy an O(P) Python pointer table and retain old roots until readers release them. Query cost was *not* benchmarked here.

## Unresolved / negative findings

1. **G3 mathematics remains open.** The guard reused from earlier work is not a formal directed-rounding interval proof. FP64 dense attention is a stored-operand reference, not a training/inference quality guarantee versus an original FP32/BF16 model.
2. **G3 concurrency remains limited.** One local threaded workload does not establish behavior on distributed host/GPU memory, DMA, partial writes, or abnormal termination. Python object freezing is not a security boundary against arbitrary reflection or native pointer mutations.
3. **G4 performance remains incomplete.** The new CPU timing only shows avoidance of a known weak previous full-copy baseline. No optimized dense/sparse attention query comparison was performed, and metadata stacking per query adds O(PD) work.
4. **G5 still BLOCKED** for pretrained model traces. G6 hardware development **NOT AUTHORIZED** because strong-baseline performance and novel mechanism are unproven.
5. **Novelty absent.** Page-granular management, prefix caches and sharing are established (vLLM/SGLang). No patent/academic originality assertion.

## Next discriminating action

Test token-granular partial-page transactional append and measure full attention-query overhead relative to an optimized paged-KV implementation. Where available, run exactly the G5 harness on actual trained model data. Reject this method as a processor direction if it fails strong baselines.

## Artifacts

- `tests/g3_transactional_kv.py`
- `tests/test_g3_transactional_kv.py`
- `tests/g3_transactional_kv_benchmark.py`
- `results/g3_transactional_kv_full_pytest.txt`
- `results/g3_transactional_kv_benchmark_rep1.json`
- `results/g3_transactional_kv_benchmark_rep2.json`
- `specs/G3_transactional_kv_cow_contract.md`

**ACT review lenses:** A1/A2/A4/A6/A8/A10/A11; no independent external specialist review.