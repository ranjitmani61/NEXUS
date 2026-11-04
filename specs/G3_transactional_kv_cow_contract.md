# G3 transactional KV metadata-coherency contract — reference implementation

**Version:** 1.0 · **Date:** 2026-11-03 · **Status:** Narrow software validation; G3 in progress  
**ACT lenses:** A1 Architecture, A2 Mathematics, A4 Runtime, A6 Memory, A8 Verification, A10 Performance, A11 Prior art. No independent human specialist review.

## Problem and hypothesis

The previous snapshot reference copies all K/V and rebuilds all page metadata on a single-page replacement. Its old mutable version allows an untracked change to invalidate metadata and produce a catastrophically false error certificate. **Hypothesis:** storing keys, values, and their bounds together in private immutable page objects, then atomically publishing a new snapshot root with only changed pages replaced, can prevent *that particular* stale-metadata path while reducing update-copy work.

This is a test of an established persistence/copy-on-write technique, **not** a novel hardware principle.

## Software contract

1. `SealedPage.create` checks dimensions, exact key dtype (FP16/BF16), float64 value dtype, and finite elements. It owns byte-backed, read-only K/V and per-page min/max metadata.
2. The `Root` tuple of `SealedPage` objects is frozen, has an epoch, and is created through `TransactionalKV` in normal use.
3. Every read captures **one** root by lock, and all subsequent metadata and K/V are read only through that root. There is no API to pass a foreign metadata object to `guarded_query`.
4. `commit(expected, replacements, appended)` prepares/validates every new page before acquiring the publication lock. It compares `expected` by object identity to the current root and publishes exactly one new root under lock. Mismatches raise `WriteConflict`; invalid requests leave the current state unchanged.
5. An existing root never changes through the supported API. Existing readers therefore finish with their captured root, regardless of later commits. The Python reference relies on CPython/NumPy/threading behavior, not GPU/DMA/RTL semantics.
6. The update allocates one new immutable page per changed/added page, plus a new `O(P)` pointer tuple, where `P` is the number of cache pages. Payload-copy work is `O(C·B·(D_k+D_v))` for `C` changed pages of `B` tokens. The previous full-copy reference does `O(P·B·(D_k+D_v))` payload work and rebuilds all bounds. **This implementation is not O(1) per update.**
7. `guarded_query` currently stacks all page metadata per query, and calls the earlier FP32-with-guard screening function. That implementation has unproved endpoint-enclosure assumptions. `dense_stream` computes FP64 online softmax from the stored quantized operands. Neither one is a production GPU kernel or formal arithmetic proof.

## Mathematical safety argument (conditional, not a formal mechanized proof)

Let `R` be a root and `P_i(R)` the immutable page at index `i`. For all pages, page-bound metadata `M_i(R)` was generated from the stored `K_i(R)` at the same creation event. Since neither page object nor root changes under the supported API, `M_i(R)` cannot become stale relative to `K_i(R)`. A commit constructs `R'` from sealed replacements and unchanged references, and publishes `R'` in one locked assignment. Thus a reader holding `R` cannot observe a mix of old and new page objects. This does not prove that numerical score bounds are rigorous, hardware memory ordering is correct, or hostile Python reflection cannot tamper with objects.

## Test plan and observed coverage

- Input aliases cannot change already sealed pages, and attempts to toggle the NumPy array writeable flag fail.
- Unchanged page objects retain identity between revisions; changed pages get new objects.
- Multi-page updates publish together; malformed staged pages roll back without publication.
- Stale optimistic transactions fail without overwriting newer changes.
- Two simultaneous writers starting from the same root: exactly one commit wins.
- Concurrent reader with 200 two-page updates / 400 queries sees only the fully committed state in one local thread-scheduling trial.
- Matched FP64 reference attention on both FP16/BF16 stored operands, 16 update iterations per dtype; screened output checked against the declared epsilon and returned bound.
- Two timed runs over cache sizes 32, 128 and 512 pages, 40 updates each, page size 16, key dimension 64, value dimension 16. Both implementations include summary maintenance and use the same generated updates.

## Missing engineering evidence

- A formally proved, directed-rounding numerical interval contract, including subnormals, NaN/Inf behavior, cancellation and matrix kernel reorderings.
- Device-side atomics, lock-free reclamation/ABA prevention, memory fences and concurrent DMA or GPU streams.
- Token-by-token partial-page append, eviction, prefix-sharing, references/retention pressure, failure recovery and multi-layer/multi-head integration.
- Optimized paged KV baselines (vLLM/SGLang), GPU or accelerator timing, energy and memory-traffic measurements, and real pretrained-model workloads.
- Pointer-root tuple creation and metadata stacking still scale with number of pages. A persistent tree or block table would be a different comparison point.

## Prior art (not a novelty claim)

- vLLM paged blocks / prefix caching: https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html
- SGLang radix cache reference counting/prefix sharing: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py

## Reproduce

```bash
cd /mnt/data
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_g3_guarded_mixed_precision.py tests/test_g3_snapshot_consistency.py tests/test_g3_transactional_kv.py -q
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 python tests/g3_transactional_kv_benchmark.py --steps 40 --output results/g3_transactional_kv_benchmark_new.json
```
