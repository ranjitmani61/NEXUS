# G3 — KV snapshot and metadata coherency contract

**Version/date:** 1.0 / 2026-11-03  
**Gate:** G3 IN PROGRESS; G5 BLOCKED; G6 NOT AUTHORIZED  
**ACT engineering review lenses:** A1 computer architecture, A2 numerical analysis, A4 compiler/runtime, A6 memory/interconnect, A8 test/verification, A10 performance. These are analysis roles, **not independent human review**.

## Falsifiable claim and scope

A page-key extrema certificate can remain logically valid after a KV update *only if* the key/value data observed by a query and the metadata used to certify skipping refer to a consistent snapshot. The existing mutable `PageCache` exposes `.K`, `.K64`, `.V` and `.version` independently, and `PageMetadata` validates only `meta.version == cache.version`. It does not enforce that data mutation increments `.version` or that `.K` and `.K64` stay coherent.

The experiment reproduces a silent incorrect admission when both key views are changed but the version is not. It then constructs immutable array snapshots backed by owned read-only bytes and unique per-process version IDs. Snapshot replacement / append constructs a new object and new metadata. Old metadata is rejected on new snapshots. **This is a correctness reference, not an efficient update mechanism or physical concurrency design.**

## Counterexample, two floating-point formats

- Pages: 2, tokens/page: 4, head dimension: 16, scalar value dimension: 1; stored keys and query use FP16 or BF16.
- Initially page 0 keys are all +5 and values 0; page 1 keys are all −5 and values 1; query is all +1. Initial metadata correctly identifies page 0 as dominant.
- Mutate page 1 keys to all +20 in the live cache and its FP64 reference view **without incrementing `.version`**; leave the cached metadata unchanged.
- Existing admission returns `admitted=True`, nominal error bound `8.49735678396593e-18`, actual normalized error `1.0` relative to current dense attention. Declared threshold: `0.02`.
- The nominal bound is **not a valid mathematical guarantee** after violation of the metadata-consistency precondition. It is reported to expose the false-safety behavior of the implementation, not to refute the underlying correct conditional theorem.

## Prototype snapshot invariant

1. `FrozenSnapshot.from_data` creates private byte-backed, non-writeable copies of K, K64 and V; modifications of original source arrays do not affect the snapshot. `FrozenMetadata` similarly freezes extrema arrays.
2. Each new snapshot receives a distinct process-local ID; reusing metadata from another snapshot is rejected as stale, including cross-snapshot metadata with identical shapes and contents.
3. `replace_page` and `append_page` do not modify the previous snapshot; they build new complete K/V buffers and require a new metadata object. Queries on old snapshots keep seeing old consistent data.
4. `safe_snapshot_query` rejects nonfinite Q, mismatched Q dtype or shape, and untrusted snapshot/metadata types instead of falling back and emitting NaN.
5. Under the existing valid, finite, stored-operand assumptions, attention accuracy continues to rely on the existing conditional FP32 interval arithmetic. **Its endpoint rounding remains unproved**, and this prototype does not certify arbitrary concurrent processes, device writes, shared memory, DMA, user bypass, or fault injection.

## Exact reproduction

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_g3_snapshot_consistency.py \
  tests/test_g3_guarded_mixed_precision.py \
  tests/g3_reduced_precision_certificate_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  tests/g3_snapshot_consistency_audit.py --output results/g3_snapshot_consistency_audit.json
```

## Outcomes

- Expanded suite: **94 passed**, including prior G3 checks and new snapshot tests (`results/g3_snapshot_consistency_pytest.txt`). This is not independent hardware validation.
- Legacy counterexample reproduced in **both FP16 and BF16**, with identical normalized error 1.0 and nominal bound about 8.50e-18.
- Six seeded snapshot stress runs × 64 page replacements = **384** updates; 384 stale metadata rejections; 0 reported numerical violations under tested inputs. These random cases all fell back to dense attention, so they **do not test admission under mutated favorable distributions** (the deterministic clustered update does).
- Measured median **1.50 ms** for a complete copy and metadata rebuild on this local Python/NumPy CPU setup (12 trials; 64 pages × 16 tokens, dimension 64; stored K+V input arrays total 262,144 bytes). Single-system microbenchmark, not an accelerator prediction.
- Copies scale as O(N·d) data movement per update, unlike the O(page-size·d) ideal affected-page update. Therefore this conservative implementation **cannot be recommended as an efficient streaming KV-cache architecture**.

## Next hardware-oriented verification obligations, not yet satisfied

- Design an API with protected, transactional updates; demonstrate no query sees a mixture of K/V and summaries under concurrent writes and reordering, including ABA/version wrap-around.
- Provide formal finite-precision endpoint rounding and token-count correctness across variable-length pages and batch/head variants.
- Compare a genuinely optimized copy-on-write per-page snapshot, epoch-based reclamation, or another versioned-buffer design at production scale and mixed precision; account for update bandwidth, lifetime management, tail latency, and energy. These techniques are not proposed as novel inventions.
- Obtain **pretrained 8K+ traces from at least two model families** and fair optimized attention baselines before G5 or a hardware authorization.

## Gate decision

**G3: IN PROGRESS.** The old mutable API fails the stated metadata-consistency requirement. The new snapshot mechanism passes limited tests but requires costly full copies and does not solve formal arithmetic, process/device concurrency, or pretrained workload validation. **G5: BLOCKED. G6+: NOT AUTHORIZED. G12: NOT DEMONSTRATED.**