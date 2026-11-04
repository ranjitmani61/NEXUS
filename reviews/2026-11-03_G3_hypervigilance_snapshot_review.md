# Engineering review — G3 hyper-vigilance / cache coherency

**Date:** 2026-11-03 · **Version:** 1.0 · **Decision:** FAIL old mutable-cache coherence; hold G3; no hardware authorization.

## Evidence-first verdict

**Confirmed failure:** Existing `PageCache` permits changing K/K64 while metadata remains at the same version. Both FP16 and BF16 reproduce nominal bound `8.49735678396593e-18` despite true normalized error `1.0`, versus tolerance `0.02`. The conditional math does not apply after untracked mutation. This failure is not proof the mathematical mass bound is wrong.

**Mitigation tested:** Read-only, private byte-backed K/V snapshots; immutable metadata; distinct process-local versions; explicit replace/append returning new snapshots; strict Q validation. With this prototype, old metadata for a new snapshot is rejected, and the prior snapshot remains unchanged. `94` tests passed. A separate script ran `384` cache-update steps with `384` stale rejections, plus deterministic post-update correctness checks.

**Performance regression:** Full-cache copies and summary rebuilds cost a median `1.50 ms` in 12 local CPU runs of one configuration (source: `results/g3_snapshot_consistency_audit.json`). This solves a narrowly defined safety concern at the expense of unacceptable asymptotic update work for large KV streams. No production performance claim.

## Risk register

| Risk | Status | Next requirement |
|---|---|---|
| Untracked mutable KV causes invalid certificates | **FAIL in original API** | Replace raw exposed mutations with enforced transactional API |
| Stale or cross-snapshot metadata | **Mitigated in limited test** | Explicit epoch/transaction semantics in compiled implementation |
| Nonfinite Q or wrong precision produces unvalidated attention | **Rejected by safe wrapper** | End-to-end input-contract integration |
| FP32/FP64 endpoint enclosure | **UNPROVED** | Directed-rounding or rigorous interval arithmetic |
| Concurrent GPU/DMA writes or epoch rollover | **UNTESTED** | Stress/model-check protocol and atomics specification |
| Full-copy update cost | **MEASURED, unfavorable** | Affected-page copy-on-write with snapshot reclamation |
| Pretrained LLM end-to-end utility | **BLOCKED** | Obtain representative model traces and optimized baselines |
| Processor novelty and ASIC viability | **NOT DEMONSTRATED** | Gate G1/G5 before G6 |

## Reproducibility and audit files

- `tests/g3_snapshot_consistency_audit.py`
- `tests/test_g3_snapshot_consistency.py`
- `results/g3_snapshot_consistency_pytest.txt`
- `results/g3_snapshot_consistency_audit.json`
- `results/g3_snapshot_consistency_stdout.txt`
- `specs/G3_snapshot_coherency_contract.md`

**Engineering lenses:** A1, A2, A4, A6, A8, A10. No independent specialist validation occurred.

## Decision

Keep attention-screening hardware on hold. Do not mask this failure by merely increasing unit-test count; consistency is a design contract that must be enforced across compiler, runtime, cache subsystem and eventual hardware. This snapshot reference is correctness infrastructure only, **not a new processor architecture**.