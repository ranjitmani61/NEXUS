# Gate review — G5 local-to-global error propagation falsification
**Date:** 2026-11-03  
**Protocol:** AGENTS.md v1.0 read and followed; heightened numerical verification  
**Discipline lenses:** A2 mathematics/numerical, A3 ML system validity, A4 runtime oracle-cost awareness, A8 verification, A10 system-level quality. No independent engineer or hardware review.

## Decision

**FAIL (local-only certificate as a sufficient AI-output safety criterion).** Previous diagnostic is valid only as a local projection measurement, not evidence of model-level quality preservation.

**G5 remains BLOCKED:** `EleutherAI/pythia-14m` weights not locally available; Transformers library not installed; direct `huggingface.co` and `raw.githubusercontent.com` DNS lookup failed in current container. No billed compute used. Hugging Face model exists publicly; existence is not executable local access.

## Measured and derived evidence

- Existing local diagnostic marks constructed thresholded-delta update as safe: local relative L2 = `0.014998312784712243` (1.50%); threshold `tau=0.02`.
- Downstream head makes true class 1, approximate class 0; output logit relative L2 = `0.9977851578566089` (99.78%). This is a deliberately adversarial constructed 2D case, not representative-model statistics.
- Eight independent layer error injections of absolute norm 0.01 accumulate to norm 0.08 even though each per-layer error is only 0.01. A linear bound captures this correctly.
- An exact-arithmetic spectral-norm recurrence for a linear chain and top1 margin guard was specified and tested on **500 seeded random chains**. No violation above `1e-10` tolerance; maximum floating-point overrun of the mathematical bound was `7.63e-17` (rounding). 135 chains met the conservative stability guard; 129 unguarded chains changed top1 in this diagnostic; those counts are not real-model rates.
- **66/66** targeted regression tests passed, including previous G5 instrumentation and trained-toy test suites. Log: `results/g5_local_to_global_regression_pytest.txt`.
- Source and test outputs: `tests/g5_end_to_end_error_contract.py`, `tests/test_g5_end_to_end_error_contract.py`, `results/g5_local_to_global_error_audit.json`, spec `specs/G5_end_to_end_quality_safety_contract.md`.

## Scope and risks

The 500-check result is not a proof for Transformers; the bound applies only to linear layers with honest local absolute-error bounds. It does not include mixed precision, nonlinearities, stale KV snapshots, inference quality or optimized hardware runtime. The margin guard assumes knowledge of the *true* logits; model-level practicality is unproven. Our output is **a stop-rule correction**, not a new processor or speedup.

Existing studies of temporal delta, value-aware screening and lossless packing do not establish a genuinely novel computational mechanism. Prior G1 novelty concerns remain; G6 hardware not authorized.

## Next discriminating work

Find authorized genuine pretrained-model execution, then run actual *end-to-end* quality and cost measurements under the revised G5 contract. Do not repeat local-only oracle benchmarks as a claim of quality preservation. If a constrained resource-free path is needed, independently verify the linear margin bound and build fixture tests only; they cannot clear G5.