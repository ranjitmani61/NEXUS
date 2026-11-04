# G4 gate review — CPU batching queueing and KV-cost sensitivity

**Date:** 2026-11-03  | **Gate:** G4 remains IN PROGRESS; G5 BLOCKED; G6+ NOT AUTHORIZED. **Novel computing mechanism: NOT demonstrated.**

**Protocol:** `AGENTS.md` read in full before work (195 lines); prior `reviews/2026-11-03_G2_G4_weight_reuse_review.md`, `specs/G2_weight_reuse_batch_contract.md`, and prior raw benchmarking results inspected. This diagnostic addresses an omitted queueing cost, not an original hardware mechanism.

## Evidence and test outcomes
- Two new local 4096×4096 FP32 service-time calibration runs, 32 batch sizes each, 5 timed repetitions/batch, one OpenBLAS thread. New code captures full raw times for audit.
- 240 original simulated queue cases in each run: 2 arrival types × 4 offered rates × 3 seeds × 5 policies × 2 timing calibrations = **240 total cases** (not 240 per run). Each case has 1,200 requests. This is 288,000 simulated request completions per version.
- Second full calibration corrected a weak numerical self-comparison. All 32 fused-output columns now cross-checked against 32 independent GEMV calls; maximum absolute differences 0.000290 and 0.000275, passed predeclared tolerance (rtol 6e-4, atol 2e-3). First calibration is preserved for measurement variability, but only second has that stronger check.
- 17/17 queue-simulator tests passed, including 150 randomized small-case differential comparisons against a separate simple reference; 3/3 sensitivity tests, plus 15/15 earlier weight-reuse tests and 12/12 earlier memory-budget tests: **47/47 test methods passed**. Unit tests do not replace representative system workloads.
- 240 supplemental sensitivity cases per calibration with an *assumed* linear per-request KV cost h, not measured KV reads. All are saved in JSON.

## Main finding: tail latency depends on offered load and service curve

Median across arrival seeds and two CPU calibrations; numbers below are from two separately executed benchmark calibrations, NOT confidence limits:

| Poisson offered load | Serial p95 range (ms) | Greedy max-32 batch p95 range (ms) | Finding |
|---|---:|---:|---|
| 150 requests/s | 5.2–5.9 | 14.0–19.8 | batching raises delay |
| 300 requests/s | 10.3–12.7 | 21.4–28.7 | batching raises delay |
| 450 requests/s | 57.5–209.5 | 22.9–35.0 | batching protects a nearly saturated/overloaded serial queue |
| 650 requests/s | 818.9–1,002.2 | 24.1–36.4 | serial queue unstable in this finite trace; batching queues less |

The differences in absolute p95 between calibrations are **material**; do not cherry-pick the first run. At 650/s this is a *finite-horizon overloaded queue*, not a steady-state latency estimate. Greedy max-8 and waiting-window policies were also evaluated, not just the two shown above; their raw results are saved.

## Conditional KV-cost falsification
For the *second* calibration, if a **hypothetical** additional independent 2 ms of KV/attention work per request is charged (without beneficial overlap):
- Poisson 150/s: serial p95 ≈18.6 ms versus batch ≈43.1 ms (batch still worse).
- Poisson 650/s: serial p95 ≈3,282 ms versus batch ≈1,096 ms. Both are severely queued, despite relative batch advantage.
- This penalty does NOT measure memory traffic or LLM compute. It is only a sensitivity perturbation to clarify how the conclusion depends on unmodeled costs.

## Failure mechanisms and uncertainty
1. **Small-batch kernel cliff**: fused GEMM at B=2 or B=3 can take far longer than one GEMV. Naive schedulers choose these kernels with little traffic and hurt low-load latency. An optimized production scheduler could switch kernels or batch differently; this model does not implement a strong competing scheduler.
2. **Calibrations vary substantially** across runs. Temperature/frequency/cache and environment are uncontrolled. No power or energy evidence.
3. **The stream of requests is synthetic**: one independent projection per request, fixed N=4096, no prefill, sampling, KV memory, variable sequence lengths, GPU throughput, token-to-token dependencies, or actual model quality. Time-to-first-token, time-between-tokens and end-to-end model latency were NOT measured.
4. **Penalty assumptions are hypothetical.** Adding `B*h` is conservative for independent nonoverlapping work but might overstate cost if overlapped or understate cost if paging/computation is expensive.
5. **No novelty**: shared-weight batched GEMM and dynamic scheduling are established in vLLM and Sarathi-Serve; newer queue-aware work also exists. A successful scheduler experiment does not prove a fundamentally new processor.
6. **G5 remains blocked:** pretrained weights still unavailable in local runtime; no paid cloud jobs launched. Hardware not authorized.

## Gate and next decision
**G4 (scoped CPU projection and simulated scheduling):** Test executed; limited findings reproducible. **G4 overall: IN PROGRESS, not PASS** because no competitive production baseline and no representative model. **G1 novelty: FAIL for conventional batching.** **G5: BLOCKED. G6–G12: not authorized.**

Stop promoting batching as an invention. Required for a hardware go/no-go: real pretrained-model profiling on a real GPU/CPU runtime, including per-layer weight versus KV traffic and p50/p95/p99 latency across arrival rates, quality, energy, and matched baseline against vLLM/Sarathi/FlashInfer. Alternatively formulate and falsify an architecturally distinct mechanism before spending on custom hardware.

## Files
- `tests/g4_queue_aware_batching.py`, `tests/test_g4_queue_aware_batching.py`
- `tests/g4_kv_penalty_sensitivity.py`, `tests/test_g4_kv_penalty_sensitivity.py`
- `results/g4_queue_aware_batching.json`, `results/g4_queue_aware_batching_verified.json`
- `results/g4_kv_penalty_sensitivity_verified.json`, summary text files
- `specs/G4_queue_aware_batching_contract.md`
- cited papers/links in specification
