# G2 — Joint draft-verification / shared-KV feasibility contract

**Version:** 1.0, 2026-11-10. **Owner:** AI-assisted Project 01, research gate only.
**Applied ACT lenses:** A1 architecture; A2 mathematics/numerics; A3 pretrained workloads; A4 runtime; A6 memory; A8 verification; A10 systems; A11 prior art. No independent human/physical review.
**Question:** Can existing long-context CPU timing anchors support a credible *conditional* path to 2× with multi-token draft verification plus reused attention/KV traffic, before any new architecture claim?
**Baseline:** Dense, exact, one-token-at-a-time target decoding (the matched-precision Qwen/Granite baselines measured by the separate G5-R study, immutable source `ranjitmani61/NEXUS@8510d89980816e06a511ab0d665188cbe7da3d6c`, `results/g5r/summary.json`). These timings are **external prior AI-assisted evidence**, not remeasured here.

## Predefined mathematical model

Let `f` be the **measured fraction of one-token dense decoding time attributable to attention**, in [0,1]. One unit equals the one-token target decode time on that same prior CPU experiment. `k>=1` is draft length; the target verifier processes **n=k+1 query positions** (including the pending prefix/bonus position), but emits a random 1..k+1 committed tokens. The per-position conditional acceptance `p` is **independent and constant** only within this illustrative model. Expected committed output tokens:

`L(p,k) = sum_{j=0}^k p**j`.

Model optimistic target-verification cost:

`V/C = (1-f) + f*[1 + beta*k] + delta*k + h = 1 + f*beta*k + delta*k + h`.

`beta` is **per-additional-query incremental shared-history attention latency as a fraction of single-query attention latency**. `beta=1`: no time saved relative to separate historical attention queries; `beta=0`: perfect reuse, with zero incremental work for each additional query (physically unrealistic if compute-bound). `delta` is draft-model cost per proposed token relative to one target token; `h` is additional verification/control/rollback overhead relative to one target token, optimistically allowed to be zero. This model assumes all nonattention target work for `k+1` verified positions costs exactly one standard token; an optimistic bound, **not true on every GPU, CPU, batch size, or compute-bound kernel**.

`Speedup = L / (1 + f*beta*k + delta*k + h)`; reaching `S>=2` requires
`beta <= (L/2 - 1 - delta*k - h) / (f*k)` when `f>0`. A negative right side means impossible even at `beta=0` in this model. A bound `>=1` does not require KV reuse in this model. For `beta=1, delta=h=0`, the asymptotic ceiling for `f>0` is `1/f` as `k -> infinity`, **not** a universal theoretical limit when verification can reuse attention.

Important: `p` is a *proposal acceptance probability*, **not** the KV-page admission fraction and **not measured** here. The bonus-token count is balanced by processing `k+1` queries in verification; no free token is created. This is only a first-order weighted-latency model: it omits nonlinear GEMM scaling, tree verification, real acceptance variation, draft quality, KV append/rollback, resident KV cache, IO vs compute ceiling, scheduling, GPU effects, tensor shape effects and data quality. Consequently `beta` from a separate CPU microbenchmark cannot be directly inserted into G5-R `f` as though it were a real target-model speedup.

## Frozen inputs / negative controls

- G5-R: Qwen2.5-0.5B at batch 4 / 32K, host A `f=0.5861178531211508`; Granite-4.0-micro at batch 1 / 16K, host B `f=0.7052066730549075`. Both taken from G5-R summary, not this work.
- `k={1,2,4,8}`; `p={0,0.5,0.7,0.8,0.9,0.95,1}`; `delta={0,0.05,0.10}`; `h={0,0.05}`; `beta={0,0.25,0.5,1}` for scenario matrix. All cells retained (no cherry-picking).
- CPU attention microbenchmark: single-head, float32 exact dense softmax; `T={4096,8192,16384}`, `d=64`, `n={1,2,4,8}`, seed 20261110; fixed K/V/Q across modes; 5 warmup and 13 repeats, `OPENBLAS_NUM_THREADS=1`, on the *current sandbox* (not same G5-R host). Historical prefix only, no causal suffix. Compute separate single-query GEMV vs multi-query GEMM, compare the actual output for each query. Report medians and p10/p90, not an energy or full-model result. Its `beta` is an **illustrative CPU-kernel measurement** and **does not prove joint-model speedup**.
- Separate mathematical correctness tests: monotonic acceptance, no-reuse ceilings, inverse beta threshold, numerical stability, stochastic acceptance simulation at fixed seed, shape/output comparison; deliberate false-free-bonus/negative-time counterexample.

## Stop/decision rules established before results

1. **G1:** Any idea consisting simply of draft verification + shared-prefix KV reuse / sparse KV verification is **not novel** if direct prior art overlaps (see research review). A successful `beta` is not invention evidence.
2. **G2:** Exclude an operating point if it cannot reach 2× even under beta=0 (negative beta ceiling). Show `p, k, delta, h` thresholds, and contrasts with beta=1. If some optimistic cells can reach 2×, label **CONDITIONAL / NOT VERIFIED**, not G2 general PASS.
3. **G3/G4:** Check exact-reference CPU numerical equivalence and timing repeatability; do not claim batched attention is new.
4. **G5:** **NOT RUN** on genuine pretrained models. Existing G5-R failure of NX-SPH/BOX/ONE/VAL stands, and this study cannot clear it.
5. **G6+:** NO AUTHORIZATION.

**Reproduce** from the project directory:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/g2_joint_verification_frontier.py --output results/G2_joint_verification_frontier.json`
`PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_g2_joint_verification_frontier.py`

No paid compute, external actions, model weights, or GPU were used in this numerical screening.
