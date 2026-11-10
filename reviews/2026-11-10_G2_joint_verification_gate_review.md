# NEXUS Gate Review — Joint speculative verification and historical-KV reuse

**Version/date:** 1.0 / 2026-11-10. **Decision:** **G1 generic combination FAIL as new invention; G2 conditional operating region exists but no new mechanism is demonstrated. G5 NOT RUN, G6–G12 NOT AUTHORIZED.**

**Owner:** AI-assisted single-session engineering analysis. **Applied ACT lenses:** A1, A2, A3, A4, A6, A8, A10, A11. Independent human review absent. Hardware/energy review not done.

## 1. Key design lesson and strongest prior art

Claude's G5-R study reported failure for NX-SPH/BOX/ONE/VAL on genuine Qwen2.5-0.5B and Granite-4.0-micro-base. The G2 FFN down-only screen on 2026-11-10 also fails its ideal 2× target under fixed whole-model weights. Thus a method must improve both temporal weight use **and** long-context attention movement.

**Prior art blocks an easy novelty claim:** 2023 speculative decoding (ICML), exact IO-aware FlashAttention (2022), Hydragen shared-prefix attention, EAGLE-3 (2025), **Dustin draft-augmented sparse verification (ICML 2026)**, and SpecStream KV-streamed verification (Sep. 2026) already implement closely related principles. Sources and comparisons: `research/G1_joint_verification_prior_art_2026-11-10.md`.

## 2. Conditional model, not measured end-to-end speedup

From the archived G5-R *separate hardware* CPU evidence:
- Qwen2.5-0.5B, batch 4 / 32K, host A: attention share `f=0.5861178531211508`.
- Granite-4.0-micro-base, batch 1 / 16K, host B: `f=0.7052066730549075`.

Let a drafter propose `k` tokens with a constant independent per-position conditional acceptance `p`. The target verifies **k+1 queries** (including the pending prefix position) so there is **no free bonus-token omission**. Expected committed length per round `L=sum_{j=0}^{k}p^j`. Assume optimistically that all nonattention target weights are executed at single-token cost `(1-f)`, while historical KV attention costs `f(1+beta*k)`; draft cost is `delta*k` and control overhead is `h`, both normalized to baseline one-token target decode. Then:

`S=L/[1 + f*beta*k + delta*k + h]`.

`S>=2` iff `beta <= [L/2 - 1 - delta*k - h]/(f*k)` for `f>0`. A negative upper bound means **mathematically impossible even at perfect attention reuse**, under *this* model. `beta` is incremental multi-query attention time vs one-query attention, not a KV-read fraction. The assumed one-pass weight work for k+1 positions and constant p are *optimistic*; true batched verification can have higher compute/attention and nonlinear effects. No actual model drafter, p, delta or h was measured. Derivation and predeclared matrix in `specs/G2_joint_verification_frontier_contract.md`.

### Selected cells from *all* 336 predeclared scenarios

For k=4 and an **assumed** `delta=0.05` per draft token, `h=0`:

| hypothetical p | L expected tokens | max beta (Granite) | max beta (Qwen) |
|---:|---:|---:|---:|
| 0.7 | 2.7731 | **0.0661** | 0.0796 |
| 0.8 | 3.3616 | **0.1704** | 0.2051 |
| 0.9 | 4.0951 | **0.3005** | 0.3615 |
| 0.95 | 4.5244 | **0.3766** | 0.4531 |

At `p=0.9` for Granite, β=0.25 gives S≈2.149× **only in this optimistic analytical model**, but β=0.5 gives S≈1.569×. At β=0.30, p must exceed approximately 0.89968 for 2×; at β=0.5, even p=1 cannot reach 2× for Granite with that assumed draft cost. If actual overhead `h=0.05`, the Granite p=0.9 β limit falls to 0.2827.

With β=1, delta=h=0, `S<=1/f` asymptotically (Granite 1.418×, Qwen 1.706×); therefore weight amortization without actual reduction of attention time cannot reach 2× on these particular recorded long-context cases. This statement does **not** apply to KV-reusing speculation with β<1.

## 3. Exact numerical reference and measured local CPU microbenchmarks

A known exact dense attention equivalence test was run on local single-head float32 arrays. `serial_attention` invokes a one-query matrix-vector attention on a fixed shared prefix; `fused_attention` computes multiple queries jointly with a matrix product. Both use stable softmax. **Max absolute numerical difference in the primary run: 1.20e-7**, a numerical equivalence check, not new hardware. Shapes: T=4096/8192/16384, head dimension 64, query counts n=1/2/4/8, seed 20261110; 5 warmup, 13 timed repeats, single-thread BLAS.

**Unexpected adverse observation:** the naive Q-stationary fused layout is slower than serial for most n=2/4 cases and delivers only mixed benefits at n=8. The measured timings and incremental β (often >1) are in `results/G2_joint_verification_frontier.json`. These are CPU single-head kernels, not a pretrained run, CUDA or a comparable `beta` measurement on the G5-R CPUs. No causal suffix costs, rollback, drafter, model weights, full-model output quality or energy included. It would be invalid to put this β directly into the G5-R `f` and advertise an LLM speedup.

**Explicitly post-hoc ablation:** tested a K-contiguous matrix multiplication form, without discarding the original variant. It was slower on these tested CPU shapes; timing and exact-output comparison are preserved in `results/G2_joint_kstation_posthoc.json`, executable `tests/g2_joint_kstation_posthoc.py`. Query counts n=2/3/5/9 (to match k=1/2/4/8 proposed tokens). Post-hoc implementation choices are not preregistered results.

**Tests:** 67/67 latest pytest checks passed; acceptance Monte Carlo, inverse thresholds, adverse parameters, free-bonus accounting, numerical attention reference, and post-hoc code correctness. These do not validate GPU implementation or trained-model accuracy. The original first primary run had 54/54 tests, expanded to 66 and 67 tests; no original scenarios were removed. No formal exhaustive proof of the runtime model is claimed; algebra is independently sanity-checked by tests.

## 4. Critical missing evidence and alternatives

- **Different hosts:** old `f` values from G5-R host A/B; new kernel timings on the current container, not commensurate. No real model target-verification benchmark, no real acceptance measurements, no target draft model, no GPU, hardware or energy.
- **Unaccounted target verification work:** batched attention has k+1 query rows; extra projection and FFN arithmetic may not fit one weight pass; verification and KV-cache rollback/control consume time; draft quality varies by task and context.
- **Prior art:** Dustin (ICML 2026) already combines sparse KV selection and drafting; SpecStream (Sep. 2026) already combines streaming KV with target verification. EAGLE-3 and production vLLM serve as stronger speed baselines than the dense G5-R CPU framework.
- **GQA:** real KV pages are shared among query heads; any claimed KV page savings must use the union across heads, not a per-head average.
- **Numerics and quality:** a batched exact attention kernel preserves the attention result up to rounding; this does not establish correct speculative acceptance, distributional equivalence, or safe approximate KV selection.
- **Benchmark sensitivity:** the simple model uses a constant p and fixed beta. Bursty rejections, cache residency, query-dependent attention and clock/thermal effects can change it.
- **Training and novelty:** no unique mathematical primitive or hardware datapath differentiates this idea from cited 2023–26 work. There is no patent, review, silicon or verified invention.

## 5. Gate decisions and the next decisive evidence

- **G1:** **FAIL** for novelty of the simple combination. Do not build an allegedly new processor from it. Patentability not assessed.
- **G2:** **CONDITIONAL / IN PROGRESS** only for an *existing known technique's* optimistic latency frontier. Some hypothetical cells can cross 2×, many cannot; no measured complete baseline/quality passes.
- **G3/G4:** Reference numerical result checked on local CPU; real system path not established.
- **G5:** **NOT RUN** in this analysis; Claude G5-R failure remains for the earlier four NX candidates. A new design is not vindicated or refuted by this stand-alone math.
- **G6–G12:** **NOT AUTHORIZED**. No breakthrough demonstrated.

**Stop condition for the next stage:** require a genuinely distinct, predeclared primitive that cannot be reduced to Dustin/EAGLE-3/SpecStream/FlashAttention/standard speculation. If a candidate emerges, measure actual draft acceptance p, verifier throughput β, *total* draft/control costs on the **same pretrained model and hardware**, and compare to the strongest available implementation at matched quality. Otherwise do not initiate hardware work.

## Reproducibility and verification

Commands:
```bash
cd <package root>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/g2_joint_verification_frontier.py --output results/G2_joint_verification_frontier.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tests/g2_joint_kstation_posthoc.py --output results/G2_joint_kstation_posthoc.json
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/test_g2_joint_verification_frontier.py
```

Full machine-readable matrix, CPU timing, environment and supplementary ablation are retained in this package. Exact timing repeatability across machines is *not* claimed. Review not independently human-signed.
