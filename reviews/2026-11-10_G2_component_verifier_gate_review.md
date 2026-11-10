# Gate review — FFN-shaped cost of joint target verification

**Date / version:** 2026-11-10, 1.0 · **Decision:** Generic mechanism G1 FAIL for novelty; G2 corrected cost frontier, CONDITIONAL; G5 NOT RUN for this mechanism; G6–G12 NOT AUTHORIZED. **No new AI processor demonstrated.**

## 1. Corrected finding

The previous joint G2 study set verification's nonattention work to one target-token cost even for `k+1` verified queries. We measured the effect of `n=k+1` queries on a float32 SwiGLU FFN proxy with the published Qwen2.5-0.5B and Granite-4.0-micro dimensions. The study uses random matrices, not pretrained weights. The ratio `rho=batched FFN(n)/one FFN(1)` is >1 throughout the measured cases; at k=4 (five target queries):

| Shape | Seed 20261110 rho | Seed 20261111 rho | Matmul batch / separate serial, first seed |
|---|---:|---:|---:|
| Qwen (896 × 4864), 3 projections | 2.200 | 2.165 | 0.452 |
| Granite (2560 × 8192), 3 projections | 2.121 | 2.166 | 0.431 |

A batched verifier is substantially **faster than five serial FFNs**, but it is not free beyond one token. In this CPU library and shape, assuming one-token FFN latency for 5 queries is contradicted. Other BLAS implementations, precisions, GPU tensor cores, host scheduling and architectures can change rho.

## 2. Important correction to initial pilot interpretation

My initial sensitivity analysis mistakenly charged the observed FFN ratio to **all** nonattention runtime. The proper narrower question uses Claude C2's separately measured MLP share `g`. We retain both cost models and the original rho=1 optimistic model; none is a rigorous hardware bound. We explicitly correct the earlier chat's approximate β=0.07 assertion: the *re-run corrected* all-nonattention proxy gave β≤0.100, while charging FFN cost to MLP alone gave β≤0.367. Both differ from the original optimistic β≤0.558. All are **modeled β thresholds**, not measured verifier performance.

For k=4, assumed p=.9, draft delta=.05 per proposed token and h=0:

| External anchor (C2 or G5-R) | Old β ceiling (rho=1) | FFN rho charged to MLP only | FFN rho charged to all nonattention |
|---|---:|---:|---:|
| Granite C2 host C, 16K b1 | 0.558 | **0.367** | 0.100 |
| Qwen C2 host C, 32K b1 | 0.432 | **0.343** | 0.120 |
| Granite G5-R host B, 16K b1 | 0.300 | **NOT AVAILABLE (no separated MLP share)** | 0.183 |
| Qwen G5-R host A, 32K b4 | 0.362 | **NOT AVAILABLE (no separated MLP share)** | 0.150 |

The MLP-only assumption optimistically keeps other projection/normalization/control costs at one-token cost; the all-nonattention assumption pessimistically scales them like the FFN. Real multiquery target performance can fall outside these bracketed values. The C2/G5-R attention shares come from **other hosts and measurement code**; our FFN times are from this sandbox. The calculated combined speedups are *not measurements*.

## 3. Complete scenarios and robustness

- 64 explicit `f × k × p` cells retained: 4 external host anchors × k={1,2,4,8} × assumed p={.7,.8,.9,.95}, delta=.05, h=0. Before charging any extra FFN work, **44/64** cells cross modeled 2× at hypothetical beta=0. If all nonattention scales with measured rho, **20/64**. For the 32 host-C cells with separately recorded g, **17/32** cross under MLP-only scaling. These scenario counts are a screening device, **not observed model successes**.
- Two independent fixed seeds, with nine and seven interleaved repeats; **all** k and per-repeat latencies retained, no result deleted. k=8 shows more variance on Qwen; do not claim stable precise speedups across hardware. Numerical output of serial and batched FFN checked using floating-point tolerances; 28/28 regression tests passed. CPU wallclock is noisy; no confidence interval on different hosts.
- Operational intensity of weight reads can improve with batch n: if weights read only once, approximately `n/2` FLOP/byte for the three dense layers in float32, excluding other traffic. FLOPs still scale linearly with n; no free compute. Roofline suggests possible memory- to compute-bound transitions, but memory bandwidth/FLOP ceilings not measured here.
- Adversarial/negative controls: p=0 yields one token; unrealistic free bonus rejected; rho increases can destroy modeled 2×; size-shape mismatch fails; beta=0/1 and invalid `f/g` checked.

## 4. Limits and disclosure

**[Measured]** synthetic float32 NumPy one-thread FFN projection times on this CPU; **[Derived]** conditional verifier latency and required β. The shapes come from published configurations, not proprietary/trained parameters. We did **not** measure actual draft accept probabilities, target end-to-end latency, attention reuse, cache rollback, Qwen/Granite model outputs, energy, GPU/TPU, SRAM/HBM traffic or silicon. The C2 `attn_core` and G5-R timing share use different instrumentation and are not perfectly comparable. Changing BLAS threads, GPU or precision may reverse measured ratios.

**Prior art:** Leviathan 2023 speculative decoding, Roofline 2009, block-level verification 2025, Dustin 2026. See `research/G1_multitoken_ffn_priors_2026-11-10.md`. No patentability assessment or independent specialist sign-off.

## 5. Gate and next discriminating work

- **G1 novelty**: FAIL for the simple shared-weight/multiquery verifier combination.
- **G2**: Mathematical cost correction and illustrative CPU proxy completed; **no credible ≥2× end-to-end demonstration**; conditional / IN PROGRESS for real target architecture.
- **G3/G4**: limited numeric equivalence and reproducible arrays; no model-level correctness.
- **G5**: NOT RUN on this candidate, historic NX page-screening G5 FAIL unchanged.
- **G6–G12**: NOT AUTHORIZED; no processor/hardware claim.

**Next admissible test:** On *one same pretrained model, one host, one precision*, measure batched target verification `k+1` on real outputs and break down attention/FFN/projection/control times; measure actual draft acceptance, PPL/argmax and end-to-end dense/EAGLE/Dustin baselines at matched quality. Before a new invention claim, exhibit a mathematical/dataflow primitive demonstrably distinct from those methods.

Reproduction and source contracts: `specs/G2_component_resolved_verifier_contract.md`; `tests/g2_ffn_verification_cost.py`; `tests/g2_ffn_verifier_scorecard.py`; `tests/test_g2_ffn_verification_cost.py`; all machine-readable `results/G2_ffn_verifier_*` files and checksum manifest. No source code or figure from external papers reproduced.
