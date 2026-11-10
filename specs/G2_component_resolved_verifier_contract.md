# G2 Component-Resolved Cost of Multi-Token Target Verification

**Version / date:** 1.0 / 2026-11-10. **Owner:** AI-assisted NEXUS research. **Gate:** G2 feasibility screen, NOT G5. **Engineering ACTs:** A1, A2, A3, A4, A6, A8, A10, A11; no independent human or device review. Governing `AGENTS.md` v1.0 read in full. No hardware/RTL authorization.

## Problem and prior gate

NEXUS's earlier G2 joint verification study assumed that all nonattention target work for `k+1` verification queries cost as much as **one** target token (normalised cost `1-f`). That is a strong, optimistic and untested memory-bandwidth assumption. This study measures **a three-projection SwiGLU FFN stand-in** over `k+1` queries and assesses its effect on the optimistic speculative-verification frontier. The prior candidate is **not novel** (2023 speculative decoding; 2025 block verification; 2026 Dustin). This work does not claim a new computing mechanism.

### Mathematical decomposition

Let `f` be the fraction of a one-token reference decode spent in attention; `g` the fraction spent in the MLP/FFN; `g <= 1-f`; `rho(n)` the measured ratio `batched_FFN(n queries) / single_FFN(1 query)` in **this independent float32 CPU microbenchmark**. Let `beta` be the incremental target attention cost per additional verification query, relative to one-token attention time, *not measured here*. Draft length `k`; verifier queries `n=k+1`; draft acceptance `p`, assumed constant and independent per proposal; draft cost per proposed token `delta`; control overhead `h`, both in reference single-token units.

`L = 1+p+...+p^k` expected committed output tokens per round (includes possible bonus), consistent with actual target verification of `k+1` queries.

Old optimistic cost: `V_ideal = 1 + f*beta*k + delta*k + h`.

**MLP-only sensitivity:** `V_mlp = 1 + g*(rho-1) + f*beta*k + delta*k + h`. The other nonattention components are optimistically charged one-token cost. This is a model, **not a hardware lower bound**.

**All-nonattention-scaled sensitivity:** `V_all = 1 + (1-f)*(rho-1) + f*beta*k + delta*k + h`. This assumes every nonattention component scales like the FFN proxy and can be overly pessimistic, **not a hardware upper bound**. True runtime may lie outside these illustrative scenarios.

`Speedup = L / V`. For cost-scaling time share `g_s`, the beta threshold for modeled 2x is:

`beta_max = (L/2 - 1 - g_s*(rho-1) - delta*k - h)/(f*k)`, f>0 and k>0.

A negative threshold means 2x is impossible in **that particular cost model even with beta=0**. `beta>1` is possible on real CPU; do not silently clamp measured beta. The FFN cost ratio does not itself measure beta or any real target-verification speedup. No statistical claim is made from finite microbenchmark repeats.

### Fixed controlled and external inputs

- Reference `k={1,2,4,8}`; assumed `p={0.7,0.8,0.9,0.95}`; `delta=0.05`, `h=0`, `beta` displayed at 0 and 0.25; 64 modelled cells (4 externally measured attention profiles × 4 k × 4 p). These acceptance/draft costs are assumptions, **not real model measurements**.
- Single-head/FFN proxy uses Qwen2.5-0.5B geometry **hidden=896, intermediate=4864**, three dense projection matrices; Granite-4.0-micro-base geometry **hidden=2560, intermediate=8192**, three projection matrices. Float32 synthetic random matrices with seeds 20261110 and 20261111. Geometries checked against public model configurations; **not official checkpoint weights**. See sources below.
- Measure `n=k+1` queries. `serial_FFN`: one-query GEMV repeated, `batch_FFN`: GEMM across n queries; `single_FFN`: one-query GEMV. Same matrices and input samples. Numerically check batch vs serial at `rtol=1e-5, atol=1e-5`, 9 reps seed1, 7 reps seed2, 2 warmups, interleaved invocation order. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for all runs; report all individual ms arrays.
- Source attention fractions (prior G5-R / C2 external studies, **not measured here**): Granite host B `f=0.7052066730549075` (16K batch1); Granite host C `f=0.3797642644870673, g=0.2590455626976019` (16K batch1); Qwen host A `f=0.5861178531211508` (32K batch4); Qwen host C `f=0.49043314352149003, g=0.14626117892390306` (32K batch1). G5-R host B and C2 host C have different instrumentation; do not infer cross-host causality.
- For hosts without a separately measured `g`, leave MLP-only result **null**, never fabricate it. MLP-only scenario computed for exactly 32 of the 64 cells.
- No measured energy, GPU, DRAM/HBM counters, real draft acceptance, or target-verifier end-to-end performance.

### Evaluation rules

1. Any median `rho>1` refutes treating FFN batched n-query cost as one token *in this implementation*; it does **not** prove a universal lower bound on latency or other hardware.
2. Compare both `rho=1` optimistic old model and component-resolved `g` model, plus deliberately pessimistic all-nonattention scaling; never substitute the latter for the primary component-resolved comparison.
3. Preserve both seeds, all n, full 64 cells, adverse timings and fail cases. Report numerical agreement and uncertainty, not fabricated speedups.
4. G1 generic speculation / GEMM memory reuse is prior art; no novel mechanism. G2 conditional only. G5 not executed on these speculative hypotheses. G6+ unauthorized.

### Code and reproduce

```bash
cd <package-root>
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tests/g2_ffn_verification_cost.py --seed 20261110 --repeats 9 --output results/G2_ffn_verifier_cost.json
python tests/g2_ffn_verification_cost.py --seed 20261111 --repeats 7 --output results/G2_ffn_verifier_seed2.json
PYTHONPATH=tests python tests/g2_ffn_verifier_scorecard.py
PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/test_g2_ffn_verification_cost.py
```

The primary benchmarking experiment was defined by script constants before this gate review; numerical model corrections were made after the initial pilot. **This is exploratory, not a pre-registered G5 comparison.** Both pilot and corrected interpretation are disclosed in the dated review.

### Source/provenance

- Qwen2.5-0.5B model configuration: https://huggingface.co/Qwen/Qwen2.5-0.5B/blame/main/config.json
- IBM Granite-4.0-micro-base model configuration: https://huggingface.co/ibm-granite/granite-4.0-micro-base/blob/main/config.json
- Roofline model (Williams, Waterman, Patterson, 2009): https://escholarship.org/uc/item/5tz795vq
- Speculative decoding, ICML 2023: https://proceedings.mlr.press/v202/leviathan23a.html
- Block verification, 2025: https://research.google/pubs/block-level-verification-accelerates-speculative-decoding/
- NEXUS prior G5-R evidence: https://github.com/ranjitmani61/NEXUS/tree/8510d89980816e06a511ab0d665188cbe7da3d6c/results/g5r
- NEXUS Claude C2 per-operation shares: https://github.com/ranjitmani61/NEXUS/tree/922dc8f3b5639f4d5c6297da8f698f9b4580a149/results/c2
- NEXUS previous joint G2 model: research branch `research/g2-joint-verification-frontier-20261110`.
