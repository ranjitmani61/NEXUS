# 2026-11-03 — G4 compiled CPU gate review (v1.0)

**Decision:** **Do not proceed to hardware.** Compiling removes the dominant Python boundary overhead in favorable cases, but does not establish a new processor mechanism or a competitive advantage on realistic workloads. **G3/G4 IN PROGRESS; G5 BLOCKED; G6–G12 NOT AUTHORIZED.** Prior-art status: the specific page-extrema screening idea is already present in Quest (2024).

**Governance:** Read `AGENTS.md` v1.0 and prior G3/G4/G5 results. Review lenses: A1 architecture, A2 numerical analysis, A3 AI systems, A4 runtime, A6 memory, A8 verification, A10 performance, A11 prior art. No external professional engineering or independent validation.

## What was actually executed

- Produced compiled C++17 attention baseline and fused screening/fallback test on a local Intel Xeon CPU. Same stored FP64 data and task-quality definition; metadata update and dense fallback included in timed candidate, initial metadata build added to `cold_ratio`. No GPU, silicon, energy, power, model weights, or external cloud jobs.
- 1,024-token, 4,096-token and 16,384-token synthetic cases, with both favorable and unfavorable inputs, were executed. Repeated 4,096-token runs were CPU-affinity pinned. There are **96 raw case rows covering 8,928 query comparisons**, with **zero recorded error-bound/tolerance violations** in the tested conditions. The ASan/UBSan build passed the small 96-query set. All 113 existing G3 Python tests and 11 restored G5 harness tests passed (124 total). G5 harness tests are unit tests with mock data, **not** a pretrained model run.
- The `1e-8` logit margin is a heuristic FP64 pad, not a formal directed-rounding bound. Non-detection of errors is not proof of numerical correctness for arbitrary data or hardware.

## CPU results — careful interpretation

Pinned 4,096 initial-token tests: 3 repeated experiments × 2 seeds × 128 appended tokens × 6 workloads. Ratios include one-time metadata construction, lower is faster **only relative to the hand-coded FP64 CPU baseline**:

| Synthetic workload | Queries admitted | Median candidate/dense total time | Range across six runs |
|---|---:|---:|---:|
| First-page strongly clustered | 768/768 | 0.061× | 0.057–0.070× |
| First-page moderately clustered | 768/768 | 0.058× | 0.032–0.146× |
| Near certificate boundary | 0/768 | 1.053× | 0.928–1.084× |
| Random keys/queries | 0/768 | 1.002× | 0.957–1.128× |
| Near-uniform logits | 0/768 | 1.063× | 0.999–1.164× |
| Rare late-important key | 0/768 | 1.042× | 1.001–1.198× |

The favorable ratios show that a **compiled synthetic screening loop** can avoid the dominant dense calculation when the first page is artificially obvious. They are **not** hardware speedups, model-token throughput, general processor efficiencies, or evidence of novelty. Timing noise strongly affects ratios when fallback runs the same dense code; a small favorable fallback ratio is not evidence that doing extra screening speeds up dense attention.

**Scale failure:** The moderately clustered case was admitted **192/192 at 1,024**, **384/384 at 4,096**, but **0/128 at 16,384** in the non-pinned scale matrix. The mass bound worsens as more omitted pages accumulate. Even a fixed score gap becomes insufficient at larger contexts: $Z_o/Z_s \sim ((N-B)/B)e^{-\Delta}$ for page size $B$ and logit separation $\Delta$.

**Correctness scope:** Max observed normalized error on the admitted moderate cases in the pinned matrix was below 0.00039; no tested admission overshot the computed certificate. The chosen certificate is conditional in exact arithmetic. No general rounding proof, finite-precision end-to-end proof, production concurrency verification, sustained memory measurement, or software security audit.

## Falsification, limitations and audit concerns

1. **Novelty is not established:** Quest (2024) already uses query-aware min/max page summaries. This exact mechanism is not an original processor architecture; adding page-level COW would also overlap vLLM's established memory management.
2. **Unrepresentative workload:** The candidate always selects the first page and is evaluated with intentionally biased synthetic queries; it is not a general model of attention page locality. The 16K moderate failure demonstrates selectivity fragility.
3. **Baseline weakness:** The baseline is a hand-coded C++ FP64 dense CPU loop, not optimized production attention, BLAS, Quest, FlashAttention, or GPU. A 0.06× ratio **does not** establish advantage over those systems.
4. **Accounting:** One-time metadata initialization is included; initial filling of KV storage is excluded on both sides. Kernel/query level only. Other model operations, memory placement, thread scheduling, OS activity, peak allocations, and all energy costs are unmeasured.
5. **Benchmark noise:** Non-admission cases scatter from ~0.93× to ~1.20× on the pinned configuration despite computing the same dense attention, so small timing differences are not trustworthy. Individual synthetic timings are not statistically independent repetitions of a full workload.
6. **Correctness boundary:** The mutable compiled cache does not implement immutable transaction publication. No hardware or concurrent safety claim follows from the single-thread test.
7. **Model data:** Hugging Face checkpoint cannot be downloaded from this runtime (DNS failure observed when requesting a public Pythia config), and no local checkpoint is cached. I did not launch paid cloud compute. Thus G5 remains blocked rather than passed by synthetic proxy.

## Gate decisions

- **G1 (specific candidate distinctness): REJECT as novel architectural proposal** — min/max page screening already exists; full patent/market landscape beyond scope.
- **G3: IN PROGRESS** — exact-arithmetic conditional bound plus tested FP64 reference; rigorous floating-point, concurrency and device proof missing.
- **G4: IN PROGRESS** — a compiled CPU falsification test is reproducible; no competitive baseline or representative model.
- **G5: BLOCKED** — pretrained-model and representative trace validation missing.
- **G6+: NOT AUTHORIZED** — do not design RTL/ASIC from this evidence. **G12: NOT DEMONSTRATED.**

**Next discriminating action:** Acquire real pretrained Q/K/V or actual full-model traces and test against Quest and optimized dense kernels with equal task quality and full costs. If that access remains unavailable, halt this candidate as an invention and research a *distinct* mechanism instead of rebranding page screening. This is a research recommendation, not a claim of background work.

**Raw evidence:** `results/g4_compiled_cpu_summary.json`, `results/g4_compiled_cpu_*.csv`, `results/g4_compiled_context_pytest.txt`, `results/g5_harness_restored_pytest.txt`; source `tests/g4_compiled_cpu_reference.cpp`; reproducer `scripts/g4_compiled_reproduce.sh`; contract `specs/G4_compiled_CPU_falsification_protocol.md`.
