# G0 Benchmark Contract — Project 01

**Version/date:** 1.0 / 2026-11-03  
**Gate:** G0 — Charter  
**Review lenses:** A1 architecture, A2 mathematics, A3 AI/ML, A4 compiler, A6 memory, A8 verification, A10 performance, A11 prior art. No independent specialists have reviewed this contract.

## 1. Falsifiable objective

Determine whether a query-adaptive, error-bounded screening mechanism can **reduce end-to-end inference cost** for long-context transformer decoding without unacceptable degradation at a **matched task-quality constraint** compared with relevant mature methods. The current code is an *existing-algorithm baseline*, not an original processor architecture.

## 2. Scope and resources

- **Primary target:** transformer **decode**, batch size 1, long KV cache, causal attention, preferably multi-head/group-query implementations when real traces are available.
- **Current capacity:** local CPU Python, NumPy, synthetic single-query single-head snapshots; *no GPU profiling, model traces, HBM measurements, RTL, FPGA, ASIC, or fabrication resources supplied*. Power, die cost and timing are unknown.
- **Current dimensions:** 128 pages × 32 tokens/page × 64 hidden dimensions = 4096 tokens; 12 predetermined seeds; float64 numerical math; float32-equivalent modeled traffic only.
- **Error contract for the synthetic audit:** output L2 error divided by maximum L2 norm of any value vector <= 0.02. This does not establish any downstream model or task-quality guarantee.
- **Baseline model:** dense exact stable softmax computation on the same synthetic data. Page-min/max query-bound screening is a *Quest-inspired reference bound*, not the original Quest implementation.

## 3. Evaluation matrix (planned beyond the synthetic baseline)

| Axis | Required before G5 | Current evidence |
|---|---|---|
| Models | >=2 independently pretrained model families with pinned checkpoint, tokenizer, attention config and versions | None |
| Data | Long-document perplexity and QA, needle-in-haystack retrieval, RULER/LongBench or equivalent, plus adversarial/anti-sparse inputs | Synthetic regimes only |
| Lengths | 8K, 32K and longer subject to model support and resources | 4K synthetic tokens |
| Query cases | Early/late decode, attention sinks, rare-token retrieval, random, flat/softmax diffuse, clustered | 4 synthetic regimes |
| Batch/heads | Batch=1 primary; batch=8 secondary; measure heads/layers, multi-query/group-query behavior | One query/head only |
| Precision | Practical BF16/FP16 and applicable quantization, verify numerical bounds with conservative rounding | Float64 math; float32-equivalent bytes |
| Baselines | Dense optimized FlashAttention variant; Quest, SparQ, BLASST/Skip Softmax, NSA/ETA where applicable, matched hardware and quality | Dense model and two metadata-bound variants only |
| Repeats | >=3 complete independent repetitions; paired seeds and bootstrap uncertainty or confidence intervals | 12 synthetic seeds; no timed runs |

## 4. Mandatory measurements

**G4 simulation:** peak + steady-state bytes by tier; K, V, metadata and build traffic; sort, dot and normalization operation costs; selection fractions; worst-case error; number of fallback events; per-query total cost; sensitivity to page length, dimension, tolerance, distribution, context reuse, and numerical precision.

**G5 real workloads and G9 PPA:** end-to-end token latency p50/p95; throughput tokens/s, workload model-accuracy difference; energy/token measured at relevant device boundary; total memory read/write at each available tier; peak memory; compilation/runtime overhead; area/timing/clock/power/thermal implications; reliability. Distinguish measured, modeled and unknown values.

## 5. Success criteria / stop conditions

- Synthetic verification **PASS for an error bound**: no actual error exceeds the declared certificate in all executed random, boundary and adversarial tests; explicitly report test coverage and finite-precision caveats.
- Synthetic performance metric: bytes are a **proxy**, never sufficient for hardware success.
- Proposed G5 success threshold (to be locked once the exact system and comparison conditions are set): >=2× improvement in a decision-relevant **measured end-to-end** efficiency metric versus a competitive baseline at matched model accuracy, no material regression in high-priority tasks, and acceptable cost/power. The target is provisional: **not a claim of novelty or feasibility.**
- Stop or pivot if preprocessing/metadata eliminates benefit, quality fails adversarial retrieval, extra work destroys latency savings, or near-identical prior art defeats any distinct mechanism claim.
- Novelty (G1, G12) and implementation viability (G6-G11) are **separate gates**, even if G5 succeeds.

## 6. Synthetic experiment — reproduce

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python processor_baseline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/attention_screening_audit.py --output results/attention_screening_audit.json
```

Record date, Python/NumPy versions, parameters, all individual results in the JSON file. The audit is not a real workload benchmark.

## 7. G0 decision

**PASS only for this exploratory charter and benchmark contract** (2026-11-03). Does NOT authorize G5/G12 claims: hardware, representative model workloads, manufacturing budget, and independent technical review are unavailable. Revisit contract before scaling beyond exploratory tests.