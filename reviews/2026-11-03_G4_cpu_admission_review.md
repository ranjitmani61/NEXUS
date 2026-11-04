# G4 Review — Executable Admission/Fallback vs Dense CPU

**Date:** 2026-11-03. **Project:** 01. **Outcome:** partial positive on favorable synthetic data; negative on the other three regimes; G4 IN PROGRESS. This is NOT G5.

## Measured results

Final v1.1 source: `tests/g4_hybrid_admission_benchmark.py`. Two complete benchmark repetitions: `results/g4_hybrid_admission_v1_1_rep1.json`, `results/g4_hybrid_admission_v1_1_rep2.json`. Each repetition has 40 timed queries per regime (two seeds × 20 queries). Three timing measurements per query. In the table below `candidate/dense` is the median per-query CPU latency ratio EXCLUDING one-time metadata building; Q20 amortizes one key-summary build over 20 queries on an unchanging cache.

| Workload | One-page admissions | Candidate/dense range across two runs | Q20 total ratio range | Interpretation |
|---|---:|---:|---:|---|
| Clustered | 40/40 in both runs | 0.234–0.242 | 0.670–0.717 | Synthetic favorable, static K |
| Random | 0/40 | 1.223–1.256 | 1.729–1.782 | Exact fallback incurs overhead |
| Uniform keys | 0/40 | 1.208–1.213 | 1.710–1.806 | Exact fallback incurs overhead |
| Needle | 0/40 | 1.207–1.221 | 1.685–1.718 | Exact fallback incurs overhead |

On clustered keys, estimated minimum query reuse to amortize the measured key-summary build was **11.4–12.4 static queries**; not a guaranteed break-even threshold on changing KV caches or another machine. Zero numerical certificate violations observed across the runs (320 timed query cases). Existing 14 admission tests passed; the combined pipeline had 27 passing tests. The tests cannot establish a rigorous floating-point bound over all possible inputs.

## Correction: metadata accounting

Initial exploratory v1.0 timing billed a full V norm scan during metadata creation. That norm was used only for *offline verification* of normalized output error and is not required for the online admission decision. v1.1 removes the V scan from charged key-summary build time; older exploratory v1.0 results are retained without rewriting. This correction improves the Q20 ratio but **does not change the three negative-regime outcomes**.

## Current limitations

1. No pretrained model checkpoint, 8K+ attention trace, model task-quality test, or competitive optimized GPU baseline. G5 remains blocked, as in the previous review.
2. CPU timings are measurements of a small reference program, not memory-bandwidth counters, power, energy, silicon PPA or a future processor.
3. The candidate uses a conservative one-page admission only; it rejects profitable multi-page possibilities. In fallback, it still scans all page summaries before dense attention, making it slower than direct dense.
4. Synthetic correlated queries and **static K/V** do not model dynamic append-and-update cost of real decoding. Metadata rebuild/refresh is a serious unmeasured cost if per-token updates are needed.
5. Similar page metadata/selection mechanisms exist in published prior art (e.g. QUEST 2024). Novelty is unproven. https://proceedings.mlr.press/v235/tang24l.html

## Gate decision

- G3: IN PROGRESS — one exact arithmetic inequality checked numerically, not formal finite precision proof.
- G4: IN PROGRESS — executable local CPU study with matched FP64 reference, static metadata amortization and negative controls; broader system cost and other baselines missing.
- G5: BLOCKED — no pretrained weights and representative 8K+ model runs.
- G6–G12: NOT AUTHORIZED; no new processor demonstrated.

**Next gate:** obtain a free/authorized pretrained trace; with dynamic append ensure page metadata correctness, and compare optimized real execution at matched task quality. Avoid another narrow synthetic tuning loop.