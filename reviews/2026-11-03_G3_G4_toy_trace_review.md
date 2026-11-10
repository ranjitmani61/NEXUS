# Project 01 — G3/G4 Trace Audit and Gate Decision

**Date/version:** 2026-11-03 / v1.0  
**ACTS applied:** A1 architecture, A2 mathematical analysis, A3 ML systems, A4 runtime, A6 memory, A8 verification, A10 systems, A11 prior art. A5/A7/A9/A12 and independent human specialist review **not performed**.

## Research question

Does the G2 anchor-delta page bound avoid meaningful KV traffic during actual *causal transformer computations* when the KV cache grows, and does it beat (i) fresh per-query sphere screening and (ii) exact dense KV reading under both on-chip/hot and off-chip/cold metadata assumptions?

## What was run

`tests/g3_g4_autoregressive_trace_audit.py`: 3 fixed independent model seeds (20261103, 20261123, 20261213), 4 token-sequence regimes (iid, periodic, piecewise, rare marker), 2 transformer layers, 2 audited heads per layer, 48 decode steps per trace. **48 traces, 2,304 per-head/layer decode queries.** Fixed 1,024-token teacher-forced sequences; two-layer, four-head, 64-wide, 16-dimension/head, causal self-attention with RoPE and normal residual/FFN arithmetic; **weights were never pretrained**. Each query sees its growing KV. Pages hold 32 tokens. Re-anchor every 8 decode queries. Float64 attention math; report **modeled float32 KV+metadata byte counts**.

`tests/g3_dynamic_kv_properties.py`: 4 seeds x 3 constructed regimes x 80 growing-KV positions x 2 screening bounds = **1,920** numerical score-bound + output-certificate checks. Includes a negative control deliberately failing with stale metadata.

Scripts were executed on CPU; no GPU, FPGA, ASIC, synthesizer, pretrained weights, profiler, or external model traces were available in the tested workspace.

## Results actually observed

| Metric | Result |
|---|---:|
| Traces / decode queries | 48 / 2,304 |
| Detected page-score bound violations | 0 |
| Detected partial-output certificate violations | 0 |
| Additional growing-KV property checks | 1,920; 0 detected violations |
| Mean adjacent-query cosine (average of trace means) | -0.01227 |
| Fresh method KV reads / dense KV reads | 99.9964% |
| Anchor method KV reads / dense KV reads | 99.9979% |
| Fresh method modeled traffic with cold metadata and page-rebuild reads / dense KV | 102.6846% |
| Anchor method modeled traffic with cold metadata and page-rebuild reads / dense KV | 101.3014% |
| Fresh method modeled traffic with metadata hot and page-rebuild reads / dense KV | 100.9043% |
| Anchor method modeled traffic with metadata hot and page-rebuild reads / dense KV | 100.9058% |
| Median anchor / fresh cold-metadata traffic ratio | 0.986515 |
| Median anchor / fresh metadata-hot traffic ratio | 1.0 |
| Worst normalized output error observed in trace audit | 0.004479 |

Note: Results are ratios of **modeled read bytes**, not speed, throughput, energy, total model traffic, or silicon efficiency. Cold/hot sensitivity is hypothetical, not profiled from real memory tiers. The small cold-metadata gain versus **fresh screening** does **not** mean improvement versus **dense**. The 2,304 queries are correlated within 48 traces; they are not 2,304 statistically independent model runs.

**Negative control:** A score bound kept stale when a new high-scoring key is appended to an unfilled page reports a false certificate of about `0.0006707`, while the true normalized output error is `1.0`. Correct metadata refresh restores correctness in this constructed case.

## Audit of limitations and cost completeness

1. **No trained-model traces or model task accuracy.** The simulated Q/K/V are from a genuine causal forward pass but with randomly initialized parameters. This negative result cannot be generalized to pretrained transformers.
2. **Metadata read placement is assumed, not measured.** Benefits diminish to zero when metadata is hot/on chip. Updating page metadata, its storage/writes, sorting, comparisons, runtime dispatch, caches, bandwidth, HBM transactions and memory coalescing need a physical cost model.
3. **Prefill and initial metadata construction are excluded** from decode-only ratio. The active page is recomputed on every append; its modeled key-read bytes *are* included. Reusing bounds requires a key-page scan for updates or a more advanced incremental bound.
4. **Matched reference is weak.** Fresh sphere screening is an algorithmic reference, not an optimized contemporary FlashAttention/FlashInfer/QUEST/BLASST implementation. Full dense KV reads are the primary separate lower-complexity comparison. No measured end-to-end advantage is established.
5. **Floating-point safety remains unproven.** Certification derived for exact arithmetic is numerically tested in float64 only.
6. **Timing, power, thermal, VLSI, compiler, and economics** remain untested. The baseline is additionally slower than dense in the tested modeled read-byte accounting.
7. **Prior art strongly overlaps.** QUEST (ICML 2024) uses query-aware page selection; TokenSelect (Nov 2024 arXiv, arXiv:2411.02886) explicitly introduces a selection cache based on consecutive query similarity; BLASST (MLSys 2026) implements GPU-optimized dynamic sparse attention. All need closer comparison before claims of novelty or value.

## Decision

- **G0:** PASS for the exploratory charter only (unchanged).
- **G1:** IN PROGRESS. Consecutive-query reuse is pre-existing; no defensible novelty claim established.
- **G2:** Query-anchor formula feasible in exact arithmetic but **FAIL as a chip-development justification on current evidence**. Candidate remains a baseline, not an architecture concept for tapeout.
- **G3:** IN PROGRESS. Positive numerical growing-KV checks and valid exact-arithmetic math; full finite-precision verification missing.
- **G4:** IN PROGRESS. Reproducible negative-control simulation exists, but no representative pretrained workload, realistic memory hierarchy, calibrated latency/energy, or full complete-cost model.
- **G5:** BLOCKED. Need real pretrained-model KV traces and competitive system/hardware results at matched quality.
- **G6–G12:** NOT STARTED. No new microarchitecture or breakthrough demonstrated.

**Stop/pivot decision:** Do not invest in RTL/ASIC for query-anchor reuse. As an algorithmic baseline it remains useful. Next engineering step: obtain legitimately usable pretrained-model Q/K/V traces, verify head/layer query correlation and KV selection at realistic long contexts, and evaluate optimized baselines. If the candidate remains weak, pivot the architecture search outside query-reuse screening rather than renaming a known method.

## Reproduction and outputs

```
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g3_g4_autoregressive_trace_audit.py --output results/g3_g4_toy_trace_audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g3_dynamic_kv_properties.py
```

Machine-readable outputs: `results/g3_g4_toy_trace_audit.json`, `results/g3_dynamic_kv_properties.json`. Mathematical specification: `specs/G3_growing_kv_invariant.md`. Libraries: NumPy, Python standard library.

Primary prior-art URLs: https://proceedings.mlr.press/v235/tang24l.html ; https://arxiv.org/abs/2411.02886 ; https://proceedings.mlsys.org/paper_files/paper/2026/hash/c6ee784cbe46d854843e4c883a3321ef-Abstract-Conference.html
