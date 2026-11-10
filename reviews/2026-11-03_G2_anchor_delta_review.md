# G2 Exploratory Engineering Review — 2026-11-03

**Protocol:** `AGENTS.md` v1.0 read before work. G0 established; G1 still IN PROGRESS.  
**Lenses:** A1/A2/A3/A4/A6/A8/A10/A11. No independent specialist review.  
**Candidate:** Anchor-Delta Upper-Bound Reuse, an exploratory candidate only.  
**Decision:** **Not approved for hardware design**; G2 for novel processor remains IN PROGRESS. No novelty claim.

## Observations: actual tests
Executed:

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g2_anchor_delta_audit.py --output results/g2_anchor_delta_audit.json
OPENBLAS_NUM_THREADS=1 python tests/g2_anchor_delta_properties.py
```

Python/Numpy version and raw observations are in `results/g2_anchor_delta_audit.json`. The synthetic experiment covered **1,120 queries** (4 regimes × 5 seeds × 7 correlations × 8 queries) with no detected certificate violations. An additional **900 independent randomized/boundary mathematical property checks** passed with floating-point tolerance. These are numerical checks, not exhaustive proof of correct finite-precision hardware.

Reported below: median across five seeds per scenario; *traffic* is modeled total KV + metadata reads over an eight-query group divided by traffic of fresh center/radius screening. Both methods use the same exact-arithmetic softmax-tail certificate.

| Synthetic regime | Query correlation | Candidate / baseline modeled bytes | Median extra KV pages / 8 queries | Candidate wins in seeds |
|---|---:|---:|---:|---:|
| Clustered | 1.000 | 0.4286 | 0 | 5/5 |
| Clustered | 0.990 | 0.4286 | 0 | 4/5 |
| Clustered | 0.950 | 6.6327 | 152 | 2/5 |
| Clustered | 0.900 | 14.5102 | 553 | 0/5 |
| Unstructured | 0.900 | 0.9865 | 0 | 5/5 |
| Uniform | 0.900 | 0.9864 | 0 | 5/5 |

**Interpretation:** On highly coherent and artificially favorable query streams, reduced metadata reads helped. As correlation weakened, the looser stale-anchor bound caused many more KV page reads and negated all possible metadata savings. In unstructured/uniform controls, nearly all KV pages are visited, so about 1.4% savings reflect removing redundant metadata reads—not useful attention sparsity. Negative controls therefore do not demonstrate a relevant breakthrough.

**Critical hardware-system caveat:** If metadata is already hot in SRAM, counted metadata savings disappear. The candidate does not reduce KV blocks relative to the fresh-bound baseline; in the test it never has fewer. Thus no demonstrated advantage survives the hot-metadata comparison.

**Dynamic-cache counterexample:** With old zero-valued keys on a page, stored `U=g=0`, appending new key `[100,0]` while retaining old metadata produces a true attention score `70.710678...` above bound zero. Hence the mechanism **must** use coherent, correctly updated metadata on every KV insertion; the frozen-cache experiment is optimistic for real autoregressive use.

## Research/citations verified during this work
- QUEST (2024, ICML): query-aware KV page selection with metadata — https://proceedings.mlr.press/v235/tang24l.html
- RetroAttention (2026, ICLR): document discusses adjacent-query selection reuse and accuracy pitfalls — https://openreview.net/pdf?id=Ql0G1Zsobn
- ReTopK (July 2026 preprint): reuses similar query support, with fallback — https://arxiv.org/abs/2607.27692
- SPIN (April 2026 preprint): hierarchical KV memory and two-level metadata — https://arxiv.org/abs/2604.26837
- KARAT (Aug 2026 preprint): sparse-attention processing-near-memory system — https://arxiv.org/abs/2608.03555
- KV-CIM (2025, IEEE TCAS II): query-aware sparse-attention digital-CIM accelerator — https://doi.org/10.1109/TCSII.2025.3580135

These sources establish **nearby prior approaches**, not that the exact bound formula was previously used nor that any specific patent claim is infringed. A thorough source code/patent analysis is still missing. Published numerical claims were not independently reproduced here.

## Gate decision
- **Mathematical G2 sub-test:** PASS under static-KV exact-arithmetic premises and tested floating-point tolerances.
- **System-level G2 viability:** NOT ESTABLISHED; test produces severe workload sensitivity and depends on a weak metadata-memory assumption.
- **G2 overall:** IN PROGRESS for a genuinely distinct processor mechanism; **Anchor-Delta candidate rejected as an immediate chip-development direction** pending contrary trace evidence.
- **G3–G12:** NOT STARTED for any new processor.

## Next bounded test
Acquire real, appropriately licensed transformer KV/query traces for long-context batch-1 decode, measure real adjacent-query angular distributions by head/layer, and test candidate against metadata-hot and optimized serving baselines with KV growth correctly handled. If similarity is low or savings vanish with metadata SRAM residence, stop this candidate; then explore a mechanism outside sparse-attention indexing rather than relabeling the same idea.
