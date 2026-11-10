# G1 Prior-Art Screen — Multi-token verification and shared historical KV

**Date/version:** 2026-11-10 / 1.0. **Scope:** Is target-weight reuse + multiquery KV reuse + draft verification itself a novel processor mechanism? **Answer: NO evidence of novelty; strong direct prior art.** Review lens A11 plus A1, A4, A6. No patent search, independent human reviewer, or hardware validation.

| Source | Verified publication | Relevant overlap | Reproduction status |
|---|---|---|---|
| Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 | https://proceedings.mlr.press/v202/leviathan23a.html | Draft k tokens; target verifies in parallel; lossless sampled distribution using acceptance/rejection mechanism | Paper checked, not replicated here |
| Dao et al., *FlashAttention*, NeurIPS 2022 | https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html | IO-aware exact attention/streaming softmax; shared-KV multiquery reuse is not new computing physics | Paper checked, not replicated |
| Hydragen (shared-prefix attention), public implementation | https://github.com/ScalingIntelligence/hydragen | Shared-prefix K/V read optimization with batched queries and exact softmax composition | Public code overview checked, not reproduced |
| EAGLE-3, NeurIPS 2025 | https://papers.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html | Strong optimized parallel target verification/decoding comparator; reports up to 6.5×, not this project's measurement | Publication checked, not reproduced |
| **Dustin**, ICML 2026 | **https://proceedings.mlr.press/v306/lee26d.html** | **Direct overlap:** draft-assisted **sparse** long-context target verification on Qwen2.5-72B, PG-19 / LongBench. Paper reports 9.17× end-to-end at 32K. This is a published claim, NOT independently validated here. | Abstract and primary proceedings checked |
| SpecStream, arXiv preprint 2026-09 | https://arxiv.org/abs/2609.33184 | Multiple verification queries share streamed historical KV; online softmax; draft/target overlap. Direct conflict with generic streaming-KV + speculation originality. | Preprint checked, not reproduced |
| Liu et al., *Speculative Decoding: Performance or Illusion?*, arXiv 2026 | https://arxiv.org/abs/2601.11580 | Real-world overhead/acceptance varies by task/batch; overhead-rich vLLM evaluations contradict equating a hardware-level reuse estimate with true speedup. | Paper abstract checked |

## Novelty gate

The default combination **does not establish G1 novelty**: parallel verification, exact memory-efficient attention, shared-KV batch reuse, sparse verification and concurrent drafting exist already. A hypothetical NEXUS mechanism would need a concrete, distinct mathematical or implementation primitive and must beat the strongest **matched quality** EAGLE-3, Dustin and production-grade vLLM / SGLang baselines. These papers show existence of designs that exceed 2× on some workloads; the 2× target is neither new nor a universal limit.

Dustin's Qwen2.5-72B performance cannot be transferred to the tiny Qwen/Granite prior G5-R CPU experiments, which used different architecture, workload, precision, implementation and hardware. Conversely, prior G5-R failure for the NX bound family does not disprove sparse draft-assisted verification based on a different token-importance predictor. Exact output-distribution preservation by speculative acceptance does not carry over automatically to *approximate sparse target verification*.

**Decision:** generic multi-token KV amortization / sparse speculative verifier is **G1 FAIL as a novel principle**. No patent novelty determination; no prototype. Continue only as a conditional feasibility study and do not label the method an invented processor.
