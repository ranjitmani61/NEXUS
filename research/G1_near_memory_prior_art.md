# G1 — Near/in-memory computing prior-art check

**Date:** 2026-11-03. **Scope:** A narrow prior-art reconnaissance, **not** exhaustive patent search or a novelty opinion.

| Public research | Date / link | What it already covers | Implication |
|---|---|---|---|
| IBM Research, *Analog In-Memory Computing for Large Language Model Inference: Opportunities and Challenges* | 2026-05-10, https://research.ibm.com/publications/analog-in-memory-computing-for-large-language-model-inference-opportunities-and-challenges | Weight computations in memory arrays and system/precision challenges for LLMs | Weight-local matrix multiplication is not a new primitive |
| Duan et al., *HPIM: Heterogeneous Processing-In-Memory-based Accelerator for LLMs* | 2025-09-16, https://arxiv.org/abs/2509.12993 | HBM-PIM for weight-intensive GEMV, SRAM-PIM for attention, hardware/software co-design | Heterogeneous SRAM/HBM PIM is already proposed; published speedups are authors' simulator claims, not independently replicated here |
| Davies et al., *LIMINAL: Exploring the Frontiers of LLM Decode Performance* | 2025 preprint, https://mast.stanford.edu/pubs/liminal_exploring_the_frontiers_of_llm_decode_performance/ | Hardware performance limits including memory capacity, bandwidth, and interconnect | Need to model capacity and system bottlenecks, not just MAC efficiency |
| He et al., *Understand and Accelerate Memory Processing Pipeline for LLM Inference* | ICML 2026, https://proceedings.mlr.press/v306/he26ab.html | Heterogeneous GPU/FPGA offloading with published benchmarks | Heterogeneous memory processing is also existing work |
| *SoftmaxPIM*, IEEE Computer Architecture Letters | 2026-07-23, https://ieeexplore.ieee.org/document/11622666 | Near-bank/in-memory softmax and GEMV pipeline | Fused weight+softmax near-memory approach also has prior art |

**Novelty finding:** Basic near-memory compute, HBM/SRAM heterogeneity, and fused softmax are prior art. Current Project 01 does not specify any new physical logic primitive, processing mechanism, interconnect guarantee, or formally differentiated microarchitecture. **G1: NOT PASSED** for the near-memory idea.

**External claims are not independent reproduction:** This project did not recreate these devices or independently validate their performance. Sources informed the research direction only.