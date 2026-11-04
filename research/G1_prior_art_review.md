# G1 Prior-art Review — Project 01

**Version/date:** 1.0 / 2026-11-03  
**Gate status:** IN PROGRESS — technical literature screened; patents/implementation details not exhaustively reviewed.  
**Lens:** A11 prior-art investigation, A1 architecture, A2 math, A3 AI systems.

The starting reference prototype screens whole KV pages using the center/radius upper bound `(q·center + ||q||*radius)/sqrt(d)` and uses a softmax-tail mass certificate to decide when to stop. **Do not call this formula a discovery.**

| Work (year) | Source | Verified published/author-described mechanism | Novelty implication |
|---|---|---|---|
| FlashAttention-2 (2024, ICLR) | https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html | Exact dense attention optimized for parallelism and GPU memory hierarchy | Essential dense implementation reference |
| QUEST (2024, ICML) | https://proceedings.mlr.press/v235/tang24l.html | Tracks per-KV-page min/max keys; estimates page importance using query; selectively loads pages | Query-aware KV page selection is pre-existing |
| SparQ Attention (2024, ICML) | https://proceedings.mlr.press/v235/ribar24a.html | Query-aware selective KV fetching for bandwidth-efficient inference | Selective cache fetching already established |
| Native Sparse Attention (2025, ACL) | https://aclanthology.org/2025.acl-long.1126/ | Hardware-aligned, trained dynamic hierarchical sparse attention | Hardware-aware sparse execution is established |
| BLASST (2026, MLSys) | https://proceedings.mlsys.org/paper_files/paper/2026/hash/c6ee784cbe46d854843e4c883a3321ef-Abstract-Conference.html | Dynamic threshold to skip low-contribution blocks; optimized kernels | Threshold/fallback research must be compared to working optimized kernels |
| Elastic Threshold Attention (2026, Google Research) | https://research.google/pubs/elastic-threshold-attention-learned-contextual-sparsity-for-long-context-decoding/ | Training-time gating, cached geometric-probabilistic screening bounds, fused kernel | Even geometric-bound screening has close modern competitors |
| Certified Block Screening for Adaptive Long-Context Attention (Apr 2026 preprint) | https://www.researchgate.net/publication/404906062_Certified_Block_Screening_for_Adaptive_Long-Context_Attention | Reports **same center-radius Cauchy–Schwarz page bound** and failure on single-token retrieval | Direct overlap; **baseline is not novel**. Preprint not established as peer reviewed |
| Runtime-Certified Bounded-Error Quantized Attention (May 2026 preprint) | https://arxiv.org/abs/2605.20868 | Runtime certification, adaptive precision and fallback | Certification-driven attention is already present |
| Certified Sparse Attention via Perturbed Retrieval (Oct 2026 preprint) | https://arxiv.org/abs/2610.24991 | Perturbed retrieval with certified sparse attention; reports CPU decoding gains under study assumptions | Certificates + selective work are also competitive research topics |
| Efficiently Approximating Attention Is Hard (Sep 2026 preprint) | https://arxiv.org/abs/2609.37261 | Conditional theoretical limitations on input-uniform subquadratic attention approximations | No claim of universally strong screening; special input distributions may still benefit |

### Outcome

- **Disconfirmed:** “Center/radius page screening + a runtime certificate constitutes a new computing mechanism.” This is **not defensible** in light of close published/preprint work.
- **Not yet established:** Any new microarchitecture, physical dataflow, certified finite-precision mechanism or favorable energy/latency/area tradeoff.
- **Open research direction, not an invention claim:** Can a physically specified memory controller reduce **end-to-end** latency/energy by scheduling metadata/blocks *better than current software kernels and near-memory designs*, while preserving useful quality? Must first research hardware and patents; this could also be existing prior art.
- **Next G1 action:** Detailed side-by-side study of closest source code and claims; search patents and deployed chips; isolate a distinct, testable mechanism **before** using the word novel.

External work's speedups and accuracies, if mentioned, are **authors' claims**, not replications by this project. URLs identify sources; no originality or freedom-to-operate opinion is given.

## G2-era addition — 2026-11-03 (screened, not exhaustive)

Additional directly relevant evidence narrows originality claims:

| Source | Link | Implication |
|---|---|---|
| RetroAttention, ICLR 2026 | https://openreview.net/pdf?id=Ql0G1Zsobn | Discusses reuse of adjacent-query sparse KV selections and its limitations; temporal reuse cannot itself be called a breakthrough. |
| ReTopK, July 2026 preprint | https://arxiv.org/abs/2607.27692 | Similar-query selected-support reuse and confidence fallback already investigated. |
| SPIN, April 2026 preprint | https://arxiv.org/abs/2604.26837 | Hierarchical page metadata/placement co-design already exists as research. |
| KARAT, Aug 2026 preprint | https://arxiv.org/abs/2608.03555 | Processing-near-memory for sparse KV indexing and attention is already a studied architecture. |
| KV-CIM, IEEE TCAS II (2025) | https://doi.org/10.1109/TCSII.2025.3580135 | Query-aware memory-side sparse attention acceleration predates this project. |

**G2 implication:** Anchor-Delta Upper-Bound Reuse has a separately testable Cauchy–Schwarz formula, but this limited search cannot establish either worldwide novelty or meaningful superiority. See `specs/G2_anchor_delta_hypothesis.md` and the G2 gate review.

## G3/G4 addition — 2026-11-03

**TokenSelect — Dynamic Token-Level KV Cache Selection** (Wei Wu et al., 2024 preprint, https://arxiv.org/abs/2411.02886) uses a Selection Cache built on consecutive query similarity to reduce selection overhead. This is close prior art for the G2 anchor-query reuse *motivation*; the exact formal bound may differ, but describing query similarity/reuse itself as novel is untenable. Recheck claim-by-claim original paper and code before any patent/novelty conclusion. Original study results are not replicated by the project's toy tests.

An untrained 2-layer causal transformer trace audit was added at `reviews/2026-11-03_G3_G4_toy_trace_review.md`. On its tested distributions, neither fresh nor reused certification skips meaningful KV tokens; the candidate exceeds the modeled dense read budget. This is **not** a trained-model conclusion and does not establish broader impossibility.

## 2026-11-03 G4 oracle-headroom addition

Reviewing page-selection conservatism exposed **no new invention**. Relevant adjacent published research:

| Work | Source | Established overlap |
|---|---|---|
| Gupta & Berant, *Value-aware Approximate Attention* (EMNLP 2021) | https://aclanthology.org/2021.emnlp-main.753/ | Value vectors matter for approximation to attention outputs; value-aware objectives predate this project. |
| *OBCache: Optimal Brain KV Cache Pruning* (ICML 2026) | https://proceedings.mlr.press/v306/gu26h.html | Output-aware KV saliency and perturbation are already studied on modern LLMs. |
| *vAttention: Verified Sparse Attention via Sampling* (ICLR 2026) | https://proceedings.iclr.cc/paper_files/paper/2026/hash/55cb562b1f5af71f6707f3ff3c7941e6-Abstract-Conference.html | Verified approximation and mixed sparse/sampled strategies already investigated. |
| *A Mathematical Theory of Top-k Sparse Attention via Total Variation Distance* (Dec 2025 preprint) | https://arxiv.org/abs/2512.07647 | Head/tail output-error identity and value-diameter refinements already studied. |

**Caution:** The on-line abstracts/landing pages were reviewed, not the full proof code or patent landscape. Any subsequent architecture centered on value geometry must demonstrate a specific advantage **and** differentiated mechanism rather than rebrand these known ideas. See `reviews/2026-11-03_G4_oracle_headroom_review.md`.

## 2026-11-03 value-geometry follow-up (additional limited scan)

- *A Mathematical Theory of Top-k Sparse Attention via Total Variation Distance* (Dec 2025 preprint): https://www.alphaxiv.org/abs/2512.07647 . Discusses exact head/tail output-error identity and value-diameter refinements. Close conceptual overlap with bounds using value geometry; the current work is **not claimed novel**.
- *vAttention: Verified Sparse Attention via Sampling* (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/55cb562b1f5af71f6707f3ff3c7941e6-Abstract-Conference.html . Existing statistical verification alternative; unlike our deterministic worst-case bound, its advertised guarantees are probabilistic. Do not conflate the two.
- *VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention* (Sep 2026 preprint): https://arxiv.org/abs/2609.15810 . Reorders/centers value tokens for quantization, strengthening the evidence that value-centered attention design itself is not new; its application and guarantees differ.
- *Value-aware Approximate Attention* (EMNLP 2021): https://aclanthology.org/2021.emnlp-main.753/ . Earlier value-aware approximation research remains direct conceptual prior art.

**Limit:** The links were reviewed to the level of abstracts and/or search excerpts, not full code/patent claim mapping. The project still has no non-obvious new architecture and no IP clearance.
