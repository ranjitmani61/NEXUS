# G1 prior-art update — Batched target FFN weight amortisation

**2026-11-10; v1.0.** Scope: Is multi-query target verification with batched GEMM/SwiGLU and shared weight loads a new processor mechanism? **No.** Original 2023 speculative decoding already parallelised target scoring across several proposed tokens. The operational-intensity difference between GEMV and GEMM and resulting memory/compute roofline was established well before 2026. The 2025 block-level verification paper further optimizes acceptance. Known 2026 NEXUS G1 review documents Dustin and SpecStream direct overlap for attention/KV.

| Source | Direct relation | This study's status |
|---|---|---|
| Leviathan et al., ICML 2023, https://proceedings.mlr.press/v202/leviathan23a.html | Parallel target verification and acceptance accounting | Read; no independent replication of their method |
| Williams, Waterman & Patterson, Roofline 2009, https://escholarship.org/uc/item/5tz795vq | Compute intensity and bandwidth bottleneck shift for reuse/streaming | Model context only; no peak compute or memory bandwidth calibration |
| Sun et al., Block-Level Verification, 2025, https://research.google/pubs/block-level-verification-accelerates-speculative-decoding/ | Proposed token verification alternatives with reduced latency overhead | Primary source checked; not replicated |
| Dustin, ICML 2026, https://proceedings.mlr.press/v306/lee26d.html | Draft-assisted sparse verification under long context | Reviewed in preceding G1 joint study; not replicated |
| Exact CPU dense baseline with measured attention shares | NEXUS G5-R and C2 branches, immutable refs in spec | Previously recorded third-party AI-assisted result; not our new measurement |

**G1 verdict:** This is a falsification **measurement audit**, not a novel architecture. We made no patent search and have no independent human review. The corrected component-cost equations are ordinary performance accounting, not claimed as new research.
