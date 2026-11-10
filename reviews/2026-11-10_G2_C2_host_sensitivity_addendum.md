# G2 Addendum v1.1 — Host-dependent attention share changes the frontier

**Date:** 2026-11-10. **Evidence update after initial v1.0 review was committed.** Not a pre-registered input to the original 336-row matrix. It does **not** change that experiment's inputs or erase negative results. New external research branch data were discovered during the post-commit remote verification.

**Sources (immutable refs, same repo):**
- Earlier G5-R branch commit `8510d89980816e06a511ab0d665188cbe7da3d6c`, `results/g5r/summary.json` (host B Granite).
- Newer Claude C2 branch commit `922dc8f3b5639f4d5c6297da8f698f9b4580a149`, `results/c2/breakdown_granite4micro_bf16.json` and `results/c2/breakdown_qwen05_bf16.json` (host C).

| Context, batch | CPU | Measured per-token attention share | Measurement |
|---|---|---:|---|
| Granite-4.0-micro-base, 16K, batch 1 | Host B (Xeon family 6 model 85, no AMX) | 0.70520667 | G5-R `dense_attn_share` |
| Granite-4.0-micro-base, 16K, batch 1 | Host C (Xeon family 6 model 207, AMX-BF16) | 0.37976426 | C2 `time_share_median.attn_core` |
| Qwen2.5-0.5B, 32K, batch 1 | Host C (Xeon family 6 model 207, AMX-BF16) | 0.49043314 | C2 `time_share_median.attn_core` |

**Strong limitation:** G5-R and C2 use different host configurations *and* measurement instrumentation; the C2 `attn_core` label and G5-R `dense_attn_share` are not established as perfectly exchangeable. Accordingly this is an **illustrative sensitivity calculation**, not a measured verifier speedup or a controlled cross-host causal estimate. C2 hook overhead is separately reported and may matter at some contexts.

Applying only the same optimistic G2 equation, for k=4, p=.90, assumed drafting delta=.05 per proposal, h=0:

| Hypothetical scenario | `beta_max_for_2x` |
|---|---:|
| Granite host B f=.7052 | **0.30046** |
| Granite host C f=.3798 | **0.55794** |
| Qwen host C f=.4904, batch 1 | **0.43204** |

For k=8, p=1.0, beta=1 (no multiquery attention reuse), delta=.05, the *optimistic* model gives **2.028×** for Granite host C and **1.278×** for Granite host B. This does **not** mean any actual multi-token decoder reaches either speed. It means the original host-B no-reuse 2× obstruction is workload-specific. The inferred nonattention weight cost of a k+1-token pass is still idealized and unmeasured.

**Gate posture unchanged:** no novel method demonstrated (Dustin/SpecStream/EAGLE-3 prior art); G2 conditional/in progress, G5 not run for the joint hypothesis, G6–G12 unauthorized. A *same-model, same-host* target-verification benchmark must measure actual acceptance, draft cost, and attention reuse together before any 2× claim.
