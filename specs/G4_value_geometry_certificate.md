# G4 value-geometry certificate — mathematical and engineering specification

**Version/date:** 1.1 / 2026-11-03  
**Status:** Exploratory numerical algorithm, NOT a proposed chip, NOT established novel  
**Engineering lenses:** A1 architecture, A2 numerical analysis, A3 AI systems, A6 memory, A8 verification, A10 performance, A11 prior art  
**Goal:** Test whether per-page value-vector summaries can certify low output error while skipping query-specific KV reads, improving over the earlier mass-only screen.

## Definition and exact-arithmetic certificate

For a query `q` and causal cache pages, let the score of token `i` be `s_i = q·k_i / sqrt(d)`. Assume we have processed a nonempty set of pages `S` exactly and computed a stable selected-only softmax partial numerator `N` and denominator `Z>0`, using some common score shift `a`:

`Z = Σ_{i∈S} exp(s_i-a)`; `N = Σ_{i∈S} exp(s_i-a)*v_i`; `y = N/Z`.

For each unseen page `j`, suppose we store a score upper bound `b_j` such that `s_i <= b_j` for every token `i∈j`. Then the unseen shifted softmax mass satisfies:

`0 <= z_j = Σ_{i∈j} exp(s_i-a) <= U_j = |j| * exp(b_j-a)`.

Store a value-center `c_j` and upper-radius `r_j >= max_{i∈j} ||v_i-c_j||₂`. The unseen page-weighted average `μ_j` belongs to the Euclidean ball centered at `c_j`, radius `r_j`, because it is a convex combination of contained values. Thus `||μ_j-y||₂ <= d_j := ||c_j-y||₂ + r_j`.

The full attention output `Y` obeys

`Y-y = [Σ_{j∉S} z_j (μ_j-y)] / [Z + Σ_{j∉S} z_j]`, hence

`||Y-y||₂ <= max_{0 <= z_j <= U_j} (Σ z_j*d_j)/(Z + Σ z_j) = B`.

The maximization is a *fractional box problem*. The unique nonnegative root `t=B` satisfies

`F(t) := Σ_j U_j * max(d_j-t,0) - t*Z = 0`.

Proof: for any `t`, the inequality `(Σ z_j*d_j)/(Z+Σ z_j) >= t` is equivalent to `Σ z_j(d_j-t) >= t*Z`. Independent `0 <= z_j <= U_j` maximize its left side by taking `z_j=U_j` for `d_j>t` and `0` otherwise. Thus `F(t)>=0` exactly when some permitted mass allocation reaches at least `t`. As `F(0)>=0` and `F(max(d_j))<=0`, its root is the supremum. **The independent-mass box is a relaxation** of physically realizable page scores, so this proves an upper bound, not tightness for the real attention distribution.

Stop if `B/Vmax <= ε`, with `Vmax = max_i ||v_i||₂ > 0`, matching previous normalization `ε=0.02`. The specific implementation requires at least one exact page read and a true global `Vmax` summary. Empty or zero-valued cache must be handled separately in real implementation; not tested as hardware.

In exact arithmetic, using exact `b_j` from the existing Cauchy–Schwarz page-key center/radius bound gives a valid sufficient certificate. **Current Python float64 evaluations are numerical demonstrations, not a proven finite-precision or hardware-safe implementation.** Exponentiation, overflow handling, rounding of score bounds, radii and the root, and strict outward rounding must be engineered separately.

## Per-query execution and comparisons

1. Compute each page-key geometric score bound `b_j=(q·key_center_j+||q||*key_radius_j)/sqrt(d)` from key metadata.
2. Visit KV pages in descending `b_j`; maintain stable selected-only softmax `N,Z`.
3. After each visit, compute the candidate output `y` and all remaining `d_j` from value-center/radius metadata. Evaluate `B` by 55-step scalar bisection of the root formula.
4. Stop on value-aware certificate ≤ 0.02, else read the next page. Fallback to dense when the certificate does not pass.
5. Baseline: earlier mass certificate `2U/(Z+U)`, same key score bounds and page priority; `U=Σ_j U_j`.
6. Oracle diagnostic: score upper bound set to exact highest score in each page, which **requires all QK operations** and cannot be claimed as a reduced-traffic method.

## Costs and limitations

- Key and value metadata per page: `2*(d+1)` float32 words in the hypothetical traffic model; this includes two centers and two scalar radii. Score-key metadata alone uses `(d+1)` float32 words per page.
- Dense KV reads: `n_tokens*2*d*4` modeled bytes/query. Candidate: `2*d*4*number_tokens_in_selected_pages + 2*(d+1)*4*number_pages` modeled bytes/query. Mass-only candidate: same but only key metadata.
- Metadata creation, maintenance, cache placement, temporal reuse, page indexing, dynamic scheduling, score-bound MACs, root-finding operations and compiler costs are excluded from these fractions. They can change the result in either direction. In particular, 55 root iterations per visited page are **not free**.
- One cold V-metadata scan requires reading all values or fusing its creation into an existing KV write. Summaries must be updated safely whenever the final page grows; stale summaries invalidate the bound.
- Tight value concentration is the only favorable regime established here. Ordinary independent value vectors, pagewise clustered vectors with dispersed centers, and untrained transformer traces showed no saved pages.
- Claiming originality from page value-centering or deterministic output bounds is invalid without detailed prior-art review; see G1.

## Reproduce

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g4_value_geometry_certificate.py --output results/g4_value_geometry_certificate.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g4_value_geometry_sensitivity.py
```

**Gate decision:** Exact-arithmetic mathematical bound derived; software checks pass on tested distributions; broad workload benefit absent. G3 finite precision IN PROGRESS; G4 exploratory IN PROGRESS; G5 BLOCKED. No RTL, PPA, GPU model inference, or silicon claim.