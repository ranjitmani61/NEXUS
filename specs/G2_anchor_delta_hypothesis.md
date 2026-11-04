# G2 Hypothesis Specification — Anchor-Delta Upper-Bound Reuse

**Version/date:** 1.0 / 2026-11-03  
**Owner/lenses:** A1 computer architecture, A2 applied mathematics, A3 ML systems, A4 compiler/runtime, A6 memory, A8 verification, A10 performance, A11 prior art  
**Human engineering review:** None  
**Gate:** G2 exploratory; **NOT** a novel processor or circuit.

## Falsifiable question
Can reusing certified KV page upper bounds between *similar* queries reduce memory traffic versus re-evaluating page metadata on every query, after counting fetched KV pages and metadata traffic?

## Proposed method
For a static KV cache, page `i` has `K_{ij}` vectors of dimension `d`, stored center `c_i`, radius `r_i >= max_j ||K_{ij}-c_i||_2`, and scalar `g_i = ||c_i||_2 + r_i`. For anchor query `a`, store per-page

`U_i(a) = (a·c_i + ||a||_2 r_i) / sqrt(d)`.

For any future query `q`, use

`U_i(q | a) = U_i(a) + ||q-a||_2 g_i / sqrt(d)`.

**Exact-arithmetic derivation:** for every key `k` on page `i`,

`q·k = a·k + (q-a)·k`

`<= sqrt(d) U_i(a) + ||q-a||_2 ||k||_2`

`<= sqrt(d) U_i(a) + ||q-a||_2 g_i`.

Thus no page score can exceed the bound in exact arithmetic. The existing certified-tail-mass stopping criterion from `tests/attention_screening_audit.py` applies unchanged *if* every page has a valid bound, the KV cache is unchanged, and numerical arithmetic is adequately controlled. **This is not a new theorem claim; it follows immediately from Cauchy–Schwarz.**

## Baseline and test contract
- Fresh per-query center/radius bound (existing `screen(..., bound='sphere')`) vs reusable anchor-delta bound. Dense exact attention is used as error reference.
- 128 pages × 32 tokens × 64 features, float64 simulation, float32-*equivalent* modeled KV and metadata bytes, 2% normalized output error target.
- Four deliberately synthetic key regimes: clustered, unstructured, uniform, rare important token; five seeds each; seven query cosine correlations `[1, .999, .99, .95, .9, .5, 0]`; eight queries per group. Frozen KV, not autoregressive growth.
- Count full center/radius page metadata on **each** baseline query; anchor case reads full metadata on initial query and two float scalars (`U_i(a), g_i`) per page subsequently. Total fetched K+V blocks are charged at identical cost.
- **Costs excluded:** metadata building, computation, sorting, cache/hierarchy residency, physical transactions, updates, compiler/runtime, energy, time, storage capacity, chip area. All exclusions can negate apparent modeled gains.
- Alternative `metadata_hot` comparison eliminates all metadata DRAM reads for both algorithms (e.g., data resident in local SRAM) to expose sensitivity to memory placement.
- No experiment represents real LLM queries or proves a deployable memory controller.

## Break-even derivation for specified synthetic configuration
- Metadata fresh query: `128 × (64+2) × 4 = 33,792 bytes`.
- Reuse per subsequent query: `128 × 2 × 4 = 1,024 bytes`.
- Per KV page fetched: `32 × 64 × 2 × 4 = 16,384 bytes`.
- Over eight queries, the maximum modeled metadata savings is `7 × (33,792 − 1,024) = 229,376 bytes`.
- Therefore **14 additional KV pages over eight queries erase the entire modeled metadata saving**. With metadata resident on chip, even one additional KV page is unfavorable under this bytes-only comparison.

## Implementation and falsification
- Main simulation: `tests/g2_anchor_delta_audit.py` → `results/g2_anchor_delta_audit.json`.
- Adversarial checks: `tests/g2_anchor_delta_properties.py` → `results/g2_anchor_delta_properties.json`.
- Reproduce with:

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g2_anchor_delta_audit.py --output results/g2_anchor_delta_audit.json
OPENBLAS_NUM_THREADS=1 python tests/g2_anchor_delta_properties.py
```

## Required caveats and stop conditions
1. **Staleness invalidates the bound.** On a growing KV cache, newly appended keys can exceed an older `g_i`. Without correct synchronized metadata updates, output certification becomes unsound. A concrete violated bound appears in `results/g2_anchor_delta_properties.json`.
2. **Correlated queries are assumed, not observed.** Adjacent real LLM queries may be substantially less similar, head/layer dependent, and change with positional encoding.
3. **The baseline could be metadata-resident.** The simulated savings assume baseline must repeatedly read full metadata from an expensive tier; contemporary accelerators may cache it.
4. **Float64 tests are not proof of float32 guarantees.** Certification needs outward-rounded bounds, finite-precision analysis, overflow/underflow handling, and verification.
5. **Prior-art overlap likely:** query reuse and memory-resident sparse attention already have substantial research, so this proposal is **NOT established novel**.
6. **Stop:** If real traces rarely have correlations necessary for low extra-KV fetching, or cached metadata invalidates the advantage, reject hardware implementation.