# Project 01 — G4 Value-Geometry Certificate Gate Review

**Version/date:** 1.1 / 2026-11-03  
**Discipline lenses applied:** A1, A2, A3, A6, A8, A10, A11. **Absent:** independent specialist review, A5/A7/A9/A12 physical/thermal/manufacturing evaluation.  
**Decision:** **REJECT as current processor investment thesis**; retain as mathematical research baseline and narrow diagnostic.

## Evidence and execution

`AGENTS.md` v1.0, the project log, the G3/G4 toy-transformer test, the G4 oracle audit and the existing prior-art review were read before work. New scripts:

- `tests/g4_value_geometry_certificate.py` (v1.1): new page-center/value-radius certificate, independent fractional-box maximization tests, random exact dense comparisons, untrained transformer evaluation, constructed controls and an impractical exact-score-upper oracle.
- `tests/g4_value_geometry_sensitivity.py`: controlled sensitivity to the standard deviation of the value vectors around a common center.
- `results/g4_value_geometry_certificate.json`: full 640 query-level records plus 1,600 box-optimum and 480 attention-certificate property checks.
- `results/g4_value_geometry_sensitivity.json`: 24 sensitivity settings across 3 random seeds.

Reproduce with the commands in `specs/G4_value_geometry_certificate.md`. NumPy float64 arithmetic; a separate float32 *byte model* for K/V and summaries. No actual memory traffic, latency, energy, numerical formal verification or product model task accuracy measured.

## Findings

| Workload/control | Value-aware pages | Mass-only pages | Value-aware traffic / dense |
|---|---:|---:|---:|
| All values identical; 4,096 tokens, 128 pages | 1 | 128 | 4.10% |
| Almost identical values; same setup | 1 | 128 | 4.10% |
| Independent random values; same setup | 128 | 128 | 103.32% |
| Pagewise coherent but different centers; same setup | 128 | 128 | 103.32% |
| Untrained two-layer RoPE transformer, median of 640 queries | All 16 | All 16 | 103.38% |

The untrained model uses 2 random-weight seeds × 4 token patterns × 2 layers × 2 heads × 20 decode positions = **640 correlated queries**, context 493–512, 16-dimensional head; 32 tokens per page. Both the geometric-score implementation and an **infeasible exact-score page-upper oracle** retained **all pages in every one** of those 640 queries. Neither saved KV reads or competed with dense on modeled per-query traffic. The exact-score diagnostic is not an implementable fast path.

No detected bound violations in **1,600 independent small fractional-maximization checks**, **480 random exact dense attention checks**, or **640 toy transformer checks**; 4 constructed controls and 24 synthetic sensitivity cases also completed. These are *numerical tests, not proof of float32/float16 correctness.* The exact-real-arithmetic inequality is given in the specification. In the 640 toy queries, selected KV entirely dense and measured error ~floating point rounding; zero stop certificate when no pages remain.

### Controlled sensitivity (2048 tokens, 64 pages, d=16, 3 seeds)

| Value noise σ | Median selected pages | Median modeled traffic |
|---|---:|---:|
| 0, .001, .003, .01 | 1 / 64 | 4.88% |
| .03 | 63 / 64 | 101.76% |
| .1, .3, 1.0 | 64 / 64 | 103.32% |

**Interpretation:** The favorable case is brittle to relatively small changes in value-vector dispersion. The threshold varies with vector norm and score distribution; the numbers are not a universal threshold. 4.10% and 4.88% include modeled *per-query reads* of both full key and value page summaries; **not** initial metadata computation or update work.

### Overhead and fairness

For a 512-token, 16-dimensional head and 16 pages, the model charges **1,088 additional bytes/query** for value summaries compared to the earlier mass-only method. Median modeled traffic: **101.69%** of dense for mass-only, **103.38%** for value-aware. Actual performance can be worse due to page-bound reductions, CPU scheduling, memory tier location and 55 root iterations after each page. Hot metadata might avoid repeated off-chip traffic, but then the savings cannot be credited as off-chip bytes without a tier-specific model. Neither baseline is a matched optimized GPU kernel, so end-to-end performance remains unknown.

The model reports a **32,768-byte full-value scan proxy** per 512-token trace as a minimum one-time value data movement if the summaries are not fused with prefill. This is a *scenario*, not necessarily additional DRAM cost; it is explicitly **excluded** from the percentages shown above. Ongoing active-page metadata update and root computation are likewise not included.

## Claim boundary and negative evidence

- **Mathematical statement:** In exact arithmetic, the box-relaxed per-page value-distance bound is a valid *sufficient* certificate; it can exploit closely coincident value vectors even when attention mass is diffuse.
- **Observed:** Narrow synthetic benefits exist, but trained-model benefit is untested, and the randomly initialized transformer and dispersed-value controls produce **no savings**.
- **Prior art:** Value-aware approximation and output-tail bounds already exist. See Gupta & Berant EMNLP 2021, vAttention ICLR 2026, Tzachristas et al. 2025, and VC-Attention Sep 2026 (links in research notes). This implementation is **not established as novel**.
- **Not proven:** Correctness under finite precision or stale metadata; physical traffic; speed/power/area; model quality; broader workload benefit; genuine architectural novelty; freedom to operate.

## Gate status and next action

| Gate | Status |
|---|---|
| G0 | PASS — narrow working charter |
| G1 | IN PROGRESS — prior art overlaps substantially |
| G2 | No new processor concept survives gate; previous anchor-delta thesis rejected |
| G3 | IN PROGRESS — exact-arithmetic bound derived; numeric tests pass; finite-precision proof missing |
| G4 | IN PROGRESS — reproducible simulations; negative performance result; full system-cost model missing |
| G5 | BLOCKED — no pretrained long-context traces / task-quality / optimized matched baselines |
| G6–G12 | NOT STARTED — hardware development not authorized by evidence |

**Next:** Do not optimize this synthetic favorable case for a chip. Obtain permission/means to evaluate actual pre-trained KV traces and compare strongest existing methods on realistic contexts; otherwise pivot from sparse attention to a different well-defined processor bottleneck and re-open G1/G2. Reject any *breakthrough* claims pending independent evidence.
