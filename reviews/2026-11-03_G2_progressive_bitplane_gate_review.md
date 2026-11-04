# Engineering gate review — progressive bitplane certificate

**Date:** 2026-11-03. **Status:** G2 mathematical/reference functionality PASS under integer constraints; G1 novelty FAIL for generic primitive; G4 performance FAIL in this Python implementation; G5 real pretrained LLM remains BLOCKED; G6+ NOT AUTHORIZED.

## Evidence of work

- Read the full governing `AGENTS.md` and last near-memory research gate review.
- Verified relevant 2020–2026 prior art; strong overlaps include ProgressiveNN, BitSET, 2024 digit-plane early termination, BitFair, and a very close 2026 SATIN certificate.
- Implemented fully packed MSB-first 8-plane INT8 reference and conservative integer interval certificate. Added randomized signed-input controls, exact tie tests, all-int8 extremes, packed-byte invariants, and 2 independently trained digit-model seeds.
- Ran 3,000 randomized prediction checks, 1,080 held-out digit predictions, and 768 targeted synthetic decisions. All 4,848 main comparisons agreed with their full INT8 counterparts. The nine automated unit tests passed (`9 passed in 0.07s`); randomized cases complement but do not prove all input cases. Integer interval certificate derivation is in `specs/G2_progressive_bitplane_certified_primitive.md`.
- Two logistic-regression trained models converged in 146 and 159 optimizer iterations respectively (no convergence warnings observed).

## Measured/local results

| Held-out digit seed | INT8 accuracy | Progressive agreement with INT8 | Average planes read | Ideal weight-bit bytes saved | CPU time vs NumPy dense |
|---|---:|---:|---:|---:|---:|
| 17 | 96.48% | 100% | 5.793 / 8 | 27.59% | 25.1x slower |
| 29 | 97.59% | 100% | 5.896 / 8 | 26.30% | 30.4x slower |

The full-precision logistic classifier happened to match the quantized classifier's held-out **accuracy fraction** in each seed; that does **not** guarantee prediction-by-prediction equivalence. Exact agreement asserted is **only progressive versus quantized INT8**.

**Uncertified 4-plane midpoint baseline:** 95.00% / 95.74% accuracy and 97.04% / 97.78% agreement with full INT8 respectively, reading a fixed 50% of weight bitplanes. The certified candidate spends more bit reads to preserve the INT8 prediction. At 6 planes fixed the agreement is 99.26% / 99.07%, still not exact.

**Falsifiers / negative controls:** identical class rows and biases required 8/8 planes in 256 signed synthetic inputs; a deliberately dominant bias needed only 1/8 planes. Other random signed inputs averaged 6.13–6.28 planes. No universally useful savings, even in an ideal bitplane-memory model.

## Limitations and risk

- Not a transformer or LLM workload; digits is an isolated small linear classifier (10x64). No pretrained model, GPU kernel, silicon, power measurement, process node, or area/timing closure.
- The model counts ideal weight-plane loads, not physical memory transfers; initial bitplane packing and low-level instruction/control costs are excluded from the ideal byte fraction.
- Python timing is implementation-dependent and massively slower, so there is **no measured processor advantage**. The dense NumPy comparison is not an optimized modern inference engine.
- A classifier argmax contract is not general neural-network inference. A generic LLM needs hidden-vector values, not merely their top-1 index.
- Prior art is close enough that the generic mechanism does not qualify as a novel design. An exhaustive patent review has not been performed.

## Gate decision

- **G2 (reference correctness):** PASS only under stated int64 bounds and for exact INT8 argmax.
- **G1 (novelty):** FAIL for generic progressive bitplanes + certified stopping. Close it as a proposed invention; retain benchmark and negative cases as reference.
- **G4 (performance):** FAIL for this Python code; hardware benefit not evaluated, physical feasibility remains open.
- **G5:** BLOCKED (no representative pretrained LLM traces). **G6–G12:** NOT AUTHORIZED / not demonstrated.

**Next discriminating gate:** Do not optimize this same non-novel method. First require a materially distinct mechanism with explicit changes relative to 2026 SATIN/BitFair (e.g., a new physical state representation or communication principle) and a complete energy/area/memory cost story before continuing hardware work.

## Reproduction

```
cd /mnt/data
python tests/g2_progressive_bitplane_audit.py
python -m pytest -q tests/test_g2_progressive_bitplane.py
```

Evidence: `results/g2_progressive_bitplane.json`; `results/g2_progressive_bitplane_stdout.txt`; `results/g2_progressive_bitplane_pytest.txt`. Governing `AGENTS.md` unchanged. All engineering lenses are analytical, not independent specialist sign-off.