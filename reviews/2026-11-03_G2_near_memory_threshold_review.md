# Gate review — G2 near-memory throughput screening

**Date:** 2026-11-03  **Status:** G2 analytical model PASS within stated assumptions; G1 novelty NOT ESTABLISHED; G5 BLOCKED; G6 NOT AUTHORIZED.

**Preflight:** Read `/mnt/data/AGENTS.md` (195 lines), project research log and original baseline; read the latest G4 queue tradeoff review. No design or metric contract was silently changed. Roles: A1, A2, A6, A8, A10, A11; no independent specialist reviewers.

**Question:** Is avoiding weight transfers across a memory-to-compute interface enough to guarantee faster single-token inference?

**Answer: NO.** The internal bandwidth, arithmetic throughput, bank communication, and per-layer sequencing requirements can negate the advantage.

## Actual work done
- Created deterministic transparent threshold model in `tests/g2_near_memory_threshold.py`.
- Evaluated **80 preselected** combinations (4 internal BW × 5 PIM compute capacities × 4 launch costs), plus 8 explicit negative/control cases.
- Ran `python -m pytest -q tests/test_g2_near_memory_threshold.py`: **18 passed**.
- Saved full input values and per-case computed quantities to `results/g2_near_memory_threshold.json`.
- No real hardware or pretrained model was run; values are **modeled** not measured.

## Derived conditional results (hypothetical, not empirical speedups)
- GPU baseline roofline `T_host` = **5.0966 ms** at BF16 B=1 T=8192 using host 3.35 TB/s and 989 TFLOP/s as illustrative peaks.
- Nominal hypothetical PIM (internal BW=8 TB/s, compute=10 TFLOP/s, launch=5 µs/layer, 64 bank fanouts): `T_near=2.4856 ms`, modeled ratio **2.0505×**. This is a sensitivity input, not a real PIM specification.
- PIM with only **1 TFLOP/s** modeled compute: `T_near=16.4856 ms`, ratio **0.3092×** — loss.
- **100 µs** coordination/layer: ratio **0.9224×** — loss.
- Internal BW only equal to host **3.35 TB/s**: ratio **0.9686×** — loss.
- B=32 with 10 TFLOP/s near-memory arithmetic: ratio **0.2433×** — loss.
- 80-case grid: **19** cases with ratio >1; **9** with ratio >=2; **61** with no modeled win. These frequencies reflect arbitrary scenario selection, not the likelihood of success.
- In nominal case per-layer launch break-even is approximately **86.59 µs**, conditional on all other assumed values and formula.

## Counterarguments and limitations
- The host reference is an optimistic analytical lower bound, **not an optimized measured H100**, and the near-memory cost model omits real physical costs; the ratio is not a valid real-world performance forecast.
- BF16 model quality, compile/runtime cost, bank utilization, control traffic, heat, error correction, silicon area, PPA, die cost, and device energy are **not measured**.
- Prior art from 2025–2026 already covers analog in-memory compute, heterogeneous HBM/SRAM PIM, and GPU/FPGA heterogeneous processing.
- The study is an **early feasibility filter**, not evidence of a new invention or G5.

## Decision / next discriminating evidence
- **G2 arithmetic:** Narrow PASS (18 tests, deterministic checks); physical feasibility not proven.
- **G1 novelty of generic PIM:** FAIL. No differentiated mechanism exists yet.
- **G4 end-to-end comparison:** Not attempted for this design.
- **G5:** Blocked — representative pretrained-model measurements still missing.
- **G6–G12:** Not authorized / unproven.
- **Next scientific step:** specify a distinct, physically implementable primitive that goes beyond published near-memory MAC and quantify energy, interconnect, thermal and error costs against named 2026 PIM implementations, *or reject the branch*. Require a real GPU/practical PIM device baseline before claiming any speedup.

## Reproduction
```
python tests/g2_near_memory_threshold.py
python -m pytest -q tests/test_g2_near_memory_threshold.py
```
No background execution, no paid compute, no hardware test, no external expert review.
