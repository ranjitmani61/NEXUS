# Engineering gate review — Drift-aware optical calibration and energy

**Version:** 1.0 · **Date:** 2026-11-03 · **AGENTS.md:** read in full before work; unchanged. **Review lenses:** A1, A2, A6, A7, A8, A10, A11; no independent physical-device specialist. **Evidence:** `results/g2_photonic_drift_energy_audit.json`, source code, tests and console log. **Research type:** exploratory simulation, not a pre-registered study.

## Decision first

**G1 (generic optical calibration novelty): FAIL / NOT ESTABLISHED. G2: CONDITIONAL PASS for idealized Poisson drift-bound certificate. G3: numerical/software tests only. G4: simulated costs; no full-system verification. G5: BLOCKED on real workloads. G6+: NOT AUTHORIZED. No new hardware or physical breakthrough.**

## Test evidence

- 36 scenario combinations, 2 independent random seeds, 40 cycles/seed, K in 1/8/64/256: **236,880 simulated readouts**.
- Under the model with the correct maximal multiplicative drift interval, **zero false admissions observed**, and `126,560` readouts were admitted. This is not proof of a zero failure probability.
- Deliberately ignoring actual drift yielded **48,908 false admissions** on exactly the same simulated data. This is a negative control, not a comparison of two equally safe algorithms.
- A deliberately unmodeled post-pilot `+2%` gain jump yielded **640 false admissions out of 640 accepted readouts** despite nominally 'guarded' operation. The assumed gain-bound theorem does not apply in this situation.
- The existing and new regression tests completed **52/52 passing** (9 new plus one new jump control + 42 inherited suites? See the exact `pytest` output; no independent formal-verification theorem checker was run).
- Output JSON captures every scenario/seed; scripts and commands are in the specification.

### Illustrative sensitivity

All the following use the *artificial high drift* `rho=10^-4` **per readout**, exact pilot Poisson model and `1%` relative error threshold. Admission is a model result, not real-world throughput.

| Calibrated readouts K | cancellation kappa | Safe admissions | Unsafe admissions with ignored drift (false count) | Illustrative photon-plus-electronics energy per readout |
|---:|---:|---:|---:|---:|
| 8 | 1 | 100% | 0 | 10.81 nJ |
| 64 | 2 | 62.5% | 1,158 | 10.67 nJ |
| 64 | 10 | 0% | 3,306 | 10.66 nJ |
| 256 | 1 | 35.55% | 12,526 | 10.67 nJ |
| 256 | 2 | 15.23% | 16,512 | 10.66 nJ |
| 256 | 10 | 0% | 15,400 | 10.65 nJ |

At kappa 10 and K>=64, the family-wise confidence requirement admitted **zero** even when drift rate was zero in this synthetic setup. Pilot reuse reduces the nominal detected-photon ratio from `3.00x` (K=1) to `1.0078125x` (K=256) relative to nominal data photons, but not necessarily useful **certified** work.

### Energy is not measured

Photonic energy numbers assume wavelength=1550nm, 20% wall-plug efficiency, 10% link/detection efficiency, and **10 nJ arbitrary per-reduction electronics**. No laser/receiver/component power was measured. The result is a toy inequality with a fixed cost assumption, not a physical energy estimate reliable enough for a design decision. For K=256, kappa 1, rho=10^-4, 35.55% certification, the all-digital energy would need to exceed **~30.0 nJ per matched operation** merely to pass this incomplete toy model's break-even; real optical costs could be much higher. No matched digital energy measurement exists. **Do not report this as an efficiency advantage.**

## Risks, omissions, and falsification

1. **Calibration freshness:** The true gain may jump or oscillate faster than declared; a conservative smooth bound can be false.
2. **Workload:** Pure signed scalar reduction with chosen cancellation factors is not a real LLM interconnect workload.
3. **Counts:** Real detectors may show correlated, overdispersed, dark-count, saturation, nonlinear or thermal behavior incompatible with independent Poisson modeling.
4. **Calibration standard:** `alpha` and pilot-exposure scales are assumed perfectly known, as is their consistency over time.
5. **Safety semantics:** `delta_cycle=0.001` is a family-wise risk budget **per calibration cycle**, not across years of operation. Across C cycles a simple union bound gives `min(1,0.001*C)`, usually vacuous at scale.
6. **Floating point:** Garwood CIs and admissibility use scipy float64, with no formally outward-rounded implementation for hardware.
7. **Fair baseline:** No calibrated optical hardware or measured matched-error digital collective exists. Per-readout communication setup, actual time, optical equipment and fallback latency are unmodeled.
8. **Novelty:** Known Poisson CIs and established optical calibration do not amount to a new principle.

## Reproduction

```bash
cd /mnt/data
python tests/g2_photonic_drift_energy_audit.py --output results/g2_photonic_drift_energy_audit.json
PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_drift_energy_audit.py tests/test_g2_photonic_count_certificate.py tests/test_g2_photonic_signed_reduction.py tests/test_g2_progressive_bitplane.py
```

**Next gate:** Do not optimize this for silicon. Need either a physically measured characterization of gain drift, laser and receiver costs with a matched digital baseline, or a distinct non-prior-art primitive that justifies further development. Before meaningful system claims, establish a global risk budget and a full reproducible hardware/LLM experiment.