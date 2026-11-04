# G2 physical-cost gate: static optical power and lifetime risk

**Version:** 1.0 · **Date:** 2026-11-04 · **Status:** analytical falsification screen, no photonic hardware tested  
**ACT review lenses:** A1 architecture; A2 numerical analysis; A6 interconnect; A7 electronic/photonic device assumptions; A8 software verification; A10 systems energy; A11 prior art. No independent photonics or silicon reviewer.

## Objective

Challenge two hidden assumptions in the previous photon-count certified AllReduce model: (1) negligible static power of real optical control components and (2) applying a 0.1%-per-calibration-cycle safety budget over long deployment lifetimes. Do not claim matched device results, novel technology, or GPU speedup.

## Source provenance and comparability

1. **Published 2026 photonic 16-tap convolution demonstrator:** Reports 16 heater phase-shifters estimated at **20 mW average each = 320 mW**. This is an illustrative photonic *heater-power reference*, not a measured full-system reduction device, and omits optical source/receiver and electronics. Source: https://www.nature.com/articles/s41377-025-02093-5
2. **Published 2025 integrated silicon photonic engine:** Reports **300 mW electrical laser** and **up to 812 mW phase-shifter electrical power**, or **1.112 W** in that upper-power configuration. This is a completely different physical engine, not our reduction unit or a lower bound for future optics. Source: https://www.nature.com/articles/s41467-025-60100-0
3. **2026 photonic AllReduce prior art:** Nick et al., *Breaking the bottleneck in AI clusters with parallel photonic integration*, Nature Communications, Aug 2026, DOI 10.1038/s41467-026-76764-1. Reports a four-projector/camera experiment integrated with GPT-2 gradient reductions, while *large-scale energy and latency benefits are predictive models*. Crucially, their model sometimes estimates **higher optical network energy than InfiniBand**, with lower total energy coming from reducing **GPU idle time**. Source: https://www.nature.com/articles/s41467-026-76764-1
4. **Our previous synthetic data:** The previous simulator's exact scenario `K=256, kappa=1, rho=1e-4 per readout` admitted 35.547% of 20,480 synthetic reads and calculated `E_dynamic=10.667009751096947 nJ/readout` from assumed photon and fixed-electronics parameters. Archived in `project_01_G2_photonic_drift_energy_package.zip`, `results/g2_photonic_drift_energy_audit.json` inside that ZIP. These are **assumed and simulated**, not optical device readings.

No device powers from (1) or (2) may be treated as the power consumption of (3) or of our prototype. Applying those values is a **cross-device stress test**, not an apples-to-apples comparison.

## Equations and units

Let `a` = fraction of reductions certified so that expensive digital fallback is avoided. Let `R` = actual sustained useful input reductions/s (not pulses/s or aggregate MAC/s), `P_static` = device power attributable to this workload in watts, `E_dyn` = dynamic optical path energy in nJ/readout, and `E_d` = **unknown** energy of a *matched* digital reduction including memory, link, compute and required accuracy.

- `E_static_nJ = 1e9 * P_static / R`.
- `E_hybrid_nJ = E_static_nJ + E_dyn + (1 - a) E_d`.
- `E_hybrid < E_d` iff `a * E_d > E_static_nJ + E_dyn`.
- Necessary break-even rate for `P_static>0` is `R > P_static / ((a E_d - E_dyn)*1e-9)` when `a E_d > E_dyn`; otherwise **no finite rate** helps.

**Accounting omissions:** data movement, modulation, ADC/DAC, real photon detection, re-calibration latency, laser startup, phase-change controls, receiver, clock distribution, peak power, chip area, cooling, optical component lifetime, actual throughput, digital baseline measurements, failure latency, and end-to-end model quality. Nothing here measures real PPA.

### Lifetime reliability

Previous certificate allocated `delta_cycle=0.001` for simultaneous errors over `K` readouts *per pilot cycle* under ideal Poisson + bounded-drift assumptions. Across `C` cycles, the basic union bound is `min(1, C*delta_cycle)` and is vacuous at C>=1000. A target `delta_global=0.001` across 1,000,000 cycles requires `delta_cycle<=1e-9`; for `K=256`, each data-channel tail noncoverage must be <=`delta_cycle/(4K)=9.765625e-13` if using the same allocation. This is a **budget calculation**, not a claim that such intervals yield useful admissions.

Even this fails if unmodeled gain jumps, correlations, sensor nonlinearities, scale errors, or malicious conditions violate the stated statistical assumptions. System reliability also depends on detection and recovery.

## Reproducibility

```bash
cd /mnt/data
python tests/g2_photonic_static_power_audit.py --output results/g2_photonic_static_power_audit.json
PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_static_power_audit.py
```

The program generates 10 hypothetical throughput/power rows, 8 digital baseline energy threshold rows, and 4 global reliability budgets. It uses published power *reference values* as explicit parameters. It does not access hardware and the software equations do not prove the assumed cost model or admission-rate transferability.

## Gate

**G2 physical feasibility: HOLD (not verified). G1 novelty: generic photonic collective already has close prior art. G5: BLOCKED. G6+: NOT AUTHORIZED.** Stop hardware-architecture claims without a measured full-device power/drift/accuracy characterization and competitive matched digital benchmark. At present this cannot justify a seventh processor class.