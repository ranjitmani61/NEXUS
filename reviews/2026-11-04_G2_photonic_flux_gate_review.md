# Project 01 — G2 photonic photon-flux and electrical-current gate review

**Date:** 2026-11-04; **status:** completed analytical/software audit; **governing policy:** `/mnt/data/AGENTS.md` read before work and unchanged.

## Decision

**HOLD / effectively suspend the present optical reduction branch.** We have not invented a distinct mechanism, fabricated a device, measured a matched baseline, or verified physical feasibility. The numerical findings here are valid only for the chosen *independent Poisson photon-counter / direct-difference estimator / efficiency* model. They do **not** constrain all possible photonic or quantum optical designs. **G1: generic novelty not established; G2: mathematical test PASS under assumptions, physical feasibility HOLD; G5: blocked; G6: not authorized.**

## Engineering lenses

A2 (numerical analysis), A7 (photonic/electrical feasibility), A8 (software checks), A10 (system costs), A11 (prior-art context). No independent electrical/device engineer or optics specialist participated; all are evaluation roles performed in a single AI-assisted analysis.

## Mathematical method and limits

Assume independent Poisson detected counts `P ~ Pois(lambda+)`, `M ~ Pois(lambda-)`, with `lambda+ + lambda- = N` and nonzero signed difference `lambda+ - lambda- = N/kappa`. Difference estimator `D=P-M` has variance `Var(D)=N`. Thus relative RMS noise, with the true nonzero signed difference as denominator, is exactly `sqrt(N)/(N/kappa) = kappa/sqrt(N)`. For an **RMS** target epsilon, `N >= (kappa/epsilon)^2` under *this estimator*. This is **not** an upper bound on the probability of a large relative error and not a physical lower bound for any quantum light source or arbitrary estimator.

For wavelength 1550nm, photon energy `Eph=h*c/lambda` with exact defining SI constants from [NIST](https://www.nist.gov/pml/special-publication-330/sp-330-section-2). If we **assume** 10% link/detection and 20% source electrical-to-optical efficiencies, illustrative electrical photon-generation energy is `N*Eph/(0.1*0.2)`. If R reductions/s, aggregate expected detector-charge flow is `I=e*N*R` and assumed source electrical power is `P=R*N*Eph/(0.1*0.2)`. The current aggregates both output channels; physical distribution across detectors remains unmodeled.

Shot-noise Poisson idealization follows conventional coherent photodetection theory. Real photodetectors can violate naive Poisson assumptions (e.g. [Optics Letters 2020](https://opg.optica.org/ol/abstract.cfm?uri=ol-45-11-3009)). Novel/quantum optical sources may alter the noise relation.

## Results from predeclared matrix

24 combinations: cancellation `kappa={1,10,100,1000}`, relative RMS `epsilon={1%,2%}`, sustained `R={1e6,1e8,1e9}` reductions/s. For the 1% epsilon, 100M ops/s slice:

| kappa | N detected/op | Electrical photon-generation energy (assumed) | Aggregate detector current | Source electric power (assumed) |
|---:|---:|---:|---:|---:|
| 1 | 1.0e4 | 0.00006408 nJ/op | 0.0000001602 A | 0.00000641 W |
| 10 | 1.0e6 | 0.006408 nJ/op | 0.00001602 A | 0.000641 W |
| 100 | 1.0e8 | 0.6408 nJ/op | 0.001602 A | 0.06408 W |
| 1000 | 1.0e10 | 64.08 nJ/op | 0.1602 A | 6.408 W |

In the `kappa=1000, eps=1%` corner, photon-generation electricity **alone** would exceed a **hypothetical** digital comparator costing 30 nJ/op, even before paying for conversion, calibration, static power or fallback. This is an **assumption-dependent no-go corner**, not evidence that all photonic reductions are inefficient; neither kappa distribution nor actual matched digital energy has been measured. **A hypothetical 10 mA total-detector-current threshold permits only ~6.24 million operations/s** in that corner under the two-channel model. 10 mA is not a sourced component constraint. Splitting across detectors can change per-device current and complicate the system-cost model.

The previous published 320 mW and 1.112 W examples refer to **different** photonic devices and are *not* matched measurements or transferable lower bounds. Current test excludes them from its energy table. The prior simulated static-power test and optical precision simulation remain intact.

## Verification actually executed

- `python tests/g2_photonic_detector_flux_audit.py --output results/g2_photonic_detector_flux_audit.json`
- `PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_detector_flux_audit.py` -> **28/28 tests passed**.
- `python tests/g2_photonic_detector_flux_audit.py --output results/g2_photonic_detector_flux_audit_repeat.json` -> machine-readable output **byte-for-byte identical**.
- 3 seeded Monte Carlo controls, **40,000 ideal Poisson two-channel readouts each**; observed relative RMS deviated by **0.31%, 0.72%, 0.15%** from analytical predictions at cancellation 1/10/100 with N=100,000. These checks validate software/model consistency, not physical detector behavior.
- All 24 analytical cases, input-validation negative controls, exact units and sensitivity tests retained.

## Nontransferable assumptions and missing evidence

1. Real photonic detector linearity, saturation, maximum mean count rate and shot-noise spectrum.
2. Actual optical wavelength, device wall-plug efficiency, link/detection efficiency, ADC/DAC, thermal/laser-control and calibration overhead.
3. Tail risk and lifetime risk: **RMS criterion is not an end-to-end reliability certificate**.
4. Real communication workload and measured cancellation factors (kappa) for its reductions.
5. A **matched** conventional digital implementation's energy, latency and quality, and a matched photonic device.
6. Actual decomposition of 0.160 A aggregate charge flow across photodetectors and electrical interfaces.
7. Device-specific mechanism to distinguish the project from published optical AllReduce and noise-calibration techniques.

## Next action / stop rule

Stop iterative photonic simulations until **either** (a) measured matched device and digital performance plus a physically distinctive primitive are available, **or** (b) a different, genuinely distinct hardware/computation hypothesis is selected. No G6 microarchitecture or chip authorization. Advance instead to a new hypothesis-selection gate with predeclared benchmark and prior-art comparison. Preserve negative results; do not relabel the hypothetical no-go as an impossibility theorem.

**Artifacts:** `specs/G2_photonic_detector_flux_contract.md`, `tests/g2_photonic_detector_flux_audit.py`, `tests/test_g2_photonic_detector_flux_audit.py`, `results/g2_photonic_detector_flux_audit.json`, `results/g2_photonic_detector_flux_console.txt`, `results/g2_photonic_detector_flux_pytest.txt`, and repeat JSON/log.
