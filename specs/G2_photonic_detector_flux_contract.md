# G2 — Photon-flux, detector-current and energy audit (predeclared protocol)

**Version** 1.0; **date** 2026-11-04. **Owner/lenses:** A2 mathematics, A7 electro-optics, A8 verification, A10 system performance, A11 prior art. No independent optics specialist or measured photodiode.

## Objective / status
Stress-test the previous G2 photonic reduction proposal at fixed *numerical precision* and sustained operation rate. This experiment is an analytical falsification screen, **not** a hardware test and **not** a general quantum lower bound. Existing photonic-reduction hardware is prior art; novelty not asserted.

## Fixed assumptions before running

- Exactly two **independent, unsqueezed Poisson-counting channels**, fixed gain, sum represented by their difference, total expected **detected** photon number `N` per reduction.
- Cancellation factor `kappa = (mean positive + mean negative)/(absolute difference)`, `kappa >= 1`; net signal nonzero.
- Relative RMS noise criterion `eps`, and direct difference-of-counts estimator; then `relative RMS=kappa/sqrt(N)`. Therefore **within this specified estimator** `N_required = (kappa/eps)^2` is exact for target shot-noise RMS; not for a tail-failure guarantee.
- Wavelength = 1550 nm, link+detector efficiency = 0.10, electrical-to-launched-light efficiency = 0.20. These efficiencies are *illustrative design assumptions*, not measurements.
- Ignore static heater, clock, driving electronics, ADC/DAC, communications, optical loss beyond specified 10%, beam-splitter losses already assumed in efficiency, calibration pilots, fallback and thermal costs. Thus calculated electrical photon-generation energy is **optimistic for this channel model**. Quantum noise below independent Poisson may be possible with other sources.
- Illustrative hypothetical total digital cost: 30 nJ per matched reduction; not measured. Aggregate detected-photon flux can be distributed over multiple detectors, so computed electron current is a **sum equivalent**, not necessarily current through one component.
- Constants `h=6.62607015e-34 Js`, `c=299792458 m/s`, `e=1.602176634e-19 C` are exact SI definitions (NIST): https://www.nist.gov/pml/special-publication-330/sp-330-section-2
- Independent Poisson photo-counting is an idealization; counterexamples from real photo detectors exist: https://opg.optica.org/ol/abstract.cfm?uri=ol-45-11-3009

## Fixed test matrix, before execution

1. `kappa in [1, 10, 100, 1000]`, `eps in [0.01,0.02]`, `R in [1e6,1e8,1e9]` operations/s. Compute photons/reduction, detected photon energy, assumed source electrical energy, electron current, dynamic electrical power.
2. Quantify 10 mA aggregate-current reference cap at each kappa for eps 1%; **10 mA is a sensitivity threshold only**, not a sourced device rating.
3. Test analytic monotonicity (kappa, eps, efficiency, throughput), units, and ideal zero-power cases; compare `eps 1%, kappa 1000` photon-generation cost with the hypothetical 30 nJ digital comparator without any static/electronics charges.
4. Independently simulate 40,000 Poisson pair reductions for `(kappa,N) in [(1,1e5), (10,1e5), (100,1e5)]`, seed `20261104`, verify the RMS relative error is within 5% of exact theory. These are **simulation checks**, not physical validation.
5. Unit tests validate consistency of the output table, failure cases and argument validation. Record all cases and negative results.

## Pass / fail criterion

- Pass *arithmetic audit* only if all unit tests pass and Monte Carlo agrees within 5% under idealized assumptions.
- Existing generic photonic architecture **remains G1 novelty not established** regardless of numerical findings.
- Mark an **illustrative no-go region** if *assumed* wall electrical photon-generation energy alone exceeds assumed matched digital energy, noting sensitivity to efficiencies and actual workload cancellation.
- G2 physical feasibility remains HOLD until measured matched device energy, speed, noise/drift, actual cancellation distributions and a real digital baseline exist. G5 and G6 remain blocked.

## Reproduce
`python tests/g2_photonic_detector_flux_audit.py --output results/g2_photonic_detector_flux_audit.json`
`PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_detector_flux_audit.py`