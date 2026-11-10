# G2 test contract — detector-count self-certification, calibration pilots, and photon overhead

**Version:** 1.1 · **Date:** 2026-11-03 · **Roles:** A1/A2/A6/A7/A8/A10/A11 review lenses, no independent physicist. **Status:** hypothesis/model test, not hardware.

## Falsifiable objective

Test whether exact Poisson-count confidence intervals remove the need for an independent `L1 = sum(abs(inputs))` collective in a *hypothetical signed photonic reduction*, without hiding detector calibration cost. Compare the following fixed five paths:

1. **Known gain (ideal oracle):** both detector gains exactly known to be 1, no pilot photons; inputs balanced.
2. **Blind gain:** true opposite drifts `+2%` and `-2%`, but confidence calculation incorrectly assumes perfect gain. This is an intentionally unsafe negative control.
3. **Bounded gain:** true drifts `+2%` and `-2%`, with specified known `+/-2%` gain uncertainty but no pilots.
4. **Pilot small:** drifts `+2%/-2%`; independently sampled per-channel known-amplitude reference pulses of `10^6` expected photons at unity gain each.
5. **Pilot large:** same, with `10^8` reference photons *per channel*, charged in the total optical photon count.
6. **Stale pilot (unsafe control):** large pilot reads `+2%/-2%` gains, but the subsequent data measurement gains experience an additional, unmodeled common `+2%` shift. This tests calibration freshness, and is expected to invalidate the pilot confidence guarantee.

Data photon scale `alpha=10^8` counts per unit positive or negative magnitude, input `L1=1`. Equal representative *only for experiment* `kappa=[1,2,10,100,1000]`. Every path uses three random seeds `20261113, 20261114, 20261115`, `1000 trials/condition/seed` (15,000 per scenario). Fixed `relative_eps=.01`, unconditional per-input false-admission risk bound `delta=.001` (not a claim of a conditional risk bound given acceptance) in the ideal independent-Poisson model. Distinguish confidence, not deterministic absolute correctness.

**Amendment rationale (v1.1):** Added a deliberate *post-calibration drift* negative control to test the hidden assumption that calibration pilots and data see identical detector gains. This was chosen before running the amended simulation and its outcome must not be used to assess novelty.

**Output acceptance rule:** Given confidence interval `[s_lo,s_hi]` for true signed sum and a point estimator `hat_s`, only admit when *all* signed sums in the interval have the same nonzero sign and `abs(hat_s - s)/abs(s) <= eps`. Implement via endpoint ratio or equivalent sign-specific inequalities. For a simultaneous confidence set covering the two detector means and (if present) the two calibrated detector gains with probability >=1-delta, the event of a falsely admitted result has probability <=delta for any fixed inputs and allowed gains. This statement assumes independent Poisson measurements and exact interval calculations.

### Confidence intervals and scaling

Data counts: `Cpos ~ Pois(alpha * gpos * P)`, `Cneg ~ Pois(alpha * gneg * M)`. Known P/M nonnegative, `P-M=s`, `P+M=L1` (L1 is **not supplied to the verifier**).

For a Poisson count `C`, Garwood two-sided `1-beta` CI for rate `mu`:

`lower = 0.5*chi2.ppf(beta/2, 2*C)` for C>0 (else 0), `upper = 0.5*chi2.ppf(1-beta/2, 2*(C+1))`.

With no pilots, each channel has `beta=delta/2`; gain is either exactly known or known in `[1-h, 1+h]` with `h=.02`. Use nonnegative interval arithmetic to infer P and M, then take their interval difference.

With two pilot channels, use **four** simultaneous CIs with `beta=delta/4` each. Pilot `Rpos~Pois(H*gpos)`, `Rneg~Pois(H*gneg)` where `H` is a *known* calibration exposure, and obtain gain CI by dividing mean CI by H. Intersect only with physical gain >0; a zero lower endpoint forces fallback. No extra L1 collective is used, but two optical pilot readouts are added.

The pilot guarantee **requires the calibration gain to equal the subsequent data gain**. If gains drift after the pilot, the interval is invalid; a purposely failing post-pilot control must reveal this. The fixed intensity-to-count scale `alpha` and the pilot calibration exposure `H` are assumed externally known; the method removes a separate `L1` collective, **not every calibration requirement**.

The estimate uses per-channel point gain estimates when pilots are present, otherwise nominal unity gain. Confidence and estimates are separate functions; the estimate must not use the unknown true gain.

### Metrics and essential negative cases

- Fraction admitted and false admitted for each kappa and each path, and CI coverage rate.
- Expected data + pilot detected photon counts; relative *photon* overhead vs the no-pilot `alpha * L1` baseline, excluding laser inefficiency, transport loss and electronics.
- Adversarial gain mismatch and exact/near cancellation, zero counts, low count scale.
- Programmatic regression tests: interval arithmetic, zero/negative handling, beta allocation, known-drift control, pilot data, reproducible seeds.
- Do not claim this mathematical self-certification is novel: Garwood intervals, balanced photodetection, optical AllReduce and calibration all have prior art. Compare with existing method; no actual NCCL/hardware measurement.

### Sources (prior art)

- Garwood exact Poisson confidence intervals: https://pmc.ncbi.nlm.nih.gov/articles/PMC3964675/
- OFC 2026 photonic AllReduce: https://opg.optica.org/abstract.cfm?uri=OFC-2026-Th3H.3
- 2026 photonic tensor processor balanced readout and calibration: https://doi.org/10.1038/s41467-026-71599-2

### Reproduction

`python tests/g2_photonic_count_certificate.py --output results/g2_photonic_count_certificate.json`

`python -m pytest -q tests/test_g2_photonic_count_certificate.py tests/test_g2_photonic_signed_reduction.py`
