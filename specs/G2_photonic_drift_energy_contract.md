# G2 exploratory contract — Shared photonic calibration under time-dependent drift

**Version:** 1.0 · **Date:** 2026-11-03 · **Status:** exploratory research protocol documented with the experiment; **not preregistered** · **Research lenses:** A1 architecture, A2 applied mathematics, A6 communication, A7 optics/electronics assumptions, A8 verification, A10 systems energy, A11 prior art. **No independent optics/hardware reviewer.**

## Objective and candidate

Determine whether a shared Poisson detector-gain calibration pilot can be reused for `K` signed optical reductions while providing a *family-wise*, probabilistic output-error certificate and amortizing optical pilot energy. The alternative is using the same pilot certificate with no gain-drift guard. Both use prior-art techniques; this is a **feasibility / failure-mode** study, not a novelty claim.

## Reproduction

```bash
cd /mnt/data
python tests/g2_photonic_drift_energy_audit.py --output results/g2_photonic_drift_energy_audit.json
PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_drift_energy_audit.py tests/test_g2_photonic_count_certificate.py tests/test_g2_photonic_signed_reduction.py tests/test_g2_progressive_bitplane.py
```

Source: `tests/g2_photonic_drift_energy_audit.py`. Prior model: `tests/g2_photonic_count_certificate.py`.

## Declared test matrix (exploratory)

- Each signed sum is `s = P-M`, `P+M=1`, `s=1/kappa`, `kappa∈{1,2,10}`.
- Expected detected count scale `alpha=10^8`; two pilot channels `H=10^8` count exposures each.
- Baseline gain at pilot is `g+=1.02`, `g-=0.98`. The simulator generates independent Poisson pilots and readouts.
- `K∈{1,8,64,256}` data readouts after each calibration; 40 cycles/seed, seeds `20261103`, `20261104`.
- **Assumed bounded drift:** data gains `g+(j)=g+(0)(1+rho*j)` and `g-(j)=g-(0)(1-rho*j)` with `rho∈{0,10^-5,10^-4}` per readout and `j=1,...,K`. These rates are artificial stress parameters, **not measurements in seconds**. The guard is `±rho*j` around the pilot gain CI.
- Compare `guarded` (true bound used) vs `stale_unsafe` (bound wrongly set to zero with same changing gains).
- Relative output tolerance `epsilon=1%`; family-wise false-acceptance bound `delta_cycle=0.001` *per K-readout calibration cycle*.
- Separate adversarial test: same data input, gains jump by `+2%` immediately after pilot; program incorrectly assumes max rate `1e-5` per readout. This deliberately violates the guard assumption.
- 36 (K, rate, cancellation) configurations × two seeds; 236,880 simulated readouts total, plus 640 abrupt-jump readouts.

## Mathematical confidence bound

For the **two pilot** count means, use Poisson Garwood two-sided intervals with per-channel noncoverage `beta_pilot=delta_cycle/4` (both together at most `delta_cycle/2`). For the `2K` data count means, use per-channel `beta_data=delta_cycle/(4K)` (all together at most `delta_cycle/2`). Take nonnegative gain intervals and widen by the **known absolute multiplicative drift bound** `1±rho*j`; interval arithmetic then bounds `P`, `M`, and `s=P-M`.

The estimate is calculated only from detector data and pilot point estimates; admission is true only if the entire possible signed sum interval yields `abs(est-s)/abs(s) <= epsilon` and all candidate `s` have the same nonzero sign.

**Per-cycle theorem, exact interval arithmetic:** Under the independent-Poisson marginal model, known data/pilot count scale, valid drift bound and correctly calculated Garwood intervals, union bound gives

`P(any false admission in one K-readout calibration cycle) <= 2*(delta_cycle/4) + 2*K*(delta_cycle/(4*K)) = delta_cycle.`

No independence between accepted decisions is required for this union bound. A probability bound for `N` calibration cycles is at best `min(1,N*delta_cycle)` without allocating a **global** risk budget. **This is not deterministic correctness**, a conditional risk bound given admission, or a guarantee if gains make unmodeled jumps, count noise departs from the Poisson assumption, `alpha` is uncertain, or arithmetic is implemented with unsafe rounding.

## Illustrative energy inequality — NOT measured

Assume wavelength `lambda=1550nm`; exact SI constants `h=6.62607015e-34 J*s` and `c=299792458m/s`. **Assumed, not observed:** wall-plug laser efficiency = 20%, combined link+detector efficiency = 10%, fixed non-photonic electronics cost per readout = 10 nJ, no energy cost assigned to digital fallback scheduling, analog encoding overhead, thermal tuning or storage.

For true mean detected data photons per reduction `D` and expected total two-pilot photons `Htot`, an illustrative per-reduction energy is

`E_opt(K) = (h*c/lambda) * [D + Htot/K]/(eta_wall * eta_path) + E_electronics`.

This is **not** complete laser, link, ADC/DAC, receiver or interconnect energy for a real system. `D` is calculated from *true* simulated per-channel gains (not simply assumed to equal `alpha`); pilot photons are amortized across the K reductions. Define simulated admission fraction `a`. If an unadmitted readout requires a digital fallback of energy `E_d`, then total modeled hybrid energy `E_opt+(1-a)E_d` beats all-digital energy `E_d` only if `E_d > E_opt/a`, for `a>0` and with no overlapping work. This is a **necessary modeled break-even**, not a verified comparison. A production optical implementation may have substantial additional costs, potentially changing the result completely.

Strong baseline required in future: measured digital collective on matched link, length, precision, fan-in, error criterion, throughput and receiver overhead (NCCL or equivalent) plus a credible integrated photonic design and its **actual** laser, calibration, modulation, detector, conversion and tuning energy.

## Sources / prior art

- NIST SI constants, https://www.nist.gov/pml/special-publication-330/sp-330-appendix-1
- Garwood Poisson intervals, https://pmc.ncbi.nlm.nih.gov/articles/PMC3964675/
- Litvin et al., *Robust calibration and energy optimization in reconfigurable photonic processors* (Optics Express, 2025), https://pubmed.ncbi.nlm.nih.gov/40984623/
- IEEE, *Calibration-Free, Self-Referenced Thermal Control Circuit for Silicon Photonics Microring Modulators* (2026), https://ieeexplore.ieee.org/abstract/document/11550344
- Project prior-art documentation: `research/G1_photonic_reduction_prior_art.md`.

## Hard limitations

No real time units or photonic drift measurements, no chip characteristics, no real optical power metering, no training/inference workload, no true digital energy comparison, and no independent patent novelty review. Unmodeled high-frequency drift or a calibration-source-scale error invalidates the theorem. This exploration alone cannot authorize architecture or chip development.
