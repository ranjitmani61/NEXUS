# G2 specification — signed photonic accumulation, statistical precision and cancellation

**Version/date:** 1.0 · 2026-11-03  
**Owner / review lenses:** A1 architecture, A2 math, A6 memory/interconnect, A7 devices, A8 verification, A10 system performance, A11 novelty. No independent physics expert has signed off.  
**Current gate:** G1 novelty for generic approach = FAIL; G2 mathematical screen = CONDITIONAL; G3/G4 hardware implementation = NOT ESTABLISHED.

## Objective and falsifiable question

Explore replacing part of a digital multi-device signed reduction with a simultaneous optical reduction. Does the apparent single-operation communication advantage survive a requirement for **relative result accuracy** in cancellation-heavy workloads? Do not conflate this with measured optical energy, O(1) system latency, or a novel physical principle.

Strongest real implementation baseline remains optimized digital collectives (e.g. NCCL); **not run here**. A reference exact sum is used solely for numerical quality comparison. No hardware speedup is claimed.

## Assumed *ideal* measurement primitive

Let input samples be real signed `x_i`, with `P=sum(max(x_i,0))`, `M=sum(max(-x_i,0))`, `L=P+M=sum(abs(x_i))`, and `s=P-M`. Define cancellation condition number `kappa=L/abs(s)` if `s!=0`.

Assume ideal emitters and two independent Poisson photon-counting channels, calibrated perfectly, with `N` **total expected detected photons** for `L=1` test inputs (more generally fixed total photon budget per known `L`). Given `L`, observe:

- `C+ ~ Poisson(N*P/L)` and `C- ~ Poisson(N*M/L)`, independent.
- `s_hat = L*(C+ - C-)/N`.
- `E(s_hat)=s` and `Var(s_hat)=L^2/N`.
- Relative RMS error `sqrt(E((s_hat-s)^2))/|s| = kappa/sqrt(N)` for `s!=0`.

For any fixed `delta` in `(0,1)` and relative tolerance `eps >0`, Chebyshev yields a sufficient **probabilistic** error budget with `N >= kappa^2/(delta*eps^2)`. This is **not** a universal quantum lower bound and not an optimal statistical construction. At exact cancellation `s=0`, relative error is undefined; the all-analog relative-error objective cannot be certified.

A runtime guard should not use the unavailable true sum `s`. With known `L`, assume photon budget `N`, risk `delta`, and maximum relative per-channel gain mismatch `g`, then set:

`b = L * (sqrt((1+g)/(N*delta)) + g)`.

For the restricted mismatch model (positive gain `1+g_actual`, negative gain `1-g_actual`, with `|g_actual|<=g`), `|E(s_hat)-s|<=g*L` and `Var(s_hat)<=L^2*(1+g)/N`. The guard accepts only when `|s_hat| > b` and `b/(|s_hat|-b) <= eps`. Under these explicit assumptions, the probability of **accepting an inaccurate result** is at most `delta` for any fixed input. It is **not** a deterministic exact-output guarantee.

## Known physical omissions and very important costs

- **Known L**: the guard and this fixed-total-photon normalization require access to `sum(abs(x_i))`. Obtaining exact `L` in a distributed system may itself require a reduction or extra channels; its latency/energy are **not counted**. The chosen normalized optical encoding is therefore optimistic.
- Positive/negative channel emission, laser wall-plug efficiency, fanout, photon losses, modulation, ADC/DAC conversion, photodiodes, calibration, saturation, nonlinear optics, coherence, drift, temperature effects, shot noise beyond ideal independent Poisson processes, and error-correcting reconstruction are not physically designed or measured.
- True model outputs such as BF16/FP16 tensors, attention kernels, and signed cancellation statistics are unmeasured. Accuracy measured against a dense reference is on synthetic 2-term sums only.
- An optical collective that computes a scalar sum is not automatically a general programmable CPU/GPU/TPU replacement.
- Nonclassical light and different sensors can change the Poisson model. Do not promote the `kappa^2` requirement to a universal law of all possible optics.

## Predeclared experiment

- Poisson budget `N=100,000,000` detected photons, normalized inputs `L=1`.
- Cancellation `kappa=[1,2,10,100,1000]` equally weighted for an **illustrative**, not representative, workload mix.
- Relative tolerance `eps=.01`, accepted-result failure risk `delta=.001` under the Poisson model.
- Three independent pseudorandom seeds 20261103, 20261104, 20261105, each 3,000 trials per kappa (45,000 independent base measurements), plus 3,000 mismatch trials per seed (9,000 total).
- Adversarial miscalibration: opposite channel gain offsets at `+/- .02`, with and without the needed 2% guard.
- An *illustrative only* digital-fallback cost model: `analog_probe=0.15` units, exact digital=1 unit, fallback after analog probe costs `0.15+1`; normalized mean `E=0.15+(1-accept_rate)`. No measured electrical energy, latency, laser power, or real NCCL baseline.
- Tests: analytic identities, stochastic first two moments, edge cases, guard response to missing drift budget, and repeated seeded runs.

## Reproduction

```bash
cd /mnt/data
python tests/g2_photonic_signed_reduction.py
python -m pytest -q tests/test_g2_photonic_signed_reduction.py tests/test_g2_progressive_bitplane.py
```

Independent seed outputs: `results/g2_photonic_signed_reduction_seed_20261104.json`, `results/g2_photonic_signed_reduction_seed_20261105.json`.

**Acceptance criterion for architectural investment:** Real signed tensor distributions from pretrained networks, complete digital-baseline comparison (latency, energy, conversion, extra L1 acquisition), manufacturable devices, robust noise calibration, and credible differentiated invention all needed. None cleared.
