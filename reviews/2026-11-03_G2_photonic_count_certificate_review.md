# Engineering gate review — G2 photonic readout without an L1 collective

**Version:** 1.0 · **Date:** 2026-11-03  
**Research lenses:** A1 architecture, A2 applied mathematics, A6 communication, A7 optical device assumptions, A8 verification, A10 systems/performance, A11 prior art. **No independent hardware/optics specialist review.**  
**Primary evidence:** `results/g2_photonic_count_certificate.json`; `specs/G2_self_certifying_photonic_readout.md`; source and test log.  
**G1 novelty:** FAIL for generic optical summation + known statistical confidence/calibration techniques. **G2 model:** CONDITIONAL PASS for mathematical confidence construction, not an architectural breakthrough. **G3 finite precision/hardware:** NOT VERIFIED. **G4 end-to-end:** NOT VERIFIED. **G5 real LLM:** BLOCKED. **G6+ physical chip:** NOT AUTHORIZED.

## Decision

**Do not begin hardware design.** Exact-count Poisson confidence intervals allow an *idealized receiver* to bound its result without separately receiving the true sum of absolute signed inputs (`L1`). This solves one missing-metadata assumption in the previous mathematical toy model, but **not** the problems of source-to-photon scale calibration, channel gain drift, pilot overhead, missing full-system costs, or novelty.

## Model and statistical guarantee

For nonnegative positive and negative components `P,M`, independent count channels are `C+ ~ Poisson(alpha * g+ * P)` and `C- ~ Poisson(alpha * g- * M)`. Assume the *absolute* source-to-count scale `alpha` is accurately known. The receiver sees counts but not `P,M,L1`.

For one Poisson observation, a Garwood interval for its mean uses Chi-square quantiles and has >=`1-beta` coverage. With two count channels, take `beta=delta/2` per channel; with two pilot-gain channels, `beta=delta/4` per channel. Apply nonnegative interval arithmetic to infer intervals on `P` and `M`, then `s=P-M`. The admitted estimate is forced to have `|hat_s-s|/|s|<=0.01` for **every** `s` in the resulting interval, excluding zero.

If the assumed model holds, the **unconditional** probability per fixed input that the mechanism both certifies and violates the 1% relative-error target is <=`delta=0.001`. It is **not** a deterministic guarantee, and it is **not** a guaranteed conditional error probability given certification. If gains drift after pilot readings, the Poisson-interval calculation is no longer valid for the data channel gain.

## Predeclared (v1.0), amended negative-control (v1.1) scenarios

`alpha=100,000,000`, `L1=1`, `kappa=[1,2,10,100,1000]` with equal weighting **only for these artificial tests**, eps 1%, delta 0.1%, 3 independent seeds, 1000 trials per kappa per scenario. 90,000 simulated cases total. Actual hardware photon budget, laser energy and runtime are **not measured**.

| Scenario | Accepted | False certifications (out of 15,000) | Nominal pilot photons per reduction | Observation |
|---|---:|---:|---:|---|
| Perfectly known gains = 1 | 60% | 0 | 0 | ideal oracle; accepts kappa 1/2/10 |
| 2% opposite channel drift **ignored** | 60% | **9000** | 0 | unsafe: optimistic and false |
| Actual opposite ±2% drift **covered** | 0% | 0 | 0 | safely rejects all tested cases |
| Pilot 10^6 photons/channel | 40% | 0 | 2,000,000 | accepts kappa 1/2 only; +2% photons vs baseline |
| Pilot 10^8 photons/channel | 60% | 0 | 200,000,000 | accepts kappa 1/2/10; +200% photons vs 10^8 baseline |
| Pilot 10^8/channel but data gain shifts +2% afterward | 60% | **9000** | 200,000,000 | **unsafe calibration freshness negative control** |

No *false* admissions were observed in the **valid-model** scenarios; the mathematical coverage relies on Garwood/Bonferroni and stated independence, not on a claim that a finite sample of tests proves safety. Poor coverage in the deliberately invalid blind/stale models is an expected, important failure. Extreme cancellation cases (kappa 100/1000) are not admitted in any tested configuration.

There is no free lunch from avoiding the separate `L1` collective: a near-exact photon-count-to-input scale and ongoing detector gains are assumed. A 10^8-photon **pilot per channel** adds 200 million expected detected photons before laser/link/electronic overhead. If calibration is reused across multiple reductions, overhead per reduction may fall, but then the allowable drift interval and accumulated risk must be modeled; this experiment intentionally does not assume cost-free reuse.

## Prior art and novelty assessment

- Garwood (1936) confidence intervals for Poisson means, reproduced in https://pmc.ncbi.nlm.nih.gov/articles/PMC3964675/ — statistical certificate itself is not novel.
- OFC 2026, integrated-photonic analog AllReduce https://opg.optica.org/abstract.cfm?uri=OFC-2026-Th3H.3 — physical reduction is established.
- Nature Communications 2026, calibrated balanced optical tensor computation https://doi.org/10.1038/s41467-026-71599-2 — balanced readout and calibration have direct prior art.

This is not an exhaustive patent search. No architectural novelty is supported.

## Reproducibility

```bash
cd /mnt/data
python tests/g2_photonic_count_certificate.py --output results/g2_photonic_count_certificate.json
PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_count_certificate.py tests/test_g2_photonic_signed_reduction.py tests/test_g2_progressive_bitplane.py
```

**Tests:** 42/42 passed, including inherited suites; three seeds each 30,000 trials. **Falsification controls:** 9,000/15,000 unsafe certifications in each of two wrong-calibration scenarios. Full per-kappa/per-seed results preserved in JSON. The simulator is not a device model; Poisson counts, gain stationarity, absolute calibration, and confidence quantiles may not hold on physical hardware. Finite-precision hardware implementation and digital NCCL comparisons not run.

## Gate decision and next authorized work

- G1: **FAIL** for claimed novelty of generic scheme.
- G2: **CONDITIONAL PASS** for idealized probabilistic bound, with L1 collective no longer needed in this altered encoding.
- G3/G4: **BLOCKED / NOT VERIFIED** (photonic devices, source calibration, drift, pilot reuse, throughput, energy, latency, best digital baseline).
- G5: **BLOCKED** (real transformer workloads).
- G6+: **NOT AUTHORIZED**.

**Next smallest test:** formalize bounded *time-dependent* calibration error and amortized pilot energy under explicit detector drift assumptions, and compare full costs with digital collective; alternatively reject the photonic branch if no distinct implementation mechanism is identified. Do not claim a novel chip.
