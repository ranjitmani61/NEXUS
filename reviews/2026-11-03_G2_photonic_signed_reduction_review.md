# Engineering gate review — cancellation-limited optical signed reduction

**Date:** 2026-11-03  
**Applied ACTS:** A1/A2/A6/A7/A8/A10/A11 (analytical review lenses only)  
**Status:** G1 novelty FAIL for generic photonic reduction; G2 conditional mathematical validation; G5 real AI evidence BLOCKED; G6+ NOT AUTHORIZED.

## Decision first

Do **not** pursue this generic optical signed-reduction scheme as a novel processor on the present evidence. Existing demonstrations cover optical AllReduce; our idealized photon-count model shows worsening relative quality under cancellation, and unmodeled calibration and metadata costs could erase any benefit.

## Evidence and controlled conditions

Pre-registered in `specs/G2_photonic_signed_reduction_contract.md`. Two independent detector channels with Poisson photon arrivals, 1e8 photons for L1=1, 1% relative error, risk 0.1%, 3 independent seeds, 3,000 trials/condition/seed (45,000 normal trials). Exact signed 2-term inputs chosen to produce condition numbers 1,2,10,100,1000. This is **not** a realistic deployed photonic apparatus or representative workload mix.

| `kappa` | Relative RMS error, 3-seed range | Analog-guard accept fraction | Raw measurements exceeding 1% (seed 20261103) |
|---:|---:|---:|---:|
| 1 | 0.0098–0.0100% | 100% | 0% |
| 2 | ~0.020% | 100% | 0% |
| 10 | 0.098–0.101% | 0% | 0% |
| 100 | 0.994–1.010% | 0% | 31.9% |
| 1000 | 10.02–10.20% | 0% | 91.7% |

**0 observed false certifications** in the 45,000 ideal Poisson trials. This is *not* a deterministic safety proof; the confidence guard is mathematically backed only under the stated stochastic/precision assumptions. Deliberate gain miscalibration (`+/-2%`) yielded **3,000/3,000 false certifications per seed** when the gain error was omitted from the guard; with the 2% gain error accounted for, the tested candidate rejected all such outputs (9,000/9,000 correctly rejected over three seeds).

At 99.9% confidence using conservative Chebyshev for **fixed known inputs**, the sufficient expected detected photon budgets are 1e7,4e7,1e9,1e11,1e13 at `kappa=1,2,10,100,1000` respectively. At 1550nm, the last represents ~1.28 microjoules of optical energy reaching detectors **per scalar**, excluding all transmitter/network/receiver losses and electronics. This is not a measured hardware power figure or a universal optical physics lower bound.

The contrived 20%-per-condition workload has ~40% acceptance. Under *invented merely for sensitivity* 0.15 analog + 1 digital fallback energy units, the algebraic cost becomes 0.75 versus digital-only 1 (ratio 1.33x). To reach 2x, this hypothetical energy model requires ≥65% acceptance. **Neither ratio is measured and neither proves system feasibility.**

## Major disqualifying assumptions

1. Pre-normalization/guard require `L1=sum(abs(x_i))` known at the receiver. Acquiring that from distributed inputs may require an extra collective; uncharged here. This could be a complete bottleneck.
2. Detector gain, source drift, saturation, laser power, ADC/DAC, actual network topology, encoding, and coherence constraints are absent. A 2% intentionally omitted gain bias breaks the claimed guard even in this toy model.
3. Strong existing optical-reduction prior art (OFC 2026 and Nature Communications 2026). The method does not define a new communication primitive.
4. No silicon/FPGA/optical prototype, no real precision-/task-specific error budget, no digital NCCL measurement, no area/energy/latency comparison, no independent expert review.

## Verification/provenance

- `tests/g2_photonic_signed_reduction.py`: implementation, random seeds, output JSON.
- `tests/test_g2_photonic_signed_reduction.py`: unit and property tests, 19 passed.
- `tests/test_g2_progressive_bitplane.py`: prior-regression suite, 9 passed.
- `results/g2_photonic_signed_reduction_pytest.log`: `28 passed in 0.13s`.
- `results/g2_photonic_signed_reduction.json` + two independent-seed files: numbers above, 54,000 simulated measurements in total including gain-mismatch controls.
- `research/G1_photonic_reduction_prior_art.md`: primary-source list, dated.

**Gate status:** G2 statistical model is internally coherent *under assumptions*, not a processor feasibility pass. G1 generic-method novelty FAIL. G3 hardware arithmetic/finite-precision NOT PROVEN. G4 end-to-end NOT RUN. G5 BLOCKED. G6–G12 NOT AUTHORIZED. **No breakthrough demonstrated.**

**Next discriminating study:** Demand a new device/communication mechanism that eliminates the extra `L1` collective or otherwise retains a provable error budget without such metadata; compare against 2026 optical AllReduce and modern digital collectives with *total* physical costs. Do not fabricate claims of a novel photonic device, or begin RTL/ASIC work.