# Gate review — Published optical costs versus photonic readout assumptions

**Date:** 2026-11-04 · **Version:** 1.0 · **Protocol:** `specs/G2_photonic_static_power_gate_contract.md` · **Project rules:** `AGENTS.md` read in full; unchanged.  
**Evidence type:** cited measurements/estimates from other published devices plus our independently executed arithmetic and software tests. No actual photonic prototype and no independent hardware review.

## Decision first

**HOLD the photonic branch.** The prior estimate omitted real-device static/control power and provided no measured end-to-end digital baseline or transferability evidence. Related optical collective mechanisms are already demonstrated. **G1 novelty not demonstrated; G2 physical feasibility unproven; G5 BLOCKED; G6 NOT AUTHORIZED.** Do not treat the results below as device performance, lower bounds for all optics, or an efficiency win.

## External evidence and provenance

- 2026 16-tap optical processor: authors **estimate 320 mW total thermo-optic heater power** (16 x ~20 mW); not full-device power or a photonic AllReduce result. [Nature / Light: Science & Applications](https://www.nature.com/articles/s41377-025-02093-5)
- 2025 silicon photonic engine: ~300 mW electrical laser and **worst-case** 812 mW tuning, total ~1.112 W in that configuration. Different task/device; not comparable as measured optical reductions. [Nature Communications](https://www.nature.com/articles/s41467-025-60100-0)
- 2026 Parallel Photonic Integration: physical projector/camera prototype with four senders and GPT-2 gradient reductions; large-cluster energy/latency claims come primarily from **modeling**. Its predicted optical *fabric* energy is higher than InfiniBand in some comparisons; the modeled system benefit arises from less GPU idle time. This is especially relevant prior art and warns against judging device-only energy. [Nature Communications](https://www.nature.com/articles/s41467-026-76764-1)

## Calculation results

From previous **simulated** case (`K=256, rho=1e-4 per readout, kappa=1`): admission `a=0.35547`, idealized dynamic photon/electronics cost `E_dyn=10.66700975 nJ/readout`.

If we *hypothetically* charged the published power references entirely to such a reduction stream:

| Illustrative power anchor (different devices) | Static energy at 1M reductions/s | Static energy at 100M reductions/s | Minimum modeled sustained reductions/s to beat a **hypothetical** 100 nJ digital baseline |
|---|---:|---:|---:|
| 320 mW heater-only estimate | 320 nJ | 3.2 nJ | >12.86 million/s |
| 1.112 W laser+phase **worst-case** | 1,112 nJ | 11.12 nJ | >44.69 million/s |

These are conditional throughput requirements, **not tested device capabilities**. The *same scenario* cannot beat a hypothetical **30 nJ digital baseline** at any throughput, even with zero static power: `a*30 = 10.6641 nJ`, slightly less than `E_dyn=10.6670 nJ`. This is a razor-thin hypothetical difference and cannot be interpreted as a robust device conclusion.

The model is intentionally incomplete: all physical interface, drift-control, A/D+D/A, thermal, packaging, idle power, fallback latency and real task-quality measurements are missing. The authors' published power references were for **different** architectures and workloads; we have not measured a single matched pair of systems.

## Reliability budget problem

The prior `delta_cycle=0.001` per-pilot bound reaches a vacuous union-bound estimate of 1 at 1000 cycles. If a deployment requires <=0.001 **total** budget over 1 million pilot cycles, `delta_cycle` must be <=1e-9 and each data-channel tail noncoverage <=9.77e-13 for K=256. Such conservative intervals may drastically reduce admissions; we have **not** tested the resulting runtime performance or independent detector noise model. A gain jump outside assumptions invalidates even that bound, as shown in the previous archived test.

## Verification actually performed

`python tests/g2_photonic_static_power_audit.py --output results/g2_photonic_static_power_audit.json` executed.  
`PYTHONPATH=tests python -m pytest -q tests/test_g2_photonic_static_power_audit.py` executed.  
**19/19 new unit tests passed**. Includes physical units, independent algebraic cross-check, loss cases at low matched digital energy, boundary and invalid input cases, zero-admission scenario, and global union-bound math. Machine-readable rows and actual console logs included. This only verifies code behavior under the model, not physical plausibility. Previous 52 tests were not rerun here and must not be counted as new tests.

## Research decision / next gate

1. No photonic ASIC, FPGA, or purchased photonic components justified by this work.
2. Need **matching** per-reduction digital energy/latency and a specified realistic AllReduce workload.
3. Need photonic-component measurement of laser, detectors, ADC/DAC, phase heating, calibration drift over seconds, SNR/bit accuracy, peak/duty-cycle power, and full fallback path.
4. Need literature review to isolate a potentially *novel* mechanism; currently photonic reduction and stabilization have strong prior art.
5. With no device-specific evidence, **pause** this candidate; revisit only after an actual measurable benefit and a non-prior-art contribution are identified.

**Result:** No breakthrough demonstrated. No hardware authorization.
