# G2 — Near-memory computing: falsifiable throughput threshold contract

**Date:** 2026-11-03  **Version:** 1.0  **Project:** Processor 01  **Status:** Analytical diagnostic, NOT a novel architecture or proposed RTL

## Objective and engineering lenses
- **Objective:** Determine which combinations of internal memory bandwidth, near-memory arithmetic throughput, launch latency, and activation fanout would have an *illustrative* advantage over a host/GPU decode roofline.
- **ACTS:** A1 computer architecture, A2 mathematics, A6 memory/interconnect, A8 verification, A10 performance, A11 prior art. **No external A5/A7/A9/A12 electrical, physical, thermal or manufacturing specialist review** was conducted.
- **Falsifiable claim:** Near-memory computing is not beneficial merely because weight transfers across an interface are removed; it still needs adequate internal bandwidth, compute, capacity, and low coordination overhead.
- **Prior art:** See `research/G1_near_memory_prior_art.md`. Memory-side computing itself is not a new mechanism.

## Inputs chosen before the sensitivity run
For a deliberately simplified 8-billion-weight BF16 (2 bytes/weight) decoder, 32 layers, 8 KV heads, head dimension 128, 4096-wide hidden vectors, B=1, T=8192:
- 16,000,000,000 weight bytes per token (one streaming read assumed; no effective weight retention/caching).
- KV bytes read = `B × T × L × 2(K+V) × H_kv × D_head × 2 bytes` = **1,073,741,824 bytes** at B=1 and T=8192.
- Matmul work approximated as `2 × 8e9 × B` FLOPs; other work (attention, softmax, normalization, sampling, attention communication) is not explicitly modeled. The same missing work is **not** guaranteed to cancel in a real chip.
- Host/GPU reference: bandwidth 3.35 TB/s, peak BF16 compute 989 TFLOP/s. **Illustrative hardware peak values, not actual measured sustained throughput**. Compare with contemporary actual GPUs and deployed inference engines before design authorization.
- Near-memory test point: internal bandwidth 8 TB/s, near-memory compute 10 TFLOP/s, per-layer coordination 5 microseconds, 64 bank fanouts; all **hypothetical**.
- Input communication: `L×B×hidden×activation_bytes×(fanout_banks+1)`; this is a *partial* host/near-memory communication proxy, not a full interconnect model.

## Equations (units shown in the runnable source)
Let `W` be weight bytes, `K` KV bytes, `F` matmul FLOPs, `B_h` host BW, `B_i` internal near-memory BW, `P_h` host peak compute, `P_i` near-memory peak compute, `L` layers, `τ` launch seconds/layer, and `A` modeled activation-transfer bytes.

- Optimistic *host roofline lower bound* (not measured runtime):
  `T_host = max((W + K)/B_h, F/P_h)`.
- Hypothetical **sequential** near-memory time model (not a proven physical lower/upper bound):
  `T_near = max(W/B_i, F/P_i) + K/B_h + A/B_h + L×τ`.
- Ratio: `R = T_host/T_near`; **R>1 is not an empirical speedup**.
- Per-layer break-even launch threshold:
  `τ* = [T_host − max(W/B_i,F/P_i) − K/B_h − A/B_h]/L`.
  If `τ* <= 0`, even zero modeled launch overhead cannot beat this specific host reference.

## Pre-specified sensitivity axes
- Internal bandwidth: 1, 3.35, 8, 16 TB/s.
- Near-memory compute: 0.5, 1, 5, 10, 40 TFLOP/s.
- Coordination overhead: 0, 5, 25, 100 microseconds per layer.
- Total grid **4×5×4 = 80** conditions, no post hoc selection.
- Additional negative controls: compute 1 TFLOP/s; 100us coordination; same memory BW as host; B=32; T=131072; activation fanout 1024; internal BW only 1 TB/s.

## Failure mechanisms / omitted costs
1. 16 GB of BF16 weights must be housed near compute; memory capacity, bank conflicts, refresh, retention, ECC and die area are not simulated.
2. Internal BW and near-memory FLOP/s are invented sensitivity axes, **not manufacturer specs**. They cannot be converted to claims about any existing PIM device.
3. No thermal/energy/PPA modeling, yield, compilation, synchronization, numerical precision, ADC/DAC (if analog), or manufacturability.
4. Attention/KV processing, softmax and full-model graph scheduling are simplified. There is no measured end-to-end token latency or output accuracy.
5. Host model assumes all weights stream once per token; batching, speculative decoding, quantization, caches, multi-GPU sharding and production kernels change the comparison.
6. Host is a theoretical roofline *lower bound*. Near-memory is an optimistic modeled schedule with incomplete overhead. Comparing these two is a feasibility screen, **not a bound on a real-world speedup**.
7. A near-memory design is established prior art and needs a **distinct, newly specified mechanism** plus fair head-to-head comparisons before the project's invention gate can be cleared.

## Reproduction
```bash
python tests/g2_near_memory_threshold.py
python -m pytest -q tests/test_g2_near_memory_threshold.py
```

Output: `results/g2_near_memory_threshold.json`. Script uses Python standard library. Pytest needed for tests. No hardware is exercised, no pretrained-model data is accessed.

**Decision criteria:** G2 arithmetic may pass with no unit failures and checked negative controls. G1 novelty cannot pass on near-memory computing alone. G5 and G6 remain blocked/not authorized without representative workloads and a credible distinctive datapath.
