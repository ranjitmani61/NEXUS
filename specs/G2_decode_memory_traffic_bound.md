# G2 — Conditional decode memory-traffic bound

**Version:** 1.0 | **Date:** 2026-11-03 | **Owner/review lenses:** A1 computer architecture, A2 mathematical analysis, A3 ML workload assumptions, A6 memory systems, A8 verification, A10 performance, A11 source check. No independent hardware review.

## Goal and gate

Assess whether a processor innovation targeting **only KV-cache reads** could credibly reach the project's provisional ≥2× end-to-end efficiency ambition on short-context, batch-one autoregressive decoding. This is a **conditional arithmetic bound**, not GPU evidence, a proof of novel architecture, or a universal no-go theorem.

## Explicitly hypothetical decoder

- Model: 8,000,000,000 parameters, 2 bytes per parameter (nominal BF16 weights; 16.0 decimal GB weights)
- Decoder: 32 layers; 8 KV heads (GQA); per-head dimension 128; BF16 KV elements (2 bytes)
- Batch: 1; context lengths 0, 4096, 8192, 32768, 65536, 131072 tokens
- GPU example: NVIDIA H200 SXM peak published HBM bandwidth 4.8 TB/s, **not achieved/effective bandwidth**; https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
- Model is *not* a measured/reconstructed checkpoint; its 128K variant is hypothetical and not evidence a particular model supports that length.

## Derivation (one time step)

Let `P` be parameter count, `w` weight bytes/param, `L` layers, `Hkv` KV heads, `D` head size, `T` context, `B` sequences and `v` KV bytes/element.

```
Weight bytes W = P * w
KV bytes     K = 2 * L * Hkv * D * T * B * v
Total traffic M = W + K
Conditional ideal KV-only gain = (W + K) / W = 1 + K/W
Required K for >=2x total-byte-volume gain = W
Minimum T for >=2x = ceil[ W / (2*L*Hkv*D*B*v) ]
Peak-bandwidth traffic-only floor t >= M / bandwidth
```

At T=8192: W=16,000,000,000 bytes, K=1,073,741,824 bytes; conditional gain `17.073741824/16 = 1.067108864×` in total **bytes**. Peak H200 bandwidth-only floor `(W+K)/4.8e12 = 3.557 ms` versus `(W)/4.8e12 = 3.333 ms`, with other overhead and bandwidth inefficiency omitted. The conditional 2× byte-volume threshold occurs at 122,071 tokens. A 50%-weight-byte reduction, assuming no additional traffic or overhead, gives only `(W+K)/(0.5W+K) = 1.882×` at T=8192; ~53.36% nominal weight-byte saving would be required for 2× total byte reduction if KV bytes stay unchanged.

## Material limitations / falsification checks

1. **NOT universal.** This 1.067× ratio is not a law limiting speedup of any KV innovation: effective bandwidth may differ by stream, operations may overlap, or an innovation may also remove attention computation / improve scheduling / change the architecture. It is an **Amdahl-style bound under the explicitly serial shared-bandwidth traffic-only model**.
2. In reality, model weights may be quantized, sharded, batched, or partially cached; KV is paged, compressed, and may be held in different tiers. Our assumptions may be false for any given deployed system.
3. KV reads can be more than/less than one full scan per decode depending on kernel/multi-query sharing/cache; this counts **a minimum ideal scan in a dense GQA model**, not bus-transaction measurements.
4. Weight-traffic amortization across multi-sequence batches changes ratios. E.g., B=16 at 8K makes KV a comparable byte-volume term; it can also move the execution into a compute bottleneck.
5. At long contexts, KV traffic grows linearly; a KV-focused mechanism may have a larger opportunity. KV-only improvements at T=128K are only a conditional byte opportunity, not proof of real-model quality, capacity, memory movement, or chip speed.
6. H200 stated bandwidth is a **vendor peak**. This report does **not** claim achieved bandwidth, observed latency, energy, or GPU speedups.
7. INT4 `0.5 bytes/param` excludes scales, indices, padding, dequantization and hardware support and is an optimistic mathematical variant.
8. No pretrained weights, external model trace, optimized dense runtime, or head-to-head benchmark was measured. G5 remains blocked.

## Results / decisions

See machine-readable `results/g2_memory_traffic_headroom.json`. Main BF16 batch-1 context results:

| Context tokens | KV decimal GB | Conditional ideal KV elimination gain |
|---:|---:|---:|
| 4,096 | 0.537 | 1.034× |
| 8,192 | 1.074 | 1.067× |
| 32,768 | 4.295 | 1.268× |
| 65,536 | 8.590 | 1.537× |
| 131,072 | 17.180 | 2.074× |

**Decision:** KV-only byte-reduction strategies are a poor candidate for meeting ≥2× **total byte-volume gain** at 4–8K context for this hypothetical BF16 B1 shape; reaching ≥2× is arithmetically impossible in this **specified** model of shared-bandwidth cost. This does **not** imply no practical 2× KV algorithm exists for other models, kernels or workloads. Reframe G0 to test weight-stream/reuse and runtime control overhead versus representative pretrained benchmark traces. Stop calling any of the prior page-screening, codec, or temporal-delta ideas a new processor design. **G2 narrow byte feasibility check: PASS; G2 novel architecture: NOT ESTABLISHED; G5: BLOCKED; G6: NOT AUTHORIZED.**

## Reproducibility

```bash
cd /mnt/data
python tests/g2_memory_traffic_headroom.py --output results/g2_memory_traffic_headroom.json
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_g2_memory_traffic_headroom.py' -v
```

Expected: 12 tests. The test count is not evidence that the model assumptions match physical GPU traffic.
