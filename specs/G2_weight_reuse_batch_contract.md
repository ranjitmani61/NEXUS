# G2/G4 — Weight reuse through batching: analytical and measured CPU contract

**Version:** 1.0 | **Date:** 2026-11-03 | **Lead review lenses:** A1 architecture, A2 mathematics, A3 AI systems, A4 runtime, A6 memory, A8 verification, A10 performance, A11 prior art. No external human review.

## Question and falsifiable hypothesis
At batch size one, weight streaming dominates an illustrative 8B BF16 8K-context decode byte budget. Is weight traffic amortization through sharing one weight stream across B independent requests a meaningful opportunity, after accounting for B independent KV streams and per-request latency?

- **Null:** Batch placement does not reduce bytes because weights are streamed separately per sequence (`reuse=0`), or the compute/launch and queueing costs dominate.
- **Existing mechanism:** Batched matrix multiplication/continuous batching. **No novelty claim**. vLLM and literature already document weight reuse vs latency/throughput trade-off.
- **Success for this study only:** Reproduce conditional byte formulas and identify a batch range in which a real single-thread CPU BLAS GEMM beats repeated BLAS GEMV at matched FP32 outputs. This does NOT clear G5 or justify a processor.

## Analytic model
Hypothetical parameters P=8,000,000,000; weights BF16 (2 bytes); L=32, KV heads=8, dim=128, KV BF16 (2 bytes), context T. Published H200 interface peak 4.8 TB/s is used solely to calculate unattainable optimistic byte-only time floors, **not achieved bandwidth**.

Let W = P*2 = 16,000,000,000 bytes, K(T) = 2*L*8*128*T*2 = 131,072*T bytes **per request**. Batch B shares weights, not independent KV states. Model a purely hypothetical share-efficiency parameter r in [0,1]:

- Independent sequential byte sum: I = B(W+K)
- Batched weight bytes: W*(1+(B-1)*(1-r))
- Batched KV bytes: BK
- Batched total M = W*(1+(B-1)*(1-r))+BK
- Relative byte reduction: I/M
- Best-case per-token bytes at r=1: W/B + K
- Optimistic batched bandwidth time floor: M / BW
- Minimum matrix/weight arithmetic estimated as 2PB scalar FLOPs, excluding attention and non-matmul operators

**r is an interpolation parameter**, not a measured memory-cache hit rate. No compute, data-movement concurrency, paging, queueing, fusion, activations, runtime scheduling, KV writes, or hardware power is modeled. The BW time floor cannot be read as expected latency. B may also decrease throughput if effective bandwidth or compute utilization changes.

## Controlled CPU microbenchmark (actual measurement, limited scope)

- Matrix is FP32 N×N, N = 2,048 and 4,096; inputs N×B for B=1,2,4,8,16,32; seeds 7 and 19.
- Compare (A) B BLAS GEMV calls `np.dot(W, X[:,j], out=Y[:,j])` versus (B) one BLAS GEMM `np.matmul(W,X,out=Y)` using **the same float32 weights and inputs**.
- Both use Fortran-contiguous arrays, preallocated outputs, same OpenBLAS build pinned to **one BLAS thread**, two warmups, seven timed repetitions/arm with randomized paired order.
- Correctness: `np.testing.assert_allclose(..., rtol=6e-4, atol=2e-3)`; record maximum absolute error.
- Two separate script runs; report medians over the four size+seed combinations in each run, not a confidence interval.
- CPU matrix microbenchmark, not a transformer, GPU, silicon, or production comparison. Large CPU cache and GEMM microkernel selection change results. Actual per-request latency with queueing would be different.

### Reproduction commands

```bash
cd /mnt/data
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_g2_weight_reuse_batch.py' -v
python tests/g2_weight_reuse_batch.py --output results/g2_weight_reuse_batch.json
python tests/g2_weight_reuse_batch.py --output results/g2_weight_reuse_batch_repeat.json
```

### Key predeclared audit requirements

1. At B=1, traffic reduction is 1. At r=0, traffic reduction remains 1 for every B.
2. KV bytes scale linearly with B; batched weights never fall below W.
3. The batch-32, T=8,192 traffic ratio is a **conditional byte-volume reduction**, not a 10.85× hardware speedup.
4. Compare whole batch completion with one individual request to expose latency-throughput trade-off.
5. Show unfavorable B=2 and random-vs-batched sequences; include **all** tested batch sizes.
6. Changing benchmark precision or compiler/backend invalidates direct numerical comparability.
7. Do not claim a 2× end-to-end model speedup or invent chip architectures from GEMM results.

## Existing prior art, consulted for scope and novelty

- vLLM architecture and batching discussion: https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- Sarathi-Serve / OSDI 2024 proceedings on batching and operational intensity: https://www.usenix.org/system/files/osdi24_full_proceedings.pdf
- FlashInfer (2025) attention serving framework: https://arxiv.org/abs/2501.01005
- H200 published peak interface specification (not measured): https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html

## Gate disposition

**G2**: Conditional byte arithmetic reproducible. **G4**: Limited CPU microbenchmark reproducible. **G1**: Not novel. **G5**: Blocked (no real pretrained traces / deployed GPU benchmark). **G6+**: Not authorized.