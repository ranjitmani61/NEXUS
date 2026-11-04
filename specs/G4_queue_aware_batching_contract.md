# G4 contract — Queueing and independent KV-cost sensitivity for model-weight batching

**Date:** 2026-11-03 | **Status:** diagnostic research only, not pretrained or new hardware | **Review lenses:** A1 architecture, A2 applied math, A3 ML systems, A4 compiler/runtime, A6 memory, A8 software verification, A10 performance, A11 prior art. Human specialist review not performed.

## Question, assumptions, null
Does a fused batched matrix-vector projection decrease *response time* as well as aggregate throughput? Can independent per-request KV costs overturn apparent gains? The null is that queueing delay, inefficient small batches and KV costs negate any benefit for some conditions.

- Each simulated request represents **one independent FP32 CPU projection** `y = W x` on an N=4096 square matrix; it is **NOT** a full transformer request or a token-generation session.
- Service-time curves use two local CPU runs (seeds 81 and 82), each with 32 batch sizes, 5 timing repetitions, 2 warmups, one OpenBLAS thread; GEMV at b=1, GEMM at b>=2. 32 input columns are independent. Preallocated outputs; kernels alone, not memory allocation, networking, prefill, attention or CUDA.
- Queue model: one nonpreemptive server, 1,200 arrivals in FCFS order from either fixed-seed exponential interarrivals (Poisson model) or idealized bursts of 16 simultaneous jobs. Arrival rates: 150, 300, 450, 650 requests/s, three arrival seeds 101–103. No background work or serving overhead.
- Five policies: individual serial; greedy max8; greedy max32; max32 with 2-ms waiting window; max32 with 8-ms window. Windows end early if enough queued requests fill the batch. At each dispatch, the batch service time is the actually measured batch-size median, not an interpolation. Arrival waiting, execution time and queue spillover are included.
- `throughput_s` = requests processed divided by time from first arrival to final completion, **not** offered request rate nor achievable GPU LLM throughput. p50/p95/p99 measured from arrival to completion within each trace.
- Optional *hypothetical* KV penalty: add `h*b` ms to batch service for b requests, using h in {0, 0.25, 1, 2, 4} ms per request. This is a sensitivity parameter, **not an observed cache read**. Its arithmetic is consistent across all policies but does not represent real attention computation or bandwidth overlap.
- Both service-time timing curves, raw 5-sample timing arrays, all simulated policy cases, and sensitivity cases are preserved. Service calibration variability is part of the report.
- Independent GEMV numerical cross-check in corrected second timing run: all b=32 outputs checked against 32 column-by-column GEMV calls with rtol=6e-4, atol=2e-3; maximal absolute difference recorded. Earlier run used a weaker same-GEMM-route check and is retained, not rewritten.

## Interpretations explicitly forbidden
- Claiming simulated p95 means a real serving measurement.
- Claiming that measured CPU projection time corresponds to H200/GPU/TPU LLM throughput.
- Concluding that the 2× target, G5, or novelty is met from conventional batching.
- Treating high-load overloaded queues as steady-state stable systems.
- Treating an assumed KV-cost penalty as real KV data traffic or power.
- Treating comparisons with simple single-thread OpenBLAS/FCFS algorithms as competitive with production vLLM/Sarathi/FlashInfer scheduling.

## Reproducibility
```bash
cd /mnt/data
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_g4_queue_aware_batching.py' -q
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_g4_kv_penalty_sensitivity.py' -q
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_g2_weight_reuse_batch.py' -q
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_g2_memory_traffic_headroom.py' -q
OPENBLAS_NUM_THREADS=1 python tests/g4_queue_aware_batching.py --output results/g4_queue_aware_batching_verified.json --dimension 4096 --reps 5
PYTHONPATH=tests python tests/g4_kv_penalty_sensitivity.py
```
These produce timing-dependent results that can differ across machines and even across successive local runs.

## Existing scheduling prior art
- vLLM continuous batching and scheduling: https://vllm.ai/blog/2025-09-05-anatomy-of-vllm
- Sarathi-Serve OSDI 2024 original paper: https://www.usenix.org/conference/osdi24/presentation/agrawal
- Queueing-aware serving prior art (October 2026): https://arxiv.org/abs/2610.14686

No architecture-level novelty is proposed or established here.