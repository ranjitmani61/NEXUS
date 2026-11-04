# Engineering gate review — Lossless codec pivot, v1.0

**Date:** 2026-11-03. **Status:** Exploratory G2 test completed; **G2 processor innovation FAIL / HOLD**, G5 pretrained model evidence BLOCKED, G6 hardware NOT AUTHORIZED. **Governing rules:** `AGENTS.md` v1.0 read. **Review lenses:** A1, A2, A4, A6, A8, A10, A11. **No external engineering review.**

## Decision first

We terminated the Quest-like query-aware page-screening concept as a candidate *novel* processor mechanism. A separate, lossless data-representation test shows reduced stored bytes only for certain synthetic distributions; compiled CPU decode-plus-scan remains slower than reading raw pages. Its underlying ideas are known (see AdaCode 2026). There is no evidence to authorize RTL/ASIC or any claim of novelty.

## Direct evidence — actual compiled experiments, not LLM inference

`tests/g2_lossless_codec_falsification.cpp` was compiled with g++ 14.2.0, `-O3 -ffp-contract=off`, linked against libzstd, and run twice on this Linux CPU runtime. For each regime, 128 pages × 32 tokens × 64 16-bit channels. Page descriptor accounted for identically in raw and encoded formats. All original 16-bit words recovered exactly. 12,282 sampled roundtrip/truncation checks succeeded in each optimized build run. AddressSanitizer and UndefinedBehaviorSanitizer execution returned exit status 0 with no logged stderr. Assertions cover short/corrupt headers and inconsistent payload size; **no general immunity to silent bit flips**.

| Synthetic regime | XOR packed bytes / raw | zstd-1 bytes / raw | XOR decode+scan / raw scan (run1, run2) | Key point |
|---|---:|---:|---:|---|
| All zeros | 0.04873 | 0.00658 | 1.99×, 2.73× | zstd much smaller. |
| Narrow positive stationary | 0.53211 | 0.18836 | 3.43×, 4.19× | zstd smaller; both decompress slower. |
| Shared-channel correlated | 0.47257 | 0.77426 | 5.76×, 4.99× | XOR smaller than naive zstd-1 in this artificial regime. |
| Independent BF16 Gaussian | 1.00000 | 0.79320 | 3.00×, 2.01× | XOR falls back to raw; zstd smaller but much slower. |
| Adversarial random 16-bit | 1.00000 | 1.00000 | 2.85×, 2.57× | Both fallback to raw; CPU XOR fallback copies unnecessarily. |

Time ratios compare one sequential checksum of decoded pages versus equivalent raw pages; **not attention kernels**, not GPU/HBM, and not full-model token time. The second decoder uses preallocated scratch and no per-channel heap allocations, reducing a known distortion in the first draft. The zstd comparison uses level 1, original interleaved source, preallocated decode output, and no optimized KV layout; this is a **convenience baseline, not strongest prior art**. CPUs, caches, allocation, compiler scheduling, and scan order confound absolute ratios. Two runs do not establish a robust timing distribution. For raw fallback, a zero-copy path could eliminate most overhead; this reference intentionally always decodes into scratch and should not be read as a fundamental cost lower bound. There is no claimed hardware cycle model or energy model.

## Derived upper ceiling on byte reduction effect

A separate idealized 8B-parameter BF16, 32-layer, 8-KV-head, 128-dim GQA example uses 16 GB weight reads per step and `131,072 * context * batch` KV bytes per step. Assuming the favorable synthetic correlated compression `r=0.47257` **and free decompression**, the ratio of total uncompressed to compressed bytes is:

- Batch 1, context 4,096: **1.017×**.
- Batch 1, context 16,384: **1.067×**.
- Batch 1, context 131,072: **1.376×**.
- Batch 16, context 131,072: **1.994×**.

These are **hypothetical ratios of byte volumes**, not upper bounds on all possible performance speedups or representative model benchmarks. The assumptions intentionally omit weight caching, mixed precision, compute overlap, attention algorithm choices, actual KV distributions, and all decoder cost. A genuinely new memory accelerator would need real measurement with realistic models and strongest modern baselines. Neither a novel architecture nor a model quality result follows from this calculation.

## Feasibility and novelty gate

- **G0:** Existing charter maintained; alternative is exploratory, not a changed project success criterion.
- **G1:** Lossless BF16 KV compression directly overlaps AdaCode (EMNLP 2026). Novelty not established. No patent search performed.
- **G2 (codec correctness):** PASS **under tested page format and inputs**, not as a computing breakthrough.
- **G2 (new architecture):** FAIL TO ESTABLISH: tested design is an existing kind of compressor with no newly distinguished physical execution mechanism.
- **G3:** Bit-exact page recovery for tested formats; error-certificate issues from page-screening are not transferred or resolved by codec tests. No concurrency/integrity proof.
- **G4:** Measured CPU checksum benchmark is a narrow exploratory result; no competitive accelerator baseline.
- **G5:** BLOCKED — pretrained KV data unavailable in the local workspace; synthetic results do not substitute.
- **G6+:** NOT AUTHORIZED. **G12 breakthrough: NOT DEMONSTRATED.**

## Critical remaining unknowns / stop condition

Real pretrained BF16 K and V distribution, multi-model/layer/input compression variability, RoPE-specific effects, rate of token appends, partial-page updates, GPU decoding throughput, HBM/controller data transfers, ECC/integrity and power, memory capacity, zstd-transposed / AdaCode baselines, concurrency-safe publication, and cost of integrated silicon. Without those, stop the current codec as an invention thesis. If a distinct physical/algorithmic mechanism is proposed, restart G1/G2 with a new, falsifiable claim rather than relabeling compression or screening.

## Reproduce / raw evidence

`specs/G2_lossless_kv_codec_pivot.md` has exact commands and assumptions. Raw `results/g2_lossless_codec_{run1,run2,sanitizer}.csv`, empty sanitizer stderr, `results/g2_lossless_codec_cost_model.json`, sources `tests/g2_lossless_codec_falsification.cpp`, `scripts/g2_lossless_codec_cost_model.py`. No network model weights and no paid cloud jobs were used.
