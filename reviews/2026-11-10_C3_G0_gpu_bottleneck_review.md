# C3-G0 review — Batch-1 decode bottleneck on a T4 GPU (cycle 3)

**Version/date:** 1.0 / 2026-11-10
**Charter:** `specs/C3_G0_gpu_decode_bottleneck_charter.md`, pre-registered with amendments A1 (before main data) and A2 (post-data defect fix, before run 2). The hashes are in `results/c3/preregistration.json`.
**Data:** `results/c3/main_run2/` (the decision run), `results/c3/main_run1/` (invalid graph mode; kept), `results/c3/microbench/` (exploratory). Index and reproduction commands: `results/c3/README.md`.
**Lenses applied:** A1, A3, A4, A6, A8, A10; A11 only to name the closest prior art (no new search).
**Missing:** independent review; energy; DRAM counters; speculative decoding (B2); serving-grade kernels.
**Owner:** this AI-assisted session. Not an independent sign-off.

## 1. Decision, evidence, unresolved risk, next gate

**Decision: C3-G0 = PASS, for target selection only.** Under the pre-registered rule, two cost classes are eligible on this workload. Eligibility is not novelty and not feasibility, and this review claims neither.

1. **Attention core is time-eligible in both families.**
   - At C* its measured removal saves 0.58–0.92 of the step for Qwen (4K–32K) and 0.47–0.76 for Granite (4K–16K).
   - It reaches ≥ 0.5 at ≥ 1 context in each family.
2. **MLP weights are byte-eligible in both families.**
   - Derived bytes give MLP shares of 0.604 → 0.454 for Qwen (4K → 32K) and 0.705 → 0.618 for Granite (4K → 16K).
   - The near-tie check, which tries every C* candidate within CI, gives the same conclusion.

**What the evidence says about these targets** [Measured, exploratory, §5]:
- **Attention's time share is a kernel-efficiency effect, not a byte effect.**
  - At C*, attention moves its K/V at **1.8–4.6% of the measured copy bandwidth** (4.3–11.1 GB/s against 241 GB/s).
  - Every stock PyTorch 2.11 SDPA variant tested on this T4 stays at 2–15 GB/s: masked or unmasked, repeated or native GQA, with the efficient or math backend. The best is at most about 1.25× better than C*'s path.
  - If attention ran at copy bandwidth, its share of a bandwidth-bound step would be ≤ 0.285 (Qwen 32K) and ≤ 0.165 (Granite 16K), so it would **not** be time-eligible.
  - The obvious way to capture this 2× is a better decode-attention kernel in software. G1 must test this against FlashAttention (arXiv:2205.14135), Flash-Decoding (Dao et al., 2023) and PagedAttention (arXiv:2309.06180). Whether any of these runs on sm_75 in this environment **was not tested**.
- **MLP weight streaming is the dominant byte stream at ≤ 16K.** It would be the dominant time class (projected share ≥ 0.5 in both families) on a bandwidth-efficient implementation.
  - It is the classic memory-bound batch-1 problem, and prior art is dense.
  - The weight-only baseline available here, bitsandbytes NF4, **fails** matched quality on both models (PPL ratio 1.09–1.10; top-1 0.80–0.88).
  - LLM.int8 passes on Granite (PPL 0.999, top-1 0.984) but is 3.3× slower than fp16 (eager only).

**Unresolved risks:**
- Single old GPU (T4, sm_75) and one software stack.
- No serving-grade kernels (vLLM, TensorRT-LLM, llama.cpp, FlashInfer), so C* itself runs 2.2–17× above the bandwidth floor.
- Two small models; contexts ≤ 32K.
- Quality measured at 4K only (4 books × 32 tokens).
- B2 not measured.

**Next gate:** G1 (prior art) for both targets, each under its own contract. My expectation, stated as a hypothesis: G1 will find that the attention target is captured by existing software kernels and that the MLP-weight target is the well-studied weight-traffic problem. **No processor mechanism is proposed or implied by this PASS.**

## 2. Preconditions (all from run 2)

- **P1 (provenance):** every model file and the four books have SHA-256 equal to `results/g5r/provenance_*.json`. Full hashes, 60 s.
- **P2 (decode consistency, static cache vs full forward, T = 4096, 3 steps, 4 books):**
  - Qwen fp32: KL ≤ 2.2e-11. Qwen fp16: ≤ 4.8e-5.
  - Granite fp16: ≤ 5.1e-5 (limit 1e-3). Argmax 1.0 for all.
  - Granite fp32 is split over 2 GPUs, so its P2 was not run, as the charter states.
- **P3 (graph = eager):** every captured fp16/nf4 graph matched eager logits exactly (max KL 0.0, argmax equal).
  - bitsandbytes LLM.int8 cannot be graph-captured (CUDA stream-capture error), so it is eager only, as §4 allows.
- **Inference only:** `params_requiring_grad = 0` for every model (the C3-D4 fix).
- **Execution:** no unit errored, was skipped, or was truncated by the budget.
- **Peak memory:** ≤ 1.63 GB (Qwen) and ≤ 7.91 GB (Granite fp16).

## 3. Results at C* (run 2; time [Measured], bytes [Derived])

C* is fp16 + CUDA graph everywhere, except Granite 16K. There, fp16 eager (170.03 ms) and graph (170.07 ms) are tied within CI, and both give the same conclusion.

| Model | T | C* step p50 (ms) | s attn core | s attn proj | s MLP | s lm_head | other floor | bytes MLP / KV | eff. GB/s (÷ copy 241) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Qwen | 4096 | 17.76 | **0.581** | 0.038 | 0.181 | 0.091 | 0.108 | **0.604** / 0.049 | 58.5 (0.24) |
| Qwen | 8192 | 28.13 | **0.730** | 0.025 | 0.122 | 0.059 | 0.070 | **0.576** / 0.093 | 38.7 (0.16) |
| Qwen | 16384 | 53.23 | **0.856** | 0.015 | 0.062 | 0.030 | 0.037 | **0.528** / 0.169 | 22.3 (0.09) |
| Qwen | 32000 | 98.97 | **0.923** | 0.009 | 0.032 | 0.021 | 0.020 | 0.454 / 0.285 | 14.0 (0.06) |
| Granite | 4096 | 65.37 | 0.471 | 0.096 | 0.361 | 0.037 | 0.060 | **0.705** / 0.047 | 109.3 (0.45) |
| Granite | 8192 | 95.33 | **0.636** | 0.067 | 0.247 | 0.024 | 0.045 | **0.673** / 0.090 | 78.4 (0.33) |
| Granite | 16384 | 170.03 | **0.764** | 0.043 | 0.147 | 0.013 | 0.184 | **0.618** / 0.165 | 47.9 (0.20) |

- s_X = 1 − t(no_X)/t(full), measured in the same mode. The "other floor" is t(no_all)/t(full).
- 95% CIs of the p50s are within ±0.5% (`summary_tables.md`).
- **CUDA graphs matter at short context.** Qwen 4K is 17.8 ms graph vs 28.6 ms eager, whose "other floor" of 0.66 is launch overhead. Granite 4K is 65.4 vs 65.9: its larger kernels hide launch cost.
- **Other formats:**
  - nf4 graph is faster than fp16 graph at 4K (Qwen 16.6 ms; Granite 54.6 ms) but fails matched quality, so it cannot be C*.
  - LLM.int8 eager: 143–212 ms (Qwen, fails quality) and 215–302 ms (Granite, passes quality).

## 4. Quality vs dense fp32 (T = 4096, 4 books × 32 tokens)

| Format | Qwen PPL ratio (book-bootstrap CI) / top-1 | Granite PPL ratio (CI) / top-1 | Matched? |
|---|---|---|---|
| fp16 | 1.0007 (1.000–1.002) / 1.000 | 1.0007 (1.000–1.002) / 1.000 | yes / yes |
| nf4 | 1.099 (1.056–1.173) / 0.797 | 1.090 (1.053–1.157) / 0.875 | no / no |
| int8 (LLM.int8) | 1.003 (0.955–1.034) / 0.953 | 0.999 (0.993–1.006) / 0.984 | no (top-1) / yes |

## 5. Exploratory analyses (not pre-registered; `tests/c3_sensitivity.py`, `tests/c3_attn_microbench.py`)

**5.1 Roofline projection** [Derived]: each class's bytes ÷ measured copy bandwidth.
- C* runs above this floor by 4.1× (Qwen 4K) to 17.3× (Qwen 32K), and by 2.2× (Granite 4K) to 5.0× (Granite 16K).
- Projected shares:
  - **MLP** 0.454–0.604 (Qwen) and 0.618–0.705 (Granite), so ≥ 0.5 in both families;
  - attention core ≤ 0.285;
  - lm_head ≤ 0.262;
  - attention projections ≤ 0.176.

**5.2 Attention efficiency at C*:** the removal time is set against the K/V bytes.
- 4.3–4.9 GB/s (Qwen) and 10.3–11.1 GB/s (Granite), i.e. 1.8–4.6% of copy bandwidth.
- At copy bandwidth the same K/V would take 0.21–1.63 ms (Qwen) and 1.40–5.58 ms (Granite), against the 10–130 ms measured.

**5.3 Attention micro-benchmark** (same T4 and torch; one layer's decode attention in a CUDA graph × layers; 63 rows; outputs within 3.3e-5 of fp32).
- **Calibration:** the C* path gives 88.7 ms/step at Qwen 32K and 125.6 ms at Granite 16K. The in-model removal values are 91.3 and 130.0 ms, so the two methods agree within 3–4%.
- **Stock alternatives:** the best stock variant is `repeat_kv` with no mask, at 78.2 ms and 116.5 ms, i.e. ≤ 1.25× faster.
  - Native GQA has no efficient kernel on sm_75. Its math path is slower at ≥ 8K.
- **Caveat:** the first rows (Qwen 4K) read higher than the in-model value (24.5 vs 10.3 ms). They probably ran while clocks were ramping, because there was no global warm-up, so those rows are not used.

**5.4 Reproducibility:** run-2 eager p50 / run-1 eager p50 = 0.82–0.96 over the 10 units valid in both runs.
- Run 2 is consistently faster. Possible causes: run 1's leaked autograd memory (C3-D4), or a different session or machine. Not separated.
- Cross-run variation of this size does not affect any rule outcome, but it bounds how far a single-run ratio should be trusted.

**5.5 Clocks:** snapshots after each context show SM 1590 MHz under load (some snapshots caught idle down-clocking to 585–810 MHz), ≤ 77 °C, and ≤ 46 W against the 70 W limit.
- No evidence of thermal throttling. The snapshots are not continuous traces.

## 6. Defects and incidents in this cycle (none hidden)

| # | Defect / incident | Effect | Status |
|---|---|---|---|
| C3-P0 | Feasibility probe: hashing filter assumed a wrong mount path, so it tried to hash the whole PG-19 corpus | Probe hit its 900 s cap; 913 s of quota | Fixed in the runner (only the listed files are hashed); recorded |
| C3-D1 | No mask function for the custom attention name: StaticCache decode attended to unfilled slots | Found by P2 in the CPU logic test (KL 1.8e-2), before any GPU data | Fixed; regression test |
| C3-D2 | Granite `_update_mamba_mask` host sync broke graph capture | Smoke 1 | Exact patch (no Mamba layers; asserted); A1 |
| C3-D3 | Models not released between formats | Smoke 1: Granite int8 OOM | `gc.collect()` + memory records; A1 |
| C3-D4 | **Graph capture with autograd enabled** | Main run 1: graph timings invalid, OOMs | `no_grad` capture with assertion, `requires_grad_(False)`; A2; full rerun (run 2) |
| C3-D5 | KV layer count missed Qwen's `full_attention` label | Derived KV bytes 0 for Qwen (run 2 as recorded) | Recomputed from verified configs in the analyzer; conclusion unchanged; v1.3 fixes the source; test |
| Q1 | Kaggle queued run 1 for about 75 min and run 2 for about 57 min | No quota cost; delay only | — |

## 7. Limitations

- **Platform:** one T4 (sm_75, 2018 generation), one software stack (PyTorch 2.11 + transformers 4.57.1 + bitsandbytes 0.50.2). Results are not portable to newer GPUs or serving stacks without re-measurement.
- **Baselines missing:** vLLM, TensorRT-LLM, llama.cpp, FlashInfer, AWQ/GPTQ kernels, torch.compile, speculative decoding (B2) and KV-cache quantization.
- **Scope:** two models (0.5B and about 3.4B); contexts ≤ 32K; batch 1.
- **Quality:** at 4K only, 4 × 32 tokens.
- **Bytes and energy:** derived bytes, not counters. No energy data (nvidia-smi board-power snapshots are not energy).
- **Attribution:** the removal method assumes additivity only for reporting. Σ s_X + floor = 1.00–1.02 at the graph-mode C* rows and 1.15 at Granite 16K (eager), so removals are close to additive in graph mode and overlap somewhat in eager mode. s_X (the measured saving from removing X alone) is the quantity the rule uses.
- **Post-data changes:**
  - A2 fixed a code defect and the decision comes entirely from the rerun.
  - C3-D5 corrected derived arithmetic. Before the fix, Qwen's MLP byte share read 0.635 at every context; after it, 0.604 → 0.454. The eligibility outcome is the same either way.

## 8. What this does and does not show

**Shown, on this T4 stack:** batch-1 decode at C* is dominated by attention time, at ≥ 4K for Qwen and ≥ 8K for Granite. That attention runs at 2–5% of the device's copy bandwidth. On a bandwidth-efficient implementation, MLP weight bytes would dominate instead.

**Not shown:** that a new processor mechanism is needed or useful for either class; any novelty; any energy or silicon property; any behaviour on newer GPUs or serving stacks.

**Honest answer to "what do we have that others do not":** still nothing. This cycle produced a clean measurement method (measured removal, with preconditions and P3), and a clear indication that on this platform the big decode costs are a known software problem and a known memory-traffic problem.
