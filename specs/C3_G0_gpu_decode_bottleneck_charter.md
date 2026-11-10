# C3-G0 — Batch-1 decode bottleneck on a GPU with stronger baselines (cycle 3)

**Version/date:** 1.0 / 2026-11-10. Pre-registered before any C3 GPU measurement. The hash and time are in `results/c3/preregistration.json`.
**Governing protocol:** `AGENTS.md` v1.0, read in full on 2026-11-10.
**Gate:** G0 for cycle 3; its outputs feed G1/G2 hypothesis selection.
**Lenses:** A1, A3, A4, A6, A8, A10. **Missing:** independent review; energy measurement (nvidia-smi board-power samples are recorded but are **not** an energy measurement); DRAM/HBM counters.
**Owner:** this AI-assisted session. Not an independent sign-off.
**User decision:** on 2026-11-10 the user chose option 1 of `reviews/2026-11-10_C2_G0_bottleneck_review.md` §9, a charter change with re-measurement on a free Kaggle GPU.

## 1. Why this cycle exists

C2-G0 ended **HOLD** (`reviews/2026-11-10_C2_G0_bottleneck_review.md`), for three reasons:
- **Platform:** the CPU host reached only 24–72% of its measured bandwidth at decode, so its time shares reflect CPU kernel efficiency.
- **Baselines:** weak — W8A8 int8 failed quality, and weight-only int8 was slow.
- **Granite:** inadmissible under the charter as written.

This cycle repeats the bottleneck question on a GPU, where batch-1 decode is expected to be closer to bandwidth-bound, with the strongest baselines that are **available and installable for free** there. **This cycle proposes no mechanism and makes no novelty claim.**

## 2. Falsifiable question

Unchanged from C2: on genuine pretrained models at batch 1, is there a single cost class X whose elimination would give ≥ 2× end-to-end over the **best measured configuration at matched quality** (C*)?

## 3. Platform and workload (fixed)

**Platform:** Kaggle "GPU T4 x2", from the feasibility probe of 2026-11-10 (inventory only, not a result):
- 2× Tesla T4 (sm_75, 15 GB, 70 W), driver 580.178.04, CUDA 13.0;
- torch 2.11.0+cu130, transformers pinned in-session to **4.57.1** (the version of all earlier NEXUS results), bitsandbytes installed in-session at its latest version, which is recorded.
- Timing uses **cuda:0 only**. cuda:1 is used only to hold half of the fp32 Granite quality reference.
- The docker image digest is recorded. Results are never pooled with C2 (CPU) or G5-R numbers.

**Models:** Qwen2.5-0.5B and Granite-4.0-micro-base, attached as Kaggle Models version 1.
- **Precondition P1:** the SHA-256 of every model file and of the four books must equal `results/g5r/provenance_*.json`. A model failing P1 is not run.

**Data:** PG-19 test books 860, 2544, 3247 and 3340. Timing uses book 860.

**Contexts:** Qwen 4096/8192/16384/32000; Granite 4096/8192/16384. Batch 1.

**Working dtype: fp16.** T4 tensor cores support fp16, not bf16. fp16 is a deployment change from the models' bf16 release dtype, so it must pass the quality test (§5) like any other configuration.

## 4. Configurations and measurements (pre-registered)

**Weight formats** (each timed in two modes):
- `fp16`: dense fp16 weights.
- `nf4`: bitsandbytes 4-bit NF4 weights, fp16 compute. This is weight-only quantization. Modules that bitsandbytes skips by default (e.g. lm_head) stay fp16, and are recorded via the per-class weight bytes.
- `int8`: bitsandbytes LLM.int8() (int8 weights with fp16 outlier decomposition). It is **not** weight-only.

**Modes:**
- **eager:** HF forward with a StaticCache (Granite: a StaticCache subclass with `has_previous_state = False`; all 40 of its layers are attention) and the stock SDPA kernel.
- **graph:** the same decode step captured once per configuration in a CUDA graph. A graph configuration is valid (**precondition P3**) only if its logits equal eager logits on the timed steps: argmax equal at every step, and max KL ≤ 1e-5. Capture failures are recorded and remove only that mode.

**Not available or not included, stated as baseline weaknesses:**
- vLLM, TensorRT-LLM, llama.cpp, exllamav2, AWQ/GPTQ kernels, and torch.compile/inductor fusion;
- a FlashDecoding-style split-K attention kernel (PyTorch's flash kernel needs sm_80+);
- speculative decoding (C2's B2) and KV-cache quantization.

**M1 — measured removal.** Per format, mode and context:
- A chunked prefill (2048-token chunks) into a StaticCache with max_cache_len = T + 16.
- Then 16 teacher-forced decode steps × 5 reps, plus 1 warm-up rep. Per rep, the (mode, ablation) units run in a seeded, shuffled, interleaved order. Wall clock is taken per step with a device synchronize; all inputs are staged on the device before timing.
- Ablations: `full`; `no_attn_core` (SDPA replaced by zeros); `no_attn_proj`; `no_mlp` (MLP Linear only); `no_lm_head` (each replaced by a zero tensor of the output shape); and `no_all`.
- **s_X = 1 − t(no_X) / t(full)** on p50 step times. This is the measured fraction of the step saved by removing X, in the same mode.
- t(no_all)/t(full) is the measured "other" floor: norms, RoPE, element-wise ops, masks, cache writes, residuals and launch overhead. Additivity, Σ s_X plus the floor against 1, is reported.

**M2 — derived bytes per step.** Linear weight storage per class (packed storage plus quantization scales), plus K/V read over max_cache_len (fp16). **[Derived]** No counters.

**M3 — measured device bandwidth.** Copy and triad on 1 GiB fp32 tensors on cuda:0, median of 5. Effective decode bandwidth = derived bytes / C* p50 step time.

**Quality.** Teacher-forced full forward (no cache) on the 4 books × 32 scored tokens at T = 4096, for every format, against **dense fp32** (Granite fp32 split over the two GPUs with `device_map="auto"`).
- Budget deviation from C2: quality is measured at 4K only.
- Matched quality: PPL ratio ≤ 1.01 **and** top-1 ≥ 0.98 against fp32.

**Precondition P2 — decode consistency, dtype-aware, fixed now before any GPU data.** Static-cache decode is compared with a full forward at T = 4096, 3 steps, on 4 books.
- It passes if argmax agreement is 1.0 and max KL ≤ **1e-6 (fp32)** / **1e-3 (fp16)**.
- Basis for 1e-3 at fp16:
  - the fp32 static path gave KL ≤ 9e-12 in the CPU logic test;
  - bf16 residuals of correct paths were 1.6e-3 to 5.1e-3 (Granite) and ≤ 1.8e-3 (Qwen) in G5-R/C2;
  - the CPU fp16 logic test gave ≤ 4.3e-5;
  - fp16 has 3 more mantissa bits than bf16.
- P2 applies to the dense fp16 and fp32 paths. The fp32 check runs for Qwen only, because Granite fp32 is split over two devices. A model whose fp16 path fails P2 is **inadmissible**. **No post-hoc override** (C2 defect C2-D8 is not repeated).

## 5. Decision rule (pre-registered; C2 §5 carried over, with M1 as defined here)

C* is the configuration (format × mode) with the lowest p50 `full` step time that passes P1–P3 (P3 for graph mode) and matched quality, for each model and context. Then:
1. **Time-eligible:** s_X ≥ 0.5 at C* at ≥ 1 context, **for each model family**.
2. **Byte-eligible:** b_X ≥ 0.5 (M2 at C*) under the same condition. This is reported separately from rule 1.
3. **"Other" is never a processor target.** If the floor dominates, the finding is "software/launch overhead dominates on this platform".
4. **B2 (speculative decoding) is not measured in C3,** so rule 4 is not applied. Any eligible class needs a speculative-decoding comparison before G2. Recorded as a limitation.
5. **If no class is eligible under rule 1 or rule 2 for both families:** "no single-bottleneck target reaches 2× on this workload". The next step is a charter change.

C* near-ties: CIs (bootstrap of the median) are reported. Eligibility is re-evaluated with every C* candidate whose CI overlaps C*'s. If the conclusion differs between such candidates, the decision is **HOLD**.

## 6. Execution, budget and integrity

**Budget:** at registration about 5,300 s of weekly GPU quota remain (refresh 2026-11-14 00:00 UTC). The feasibility probe used 913 s; it hit its cap because of a path bug in its hashing filter, which was mine.

**Run 1, smoke:**
- Every code path at tiny sizes (T = 512/1024, 4 steps, 1 rep); session cap 900 s.
- Its outputs are **not results**.
- It may motivate a dated amendment before run 2. Such an amendment may only make eligibility harder to reach, or fix a defect.

**Run 2, main** (`tests/c3_build_kernel.py --plan main`):
- Session cap 3,000 s. Internal budgets: Qwen 1,000 s, Granite 1,500 s.
- Any unit not completed within budget is recorded as **BLOCKED** and run only after the quota refresh, with the same code (its SHA-256 is recorded).

**Integrity:**
- The code under test (`tests/c3_gpu_breakdown.py`) is embedded in the kernel verbatim (base64), and its SHA-256 is checked inside the session.
- Raw per-step times are saved. nvidia-smi clocks, temperature and power snapshots are saved at stage boundaries, to disclose throttling.
- Thresholds in §4–§5 do not change after data. Exploratory analyses are labelled.

**Pre-data defect, already fixed:** C3-D1. The custom attention name had no registered mask function, so transformers 4.57.1 built no mask, and StaticCache decode attended to unfilled slots (fp32 KL 1.8e-2). It was found by the P2 check in the CPU logic test. Fixed, with a regression test (`tests/test_c3_gpu_breakdown.py`).

## 7. Reproduce

```bash
python tests/c3_build_kernel.py --plan main --output kernel_main.py
python -I tests/c3_kaggle.py push kernel_main.py --title nexus-c3-main --gpu --internet --timeout 3000 \
  --model qwen-lm/qwen2.5/transformers/0.5b/1 ibm-research/granite-4.0/transformers/granite-4.0-micro-base/1 \
  --dataset tunguz/the-pg19-language-modeling-benchmark-dataset
python -I tests/c3_kaggle.py output <user>/nexus-c3-main --dest results/c3/main
```

## Amendment A1 (2026-11-10, after the GPU smoke run; before any main-run data)

The smoke run (Kaggle kernel `nexus-c3-smoke` v1; T = 512/1024; outputs are not results) exercised every path.

**What worked:**
- pip pin (transformers 4.57.1, bitsandbytes 0.50.2).
- P1 path matching on the real Kaggle mounts.
- Qwen: the fp32 reference, P2 for fp16 (KL ≤ 4.5e-5), and CUDA graphs for fp16 and nf4, whose graph logits were identical to eager (KL 0).
- The Granite fp32 reference split over two GPUs, and Granite P2 for fp16 (KL ≤ 2.9e-4).

**Two defects, fixed in `tests/c3_gpu_breakdown.py` v1.1:**
- **C3-D2.** Granite graph capture failed in every configuration. `GraniteMoeHybridModel._update_mamba_mask` evaluates `if cache_position[0] > 0`, a host sync that invalidates capture. Its result is used only by Mamba layers, and Granite-4.0-micro has none. The method is replaced by one that returns None, under an assertion that every layer is attention. On CPU the patched and unpatched consistency values are identical. P3 still gates every graph configuration.
- **C3-D3.** Models were not released between formats: Granite int8 OOM'd at load with 14.48 GiB held. `gc.collect()` now runs before `empty_cache()`, and the memory still allocated after each release is recorded.

**Also seen** (smoke, tiny samples, not results):
- bitsandbytes int8 (LLM.int8) cannot be graph-captured; eager only, as allowed by §4.
- nf4 quality looked poor (Qwen PPL ratio 1.39 on 4 × 8 tokens).

**Unchanged:** thresholds, rules, configurations, contexts and budgets.

**Added:** one Granite-only re-smoke (same sizes; not a result) to verify both fixes on the GPU before the main run.
