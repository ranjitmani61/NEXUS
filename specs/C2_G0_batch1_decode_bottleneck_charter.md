# C2-G0 — Batch-1 decode bottleneck re-selection charter (cycle 2)

**Version/date:** 1.0 / 2026-11-10 (pre-registered before any cycle-2 measurement; the hash and time are in `results/c2/preregistration.json`).
**Governing protocol:** `AGENTS.md` v1.0, read in full on 2026-11-10.
**Gate:** G0 for cycle 2; its outputs feed G1/G2 hypothesis selection.
**Lenses:** A1, A3, A4, A6, A8, A10, A11. **Missing:** independent review; GPU/accelerator, energy and DRAM counters (none available).
**Owner:** this AI-assisted session. Not an independent sign-off.

## 1. Why this cycle exists

Cycle 1 ended with **G5 FAIL** for every NEXUS attention-page candidate (`reviews/2026-11-04_G5R_real_model_gate_review.md`). In that study, attention was 15–49% of batch-1 decode time on Qwen2.5-0.5B (host A) and 56–70% on Granite-4.0-micro (host B).

Every other branch in the record is closed by G1 novelty failure or suspended: weight batching, near-memory compute, progressive bitplanes, lossless KV codec, temporal delta and photonic reduction (`processor_project_01.md`).

Before any new mechanism is proposed, this cycle measures where batch-1 decode time and bytes actually go on genuine models, and how much remains after the strongest available baselines. **This cycle proposes no mechanism and makes no novelty claim.**

## 2. Falsifiable question

On genuine pretrained models at batch 1, is there a single cost class X whose elimination would give ≥ 2× end-to-end over the **best measured baseline configuration** at matched quality?

- If such an X exists, it is eligible as the target for cycle-2 hypothesis generation (G1/G2).
- If none exists, single-bottleneck mechanisms cannot reach the 2× target on this workload, and the charter must change (workload, batch, platform or target metric) before new hypotheses are generated.

## 3. Workload (fixed)

- **Models**, same weights and provenance as G5-R (`results/g5r/provenance_*.json`):
  - Qwen2.5-0.5B: bf16 for the breakdown; fp32 for the int8 baseline, which requires fp32.
  - IBM Granite-4.0-micro-base: bf16.
- **Data:** PG-19 test books from `results/g5r/pg19_selection.json`.
- **Contexts:** Qwen 4K/8K/16K/32K; Granite 4K/8K/16K. Batch 1 throughout.
- **Platform:** this VM, 4 threads. The host is fingerprinted in every output. Results from different hosts or VMs are **never pooled**.
- **Cache:** Granite uses the fixed hybrid cache (`tests/g5r_cache.py`). Every Granite generation path must pass the standing decode-consistency check (`tests/g5r_decode_consistency.py`) on this host before its results are used.

## 4. Measurements (pre-registered)

**M1 — measured per-component decode-time breakdown.**
- Teacher-forced decode with a dense prefill, as in `tests/g5r_timing.py`: 16 steps × 5 reps, seeded shuffled order, fresh cache copy per rep.
- Wall clock is attributed to five classes:
  - **attention core** (the SDPA call);
  - **attention projections** (q/k/v/o Linear);
  - **MLP** (all MLP Linear);
  - **lm_head**;
  - **other** (embedding, norms, RoPE, residuals, cache update, framework/Python dispatch), computed as step time minus the sum of the hooked classes.
- Instrumentation: forward pre/post hooks on leaf `nn.Linear` modules plus the attention-function timer. CPU execution is synchronous, so the intervals are meaningful.
- Hook overhead is measured by timing the same configuration with and without hooks. If it exceeds 5% of step time, it is reported as a limitation.

**M2 — derived bytes per decode step, by stream.**
- Weight bytes: all Linear weights read once per step at batch 1, lm_head included.
- KV bytes: 2 · L_attn · n_kv · d_head · T · bytes, read, plus one token written.
- Activations: reported but expected to be negligible.
- This is **[Derived]** from the configuration and dtype; no hardware counters exist.

**M3 — measured host memory bandwidth.**
- A STREAM-like triad (`a = b + s·c` on 1 GiB tensors, 4 threads), median of 5.
- Derived effective decode bandwidth = (weight + KV bytes) / measured step time. Effective bandwidth near the triad value indicates bandwidth-bound decode on this host; well below it indicates compute- or overhead-bound.

**B1 — int8 dynamic quantization** (`torch.ao.quantization.quantize_dynamic`, all Linear, qint8) on Qwen fp32 at 4K/16K/32K.
- Timing as in M1.
- Quality: teacher-forced, 4 books × 32 tokens, against the fp32 dense model.
- Matched quality is the G5-R rule: PPL ratio ≤ 1.01 and top-1 ≥ 0.98.
- Granite is out of B1's scope: fp32 Granite (about 13.6 GB) does not fit in this VM's 15 GB with a KV cache, and no bf16 int8 path is installed. This is recorded, not hidden.
- B1 is a CPU stand-in for weight-only quantization (GPTQ arXiv:2210.17323, AWQ arXiv:2306.00978). It is not those methods.

**B2 — prompt-lookup speculative decoding** (`generate(..., prompt_lookup_num_tokens=10)`, greedy, as introduced by apoorvumang/prompt-lookup-decoding; speculative decoding: Leviathan et al., arXiv:2211.17192).
- Setup: 64 new tokens after an 8K-token prefix, on both models, measuring end-to-end tokens/s against plain greedy `generate` (same host, interleaved, 3 reps).
- The emitted tokens must be identical to plain greedy. Any mismatch is reported and investigated, and the speedup is not counted.
- Two input kinds, labelled in advance:
  - **typical:** a book continuation;
  - **favorable:** a prefix that repeats one 2K-token chunk, so copying is likely.
- Only the typical case enters the eligibility rule. The favorable case is reported for context.

**Optional (decided now, run only if feasible on this VM):** KV-cache quantization (KIVI, arXiv:2402.02750, or an HF `QuantizedCache` backend). No backend is installed. It is attempted only if a package installs from the public index without paid resources. Otherwise it is recorded as not run.

## 5. Decision rule (pre-registered)

Let C* be the fastest configuration among {dense, B1} that meets matched quality, for each model and context.
- s_X is the share of class X in C*'s measured decode-step time (M1).
- b_X is the share of X in C*'s derived bytes (M2), where defined.

1. **X is time-eligible** if s_X ≥ 0.5 in at least one configuration **for each model family**. Eliminating X entirely would then give ≥ 2× over C*.
2. **X is byte-eligible** if b_X ≥ 0.5 under the same condition. This is the optimistic bandwidth-bound-device view and is reported separately from rule 1.
3. **"Other" is not a processor target.** If it dominates, the finding is "software overhead dominates on this platform"; it does not make X eligible.
4. **B2 sets an additional bar.** If B2 (typical) is faster than C* with identical tokens, a new mechanism must exceed 2× over B2 as well. Eligibility for that target is re-derived using B2's tokens/s.
5. **If no X is eligible under rule 1 and rule 2 for both families:** "no single-bottleneck target reaches 2× on this workload". The next step is a charter change, not a new mechanism.

Eligibility is **not** novelty, and **not** feasibility. Any eligible class proceeds to G1 (prior art) and G2 (feasibility) under its own contract.

## 6. Disqualifiers and integrity

- Each result is recorded with model, dtype, context, host fingerprint, torch/transformers versions and seeds.
- Raw per-step times are saved.
- Any run whose decode-length or cache-length guard fails is quarantined, as in G5-R.
- No result from a different host or VM is mixed into a ratio.
- Thresholds in §5 are not changed after data collection. Exploratory analyses are labelled as such.

## 7. Reproduce

Commands are recorded in `results/c2/README.md` as runs complete.

## Amendment A1 (2026-11-10, before any recorded cycle-2 run; motivated by the pipeline smoke test)

The smoke test was Qwen fp32 at T = 512, 4 decode steps, 1 rep, 8 quality tokens on one book. Its output was written to the scratchpad and is not a result. It showed two things:
- B1 as registered (per-tensor qint8, all Linear) gave PPL ratio 4.6 and top-1 0.25. A weak baseline would set C* too low and make eligibility too easy.
- Hook overhead was 10–16%, above the 5% expected in §4.

Changes, which make the "no eligible target" conclusion **harder** to reach, never easier:

1. **The B1 family is extended** with two standard dynamic-int8 settings:
   - `int8dyn_pc`: per-channel weight scales (`per_channel_dynamic_qconfig`), all Linear;
   - `int8dyn_pc_nolm`: per-channel, lm_head kept fp32.

   All three variants are timed and quality-checked and all are reported. C* is the fastest **dense or B1-family** configuration that meets matched quality.
2. **Rule 1 uses an upper bound on each class's share** of the un-instrumented step: s_X^up = min(1, s_X^hooked · (1 + o)), where o is the measured hook overhead. This assumes the hooks never make a class's own work faster. A class is rejected only if even s_X^up < 0.5. The hooked shares and o are reported alongside.

Nothing else changes.
