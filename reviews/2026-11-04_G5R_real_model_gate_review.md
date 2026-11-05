# G5-R Gate Review — NEXUS candidates on genuine pretrained transformers

**Dates:** 2026-11-04 to 2026-11-05 · **Version:** 1.0
**Governing protocol:** `AGENTS.md` v1.0, read in full before work, unchanged.
**Contract:** `specs/G5R_real_model_benchmark_contract.md` v1.0 plus amendments A1–A5 (scope and order only; thresholds unchanged) and notice N1 (the platform changed mid-study). SHA-256 and UTC times are in `results/g5r/preregistration.json`.
**Review lenses applied:** A1, A2, A3, A4, A6, A8, A10, A11. **Missing:** any independent human reviewer, GPU and energy measurement (impossible on this platform), and A5/A7/A9/A12 (not applicable to a CPU software study).

This is a "signed-off-style" engineering report written by one AI-assisted session. It is **not** an independent sign-off.

---

## 1. Decision

**G5: FAIL for every existing NEXUS attention-page candidate** (NX-SPH, NX-BOX, NX-ONE, NX-VAL), and, by the derived family bound in §5.5, for valid-bound variants such as anchor-delta reuse and guarded mixed-precision metadata.

This follows the pre-registered rule (contract §11): *FAIL if, on any genuine pretrained family in the primary setting, a candidate shows a violation, or cannot reach matched quality, or reaches it with < 2× end-to-end speedup.* On Qwen2.5-0.5B, the candidates reach matched quality only because they read essentially every KV page. Their measured end-to-end decode speed is **0.04×–0.80× of dense**, never above 1×, at every context (4K–32K) and batch size (1 and 4).

| Evidence (Qwen2.5-0.5B, genuine weights) | Value | Label |
|---|---|---|
| Mean fraction of KV pages read: NX-BOX / NX-SPH / NX-VAL, 4K → 32K | 0.996→0.978 / 1.000→0.996 / 0.996→0.978 | Measured |
| KV-head union of NX-BOX pages, every layer, 4K | 1.000 | Measured |
| NX-ONE single-page admission rate, 4K | 0.0 | Measured |
| Best possible for *any* valid-bound mass-certificate method (OR-SCORE), 4K → 32K | 0.640 → 0.526 | Measured oracle |
| Certificate violations (log-domain evaluation), 2 × 86,016 + 43,008 + 21,504 + 2 × 21,504 head-steps | **0** | Measured |
| End-to-end decode speed, NX-BOX vs dense (bf16, batch 1 / batch 4) | 0.29×→0.08× / 0.10×→0.04× | Measured |
| Zero-overhead ceiling for the whole family, worst case for this claim (batch 4, 32K) | 1.38× (< 2×) | Derived |

**This is not a BLOCKED result.** Genuine weights, real data and valid measurements were obtained. The failure mechanism (§5.2) is a property of the bounds on real attention, independent of hardware: page-read fractions do not depend on the CPU. What remains unverified (§7) cannot turn this FAIL into a PASS, because the candidates do not skip KV reads on real models. A faster machine cannot make a method that reads ≥ 97.8% of the KV cache, plus metadata, ≥ 2× faster through KV savings.

**Second family (IBM Granite-4.0-micro-base, 8K, 4 books) confirms the result:**
- NX-BOX / NX-SPH / NX-VAL read 99.9% / 99.98% / 99.9% of pages, with 0 violations; OR-SCORE is 0.572.
- Measured speed (host B, native-GQA dense) is 0.365× (8K) and 0.163× (16K) for NX-BOX, and 0.93× / 0.89× for NX-ONE.
- The family ceiling at 8K is 1.32×.

Under the contract a PASS needs both families and FAIL needs one; both families fail.

## 2. What was tested

| ID | Candidate (archived definition) | How it was tested |
|---|---|---|
| NX-SPH | mean/radius page bound + mass certificate `2U/(Z+U) ≤ 0.02`, greedy | shadow (every layer/head/step), live, timed |
| NX-BOX | min/max page bound + the same certificate | shadow, live, timed |
| NX-ONE | one-page admission or dense fallback | shadow (admission rate), live, timed |
| NX-VAL | value-geometry certificate with box bounds | shadow; live at 4K |
| OR-SCORE | exact minimum page count for **any** mass-certificate method whose page bound is ≥ the true page max (derived in contract §7; brute-force verified) | shadow |
| OR-MASS | minimum for any mass-only certificate, even with exact page masses | shadow |

**Baselines:**
- **Dense:** PyTorch CPU SDPA with native GQA (see defect D5 for the Granite correction).
- **Quest-style:** top-k pages by min/max bound, page 16, budgets 512–4096, first 2 layers dense.

**Models:**
- **M1:** Qwen2.5-0.5B base, Kaggle `qwen-lm/qwen2.5/transformers/0.5b` v1.
- **M2:** IBM Granite-4.0-micro-base, Kaggle `ibm-research/granite-4.0/transformers/granite-4.0-micro-base` v1.

**Data:**
- PG-19 test books chosen by a fixed rule (8 at 4K/8K, 4 at 16K, 2 at 32K);
- passkey needles in PG-19 haystacks;
- uniform-random-token and highly repetitive adversarial inputs.

Decode: 32 teacher-forced steps after a dense prefill.

## 3. Provenance and validity

- **M1 Qwen2.5-0.5B:**
  - SHA-256 for all 10 files.
  - No publisher signature exists, so provenance rests on the QwenLM Kaggle organisation plus a functional check: PG-19 perplexity 13.1–25.9, against 189,356 for a same-config random-init control.
  - Hugging Face revision cross-check impossible (host blocked).
- **M2 Granite:**
  - Sigstore bundle cryptographically verified: DSSE, identity `Granite-verify@ibm.com`, IBM OIDC issuer, offline trust root.
  - 11/11 signed file digests and the subject digest match the local SHA-256 values.
  - Perplexity 8.5–15.9, against 116,416 for random init.
- **Capture:**
  - `g5r` dense logits are bit-identical to stock SDPA for both models.
  - The float64 shadow reconstruction is within 2.2e-6 (Qwen fp32) and 2.4e-3 (Granite bf16) of the output norm.
- **Decode consistency** (cached decode against a full forward, at 8K):
  - Qwen fp32: max logit difference 1.7e-5, KL ~1e-12.
  - Granite bf16, after the fixes: argmax agreement 1.0, but max KL up to 5.1e-3, which **fails** the pre-set KL ≤ 1e-3 criterion on 3 of 4 books.
  - A bf16 control on Qwen, whose path is verified correct in fp32, gives KL up to 1.8e-3 and also fails it on 1 of 4 books. So the criterion, set with fp32 in mind, is too strict for bf16. Granite's residual is the same order as normal bf16 cached/batched differences. The criterion is not relaxed after the fact. Caveat: the control ran on host B and the Granite check on host A.
- **Independent verification:** a separately written NumPy/mpmath checker and the archived 2026-11-03 oracle agree with the harness on all 294 real query heads checked (0 mismatches, after checker fix D3).

## 4. Defects found (all investigated, none hidden)

| # | Defect | Effect | How found | Status |
|---|---|---|---|---|
| D1 | Archived certificate evaluation: shift by max bound, then `exp` → **underflow** | `cert = 0` while error > 0 on real Qwen layer 0; naive evaluation stops early in hundreds of head-steps per run (table §5.1) | 4 "violations" in the first smoke run | Fixed: log-domain evaluation; regression test |
| D2 | Kaggle served one PG-19 book as a **zip** | Binary data would have been tokenised as text | Size mismatch | Fixed: detect and extract; size check |
| D3 | Independent checker: `u = max(0, u − bet)` → **catastrophic cancellation** (the same pattern as `processor_baseline.py`) | One false early stop (115/128) | Checker/harness mismatch on a real head | Fixed: mpmath tail sums |
| D4a | Granite returns **no KV cache** unless one is passed | Decode attended only to the current token (T=1); perplexity 2,967 | Reconstruction "exactly 0.0" | Fixed: explicit cache and length guards |
| D4b | transformers 4.57.1: an **empty hybrid cache reports length = batch size** | Prefill RoPE positions offset; book 3340 perplexity 298 vs 8.3 | Cross-model comparison, then per-layer probe (keys differ 15.9%, values identical) | Fixed: subclass; standing decode-consistency check |
| D5 | Under a custom attention name, Granite's mask builder caused the SDPA wrapper to use **`repeat_kv` copies** | Granite dense baseline 4.6× slow per layer; inflated Quest speedups and Amdahl ceilings | Implausible attention share (73–82%) | Fixed: native-GQA dense; Qwen verified unaffected |
| D6 | The log filter (`grep -v warn`) **hid** the transformers warning for D4a | Delayed detection | Post-mortem | Later queues log unfiltered; guards make it a hard error |
| D7 | The container restart **moved the session to a different CPU** (model 207 with AMX → model 85 without AMX/AVX512-BF16), and `g5r_timing.py` v1.0 **hard-coded** the original CPU string | Post-restart timing metadata claimed the wrong host; bf16 prefill about 7× slower | 7× prefill slowdown → `lscpu` | Host fingerprint recorded by every driver; post-hoc labelled annotation; `host_log.json`; notice N1 (no cross-host pooling) |
| — | Two synthetic test constructions of mine were wrong (stale witness; one-head alignment) | Tests failed for the wrong reason | Unit tests | Corrected and documented |

All invalid outputs are kept in `results/g5r/invalid/` with explanations. The diagnostic scripts and outputs are in `tests/g5r_diag/` and `results/g5r/diag/`.

## 5. Results (complete tables: `results/g5r/summary_tables.md`)

### 5.1 Pages read on real traces (Qwen2.5-0.5B, shadow mode, every layer × head × decode step)

| T | books | head-steps | NX-BOX | NX-SPH | NX-VAL | NX-BOX reads all pages | OR-SCORE | OR-MASS | violations | naive-eval false early stops (box/sph) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4096 | 8 | 86,016 | 0.996 | 1.000 | 0.996 | 95.9% | 0.640 | 0.418 | 0/0/0 | 361/984 |
| 8192 | 8 | 86,016 | 0.994 | 0.999 | 0.993 | 94.7% | 0.618 | 0.393 | 0/0/0 | 375/1430 |
| 16384 | 4 | 43,008 | 0.987 | 0.998 | 0.987 | 88.2% | 0.562 | 0.340 | 0/0/0 | 247/906 |
| 32000 | 2 | 21,504 | 0.978 | 0.996 | 0.978 | 77.3% | 0.526 | 0.316 | 0/0/0 | 259/720 |
| 4096 random tokens | 2 | 21,504 | 0.995 | 1.000 | 0.994 | 94.9% | 0.744 | 0.561 | 0/0/0 | 52/223 |
| 4096 repetitive (PPL 1.00) | 2 | 21,504 | 0.996 | 1.000 | 0.996 | 96.2% | 0.626 | 0.405 | 0/0/0 | 171/539 |

Per layer at 4K (`layers_qwen05_T4096.json`), NX-BOX's **KV-head union is 1.000 in every one of the 24 layers**. Under GQA a page can be skipped only if all 7 query heads sharing it skip it, and that never happened. The KV traffic saved is therefore **zero**.

### 5.2 Mechanism: why the certificate never fires

On 21 real captured decode samples (294 heads, 37,632 page bounds; `adversarial_metadata_qwen05_T4096.json`), the bound slack (upper bound minus the true page max score) was:
- **box:** median **16.2 nats** (p05 10.4, min 5.9);
- **sphere:** median **24.4 nats**.

For comparison, the spread of the true page maxima *within a head* has a median of **13.7 nats**. The bound overestimate is larger than the whole range of real page relevance: the least relevant page's bound exceeds the most relevant page's true maximum. Bound mass per page is overestimated by about e¹⁶ ≈ 10⁷, so `U` (unvisited bound mass) cannot fall below `0.0101 · Z` until almost every page has been visited.

### 5.3 Live end-to-end quality (teacher-forced; book-cluster bootstrap 95% CI)

- **NX candidates:** PPL ratio 1.0000, KL 0, top-1 agreement 1.0 at every context. This is trivially so, because they compute (nearly) dense attention.
- **Quest** (the comparator, not a candidate):
  - meets the matched-quality rule (PPL ≤ 1.01 and top-1 ≥ 0.98) only at **budget 2048 @ 4K** (reading 78% of KV pages) and **budget 4096 @ 8K** (78%);
  - fails at every tested budget at 16K and 32K. For example, budget 4096 @ 16K gives PPL ratio 1.022 [1.004, 1.041] and top-1 0.992, reading 53% of pages.
- **Passkey (16K, 6 trials):** dense, NX-BOX and all Quest budgets 6/6 correct. Further contexts: §5.8.

### 5.4 Measured CPU decode speed (bf16; speedup = dense time / method time, including metadata build/update, screening, gathers and fallback)

| batch | T | dense attention share f | Amdahl ceiling 1/(1−f) | NX-BOX | NX-SPH | NX-ONE | Quest-1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 0.145 | 1.17× | 0.29× | 0.27× | 0.65× | 0.59× |
| 1 | 8192 | 0.243 | 1.32× | 0.24× | 0.23× | 0.76× | 0.68× |
| 1 | 16384 | 0.373 | 1.60× | 0.18× | 0.17× | 0.71× | 0.76× |
| 1 | 32000 | 0.494 | 1.98× | 0.08× | 0.09× | 0.69× | 1.02× [0.93, 1.15] |
| 4 | 8192 | 0.382 | 1.62× | 0.10× | — | 0.76× | 0.54× |
| 4 | 16384 | 0.517 | 2.07× | 0.05× | — | 0.75× | 0.86× |
| 4 | 32000 | 0.586 | 2.42× | 0.04× | — | 0.80× | 1.04× [0.94, 1.26] |

Dense bootstrap self-ratio CIs (the noise floor) are about ±10–13%.
- **Measured** for Qwen on host A (4 vCPU Xeon family 6 model 207, AMX-BF16), 16 steps × 5 reps, interleaved.
- **Granite** (host B: Xeon family 6 model 85, no AMX; native-GQA dense in 320/320 calls; 8 steps × 3 reps per A4):

| model | host | batch | T | f | Amdahl ceiling | NX-BOX | NX-ONE | Quest-1024 | Quest-2048 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Granite | B | 1 | 8192 | 0.564 | 2.29× | 0.365× [0.338, 0.374] | 0.93× | 1.35× | 0.66× |
| Granite | B | 1 | 16384 | 0.705 | 3.39× | 0.163× [0.159, 0.171] | 0.89× | 1.75× | 1.36× |

  Quest-1024's speedup on Granite comes with **unmatched** quality (top-1 agreement 0.914 at 8K). Quest-4096, the budget that does meet matched quality, was not timed on Granite (A4).
- The earlier Granite timing with a handicapped dense baseline (D5) is quarantined. Do not compare host-A and host-B speeds.
- **Not** GPU, **not** energy (no counter), **not** llama.cpp.
- Quest here is an unoptimised Python-gather implementation, so its speed is a **weak** comparator. Its quality numbers are unaffected.

### 5.5 [Derived] Ceiling for the whole NX family

For any method that must read a page fraction ≥ r while attention is a fraction f of decode time, the end-to-end speedup is ≤ 1/(1 − f(1 − r)) at zero overhead. Using the measured f and r = OR-SCORE, this is optimistic for the candidates: it assumes no overhead and uses the per-query-head fraction rather than the per-KV-head union.

| batch | T | family ceiling | page fraction needed for 2× | NX-BOX zero-overhead ceiling | NX-BOX measured |
|---:|---:|---:|---|---:|---:|
| 1 | 4096–32000 | 1.06× → 1.31× | impossible (f ≤ 0.5) | 1.001× → 1.011× | 0.29× → 0.08× |
| 4 | 16384 | 1.29× | ≤ 0.033 | 1.007× | 0.05× |
| 4 | 32000 | 1.38× | ≤ 0.147 | 1.013× | 0.04× |
| Granite, 1 (host B) | 8192 | 1.32× | ≤ 0.113 | 1.001× | 0.365× |

Even with *perfect* per-page score bounds, a mass certificate at tol 0.02 needs ≥ 53% of pages on Qwen and ≥ 57% on Granite (8K), while 2× would need ≤ 15% and ≤ 11% respectively.

### 5.6 Independent verification

- 294 real heads; harness, mpmath checker and archived oracle agree (0 mismatches).
- The v1.0 float64 checker's mismatch is kept as evidence (D3).
- Unit and property tests: 108 pass (67 core + 41 timed-path; synthetic math and timed-path correctness only). **Tests passing does not establish hardware correctness.**

### 5.7 Adversarial results

- **Unguarded BF16/FP16 metadata** and **worst-case stale metadata** (smallest-box slot reuse) gave **0** bound violations and 0 false certificates on 294 real heads. This is **not** robustness by design: the 5.9-nat minimum slack absorbs these perturbations. The hazard is real (the synthetic witness stands) but is masked by bounds that are too loose to be useful.
- **Naive (archived-pattern) evaluation** produces false early stops on every real run (§5.1, last column). **Any implementation must evaluate in the log domain without subtract-from-total.** This matters far more in fp32/bf16 hardware arithmetic.
- **Error accumulation:** live runs replace attention in all layers for 32 steps. Because NX reads (nearly) everything, it accumulates no error; Quest's local per-head error exceeds 0.02 in 21–41% of head-steps at budget 1024 (37–54% at budget 512), which shows up as KL 0.011–0.044 and top-1 agreement 0.80–0.94.

### 5.8 Second family (Granite-4.0-micro-base) and remaining runs

**Granite 8K traces** (4 books, 204,800 head-steps, all three Granite fixes, decode-length guards passed):

| T | NX-BOX | NX-SPH | NX-VAL | NX-BOX@16 | NX-BOX reads all pages | OR-SCORE | OR-MASS | violations | naive-eval false early stops (box/sph) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8192 | 0.999 | 1.000 | 0.999 | 0.993 | 95.0% | 0.572 | 0.326 | 0/0/0 | 535/9,889 |

**Live quality (Granite 8K):**
- NX-BOX: PPL ratio 1.0055 [0.998, 1.013] against bf16 SDPA dense, but **1.0002 [0.995, 1.004] against the float64 dense64 control**. So the small offset is arithmetic precision, not page selection: dense64 itself is 1.0053 against bf16 dense.
- Quest meets matched quality only at budget 4096, reading 70% of KV pages. Quest-1024 has top-1 agreement 0.914.

**Passkey retrieval** (6 trials per row: 3 depths × 2 keys; question and answer decoded with the method; prefill dense and shared). Entries are correct/6 and the mean fraction of KV pages read during the answer:

| model, T | host | dense | NX-BOX | Quest-1024 | Quest-2048 | Quest-4096 |
|---|---|---|---|---|---|---|
| Qwen 16,384 | A | 6/6 | 6/6 (0.987) | 6/6 (0.24) | 6/6 (0.36) | 6/6 (0.54) |
| Qwen 32,000 | B | 6/6 | 6/6 (0.978) | **3/6** (0.17) | **5/6** (0.24) | 6/6 (0.36) |
| Granite 8,192 | B | 6/6 | 6/6 (0.999) | 6/6 (0.27) | 6/6 (0.44) | 6/6 (0.71) |

- The failures are Quest's, at 32K, at depths 0.1 and 0.5. The wrong answers continue the book (e.g. "in the drawer.", "the key to the pass.").
- **This test does not discriminate the NX candidates.** They pass because they read 97.8–99.9% of pages, which is the reason they FAIL on speed.
- With 6 trials, 6/6 has a 95% Clopper–Pearson interval of [0.54, 1.00] and 3/6 of [0.12, 0.88]. The passkey rows are a sanity check, not a precise accuracy estimate.
- The Qwen 16K row ran before the restart (host A). Its host record is empty: the v1.0 host parser bug, fixed later. Host A is inferred from the queue log time (00:02 UTC, before the 00:13 restart).

**Qwen fp32 cached-decode consistency on host B** (4 books, 8K, 3 steps): max KL 4.7e-12, max |Δlogit| 2.0e-5, argmax agreement 1.0. It passes the pre-set criterion, so the cached decode path is exact to fp32 rounding on the post-restart host too (`results/g5r/decode_consistency_qwen05_T8192.json`).

**Granite 16K and 4K traces, Qwen 8K passkey:** *[pending at the time of this revision; added below when complete].*

## 6. Failure analysis summary

1. The mass certificate needs `U ≤ 0.0101 Z`. Real bounds overestimate per-page mass by about 10⁷ (median), so the certificate is unsatisfiable until about 98–100% of pages are visited.
2. Even an oracle with exact page maxima needs 53–64% of pages (OR-SCORE), because real attention is spread across many pages (OR-MASS 32–42%) and the 0.02 tolerance requires 99% of the mass.
3. With grouped-query attention, the page actually read is the union over the 7 query heads of a group; for NX-BOX it is 100%.
4. Per-step screening cost (bounds, sort, doubling gathers) is paid on top of what is effectively a dense read, so the measured speed is 0.04–0.80×. It gets worse with context and batch.
5. The batch-1 Amdahl ceiling is < 2× at every context Qwen supports (≤ 32K). Batch 4 lifts it to 2.42×, but the family bound (5.5) keeps the NX family ≤ 1.38×.

## 7. What remains unverified (explicitly)

- No GPU, FlashAttention-GPU, vLLM or llama.cpp baseline; no energy measurement (no counter on the VM); no HBM/DRAM byte counters.
- No optimised (CUDA) Quest. Quest's *speed* here is a weak comparator; its quality is valid for this adaptation (GQA union, page 16, first 2 layers dense).
- Only two model families, both small (0.5B and 3.4B); long-context instruction or retrieval benchmarks (RULER, LongBench) not run.
- Qwen weights not cross-checked against the Hugging Face revision (host blocked). Granite's signature was checked offline against an embedded trust root, without revocation check.
- Granite cached-decode residual (KL up to 0.0051) fails the pre-set criterion. A bf16 Qwen control on a verified path shows the same order (≤ 1.8e-3), which supports, but does not prove, that it is ordinary bf16 rounding. The control ran on a different host.
- **The platform changed mid-study** (notice N1). Qwen timings are from host A (AMX), Granite timings and post-restart traces from host B (no AMX). Page-read fractions are host-independent; attention shares and speeds are not, and they are not pooled.
- The decision covers attention-page candidates. The photonic, near-memory, codec, bitplane, temporal-delta and batching branches were **not** re-tested here; their earlier gate status stands (no G5 evidence).

## 8. Gate status after this review

| Gate | Status |
|---|---|
| G1 prior art | unchanged: page screening overlaps Quest and others; novelty not established |
| G3 algorithm | **conditional PASS** for the log-domain certificate on real traces (0 violations); **FAIL** for the archived float64 evaluation pattern (D1, D3) |
| G4 simulation / complete costs | **FAIL** for a benefit: full costs measured, slower than dense |
| **G5 real workload** | **FAIL** (all NX attention-page candidates; M1 alone suffices under the pre-registered rule) |
| G6–G12 | **NOT AUTHORIZED**; no basis for microarchitecture, RTL, hardware or a breakthrough claim |

**Recommended next step:** do not invest further in certified page screening as the processor thesis. If attention-traffic work continues, it should start from a method whose page-read fraction on real models is far below 15% at matched quality, which none of the tested mechanisms achieve. Otherwise, return to hypothesis selection (G1/G2) with a different bottleneck. At batch 1 on these models, weights, not KV, dominate decode below about 32K tokens.
