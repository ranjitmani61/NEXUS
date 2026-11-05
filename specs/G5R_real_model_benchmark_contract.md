# G5-R Real-Model Benchmark Contract — pre-registration

**Version/date:** 1.0 / 2026-11-04 (written before any real-model comparison data was produced)
**Gate:** G5 — Real workload. Also touches G3 (correctness on real traces) and G4 (complete costs).
**Governing protocol:** `AGENTS.md` v1.0, read in full before this work. Not modified.
**Review lenses applied:** A1, A2, A3, A4, A6, A8, A10, A11. Missing: independent human review of any kind; A5/A7/A9/A12 not applicable to a CPU software study.
**Owner:** AI-assisted analysis in a single cloud session. No independent reviewer.

## 1. Question

Do any *existing* NEXUS attention-page candidates give a defensible end-to-end advantage over a strong, production-relevant dense baseline on genuine pretrained transformers, at matched output quality, after charging every metadata, preprocessing, fallback and runtime cost?

This is a falsification study. Its job is to find out whether the candidates work, not to show that they do.

## 2. Platform and hard limits (known before running)

- 4 vCPU Intel Xeon (family 6, model 207, Emerald Rapids class; AVX-512, AMX-BF16), 15 GiB RAM, **no GPU**, no RAPL/powercap interface.
- Consequences fixed in advance: **no GPU measurement, no energy measurement** (energy will be reported as *not measured*, never estimated as if measured), no FlashAttention-GPU or vLLM baseline.
- `huggingface.co` is refused by the environment network policy (HTTP 403 on CONNECT). Weights and data come from Kaggle (official publisher organisations for weights). Hugging Face revision hashes therefore cannot be cross-checked; local SHA-256 of every file is recorded instead.
- Gated models (Llama 3.2, Gemma 3) return "User has not consented to terms of use". Accepting a licence on the user's account is an external action outside this mission's authorisation and was **not** done.

## 3. Models (exact sources)

| ID | Model | Source | Role |
|---|---|---|---|
| M1 | Qwen2.5-0.5B (base) | Kaggle `qwen-lm/qwen2.5/transformers/0.5b` version 1, publisher QwenLM, Apache-2.0 | primary family A |
| M2 | Granite-4.0-micro-base | Kaggle `ibm-research/granite-4.0/transformers/granite-4.0-micro-base` version 1, publisher IBM Research, Apache-2.0 | family B (unrelated) |

Provenance manifests: `results/g5r/provenance_*.json`. Functional check that the weights are pretrained: PG-19 perplexity must be far below the ~vocabulary-size perplexity of an untrained model. A model that fails this is excluded rather than used.

## 4. Candidates under test (all pre-existing in the record)

| ID | Definition (taken from archived specs) | Source in record |
|---|---|---|
| NX-SPH | Page key mean `c`, radius `r = max ||k-c||`; score bound `(q·c + ||q|| r)·scale`; visit pages in descending bound; exact softmax over visited pages; stop when mass certificate `2U/(Z+U) <= 0.02` | `processor_baseline.py`, G0 audit |
| NX-BOX | Same, with coordinate min/max (box support) bound | G0 audit "box", G4 dynamic-KV min/max |
| NX-ONE | NX-BOX restricted to admitting exactly one page; otherwise dense fallback | G4 hybrid admission / dynamic KV / compiled CPU |
| NX-VAL | Value-geometry certificate (`max_box_fraction` over unvisited page mass bounds and value-ball distances), box score bounds | G4 value-geometry, G5 trace oracle |

The current partial page (fewer than `page_size` tokens) is always visited exactly. Metadata covers full pages only, and is updated as soon as a page fills. Normalisation is by `Vmax = max ||v_i||` over visible tokens of that KV head, as in the archived specs.

**Family lower bound (derived, §7):** any mass-certificate method that uses a *valid* per-page score upper bound (NX-SPH, NX-BOX, NX-ONE, anchor-delta reuse, guarded mixed-precision metadata) needs at least as many pages as **OR-SCORE**, the minimum over all page subsets under exact per-page maximum scores. OR-SCORE is computed exactly. **OR-MASS** (exact page masses known) is the minimum for any mass-only certificate whatsoever. Both oracles require full attention and are diagnostic only.

Out of scope here (no real-model path or hardware): photonic, near-memory, progressive bitplane, lossless codec, batching/queueing. They are not evaluated, and their gate status is unchanged.

## 5. Baselines

- **B-DENSE:** `torch.nn.functional.scaled_dot_product_attention` (PyTorch CPU kernel) inside the same Hugging Face model, same precision, same threads. This is a production-relevant *CPU framework* baseline. It is **not** the strongest CPU runtime (llama.cpp) and not GPU FlashAttention; that gap is disclosed.
- **B-QUEST:** Quest-style query-aware page selection (Tang et al., ICML 2024): page size 16, box upper bound per query head, top-K pages within a token budget `B ∈ {512, 1024, 2048, 4096}`, current page always included, first two layers dense. No certificate. Grouped-query models: per-query-head selection; KV traffic counted as the union per KV head. Quest was published for MHA; this adaptation is ours.

## 6. Workloads

- **Text:** PG-19 test split from Kaggle `tunguz/the-pg19-language-modeling-benchmark-dataset`. Selection rule fixed before reading any content: books sorted by numeric id, first 8 with at least 250,000 bytes (`results/g5r/pg19_selection.json`).
- **Contexts:** 4K, 8K, 16K, 32K tokens for M1 (32K is its native maximum); 4K, 8K, 16K for M2, adding 32K only if runtime permits. Any context not run is reported as not run.
- **Decode:** after prefilling `T` tokens, 32 teacher-forced decode steps on the book's true continuation.
- **Passkey retrieval:** a 5-digit key inserted at depth {10%, 50%, 90%} in a PG-19 haystack, with prompt format following Mohtashami & Jaggi (2023). Greedy generation; exact match. Seed 20261104.
- **Adversarial inputs:** uniformly random token ids (anti-structure), and a highly repetitive input.
- **Batch:** 1 is primary; batch 4 is secondary, for latency only.

## 7. Derived dominance result used in the analysis

Let page `j` have exact softmax mass `a_j` and exact bound mass `b_j = n_j exp(m_j - s) >= a_j`, where `m_j` is the exact page max score. Let `t = 0.02` and `c = t/(2-t)`. The mass certificate holds for visited set `S` iff `sum_{j not in S} beta_j <= c sum_{j in S} a_j`, where `beta_j` is the bound mass actually used. A valid bound has `beta_j >= b_j`, so its stopping set also satisfies the condition with `b`. The condition with `b` is `sum_{j in S}(b_j + c a_j) >= sum_j b_j`. With unit cost per page, the smallest such `S` is the greedy prefix by `b_j + c a_j`. Hence `|S_method| >= OR-SCORE` for every valid-bound mass-certificate method. Exact arithmetic; numerical evaluation in float64.

## 8. Measurements

- **Trace (shadow mode, no effect on the model):** every layer and query head at each decode step: pages visited per method, per query head and per-KV-head union; certificate; true error `||O - O_S|| / Vmax` and `/ ||O||`; violations (`true > cert·(1+1e-6) + 1e-9`).
- **End-to-end (live mode, candidate output replaces attention):** per-token NLL, perplexity ratio to dense, KL(dense‖candidate), top-1 agreement, passkey accuracy.
- **Systems (bf16 and, for M1, fp32):** per-token decode latency p50/p95 over repeated runs, tokens/s, attention-only time, metadata build and update time, peak RSS, KV and metadata bytes, Amdahl ceiling `1/(1-f)` where `f` is the measured attention share of decode time. Five timed repetitions after warm-up, interleaved, with median and bootstrap 95% CI.
- **Energy:** not measured (no counter).

## 9. Independent verification

1. Separately written float64 NumPy recomputation of certificates and errors on a random sample of captured (layer, head, step) traces.
2. The archived `tests/g5_trace_oracle.py::evaluate_one` on a sample of real traces (cross-implementation agreement on the oracle quantities).
3. Capture validity: the custom dense attention against SDPA outputs and full-model logits against the stock model.

## 10. Adversarial tests

- Stale page metadata (a missed update), measured on real keys.
- Unguarded BF16/FP16-rounded min/max metadata (bound violations on real keys).
- Passkey needle; random-token input.
- Error accumulation across layers and decode steps (local certificate against global KL/NLL).
- Admission rate against cache length.

## 11. Pre-registered decision rules

**Matched quality:** perplexity ratio to dense `<= 1.01`, top-1 agreement `>= 0.98`, and passkey accuracy `>=` dense accuracy on the same cases.

**A NEXUS candidate PASSES G5 only if all of the following hold:**
- Zero certificate violations.
- Matched quality.
- `>= 2.0×` measured end-to-end decode speedup (tokens/s) against B-DENSE, including all metadata, fallback and selection overhead.
- On **both** model families, at batch 1, at the longest context run on both (at least 8K).
- Not worse than B-QUEST at matched quality, otherwise it offers no advantage over published prior art.

**FAIL** if, on any genuine pretrained family in the primary setting, a candidate shows a certificate violation, or cannot reach matched quality, or reaches it with less than 2× end-to-end speedup. This asymmetry is deliberate: the claim under test is a *robust* advantage, so one clear counterexample on a representative real model refutes it within the tested scope.

**BLOCKED** if essential evidence (a genuine model, or a valid measurement) cannot be obtained. **BLOCKED is never reported as PASS.**

Thresholds are not changed after data is seen. Any deviation will be recorded as a dated amendment with its reason, and the original criteria will still be reported.

## Amendment A1 — 2026-11-04 (recorded after the M1 4K run started, before any 8K/16K/32K or M2 data)

**Reason:** CPU runtime. Measured cost is about 1 minute per book at 4K for M1. Extrapolated cost is several hours per configuration at 32K for M1, and for every M2 context (M2 has 40 layers and 40 query heads, and is 3.4B parameters in bf16).

**Changes (scope only; no threshold, metric or decision rule changes):**
1. **M1 at 32K:** the first 4 books by the selection rule, not 8.
2. **M1 at 32K and all of M2:** live runs restricted to `{nx_box, quest:1024, quest:2048, quest:4096}`.
   - `nx_sph`, `nx_one` and `nx_val` are still evaluated on **every** head-step in shadow mode, which computes their exact selections.
   - Whenever they read all pages, their live output is the dense output; this was confirmed by the M1 4K live runs.
3. **M2:** the first 4 books by the selection rule at each context run.
4. **Implementation note, fixed before the passkey runs:** the passkey prompt is prefilled only up to the end of the haystack. The question and answer tokens are decoded one at a time with the method under test. Otherwise the first answer token would come from the dense prefill, and the test could not discriminate between methods.

**Also recorded — numerical defect found and fixed before any reported data:** the archived certificate evaluation pattern (shift by the maximum upper bound, then `exp`) underflowed on real Qwen2.5 layer-0 keys and reported false certificates (`cert = 0` while error > 0). The candidates are now evaluated in the log domain (`g5r_core` v1.1). The naive pattern's early-stopping events are still counted (`*_naive_eval_stops_earlier_heads`) as an adversarial finding.

## Amendment A2 — 2026-11-04 (recorded after 3 of 8 M1 4K books, before any 8K/16K data)

**Reason:** measured about 2.5 min per book at M1 4K with all 8 live methods. In those books the live `nx_*` runs reproduced the dense output exactly (ΔNLL = 0, KL = 0, top-1 agreement = 1) because they read 99.6–100% of pages.

**Change:**
- For M1 at 8K and 16K, live runs are also `{nx_box, quest:1024, quest:2048, quest:4096}`.
- All four NX candidates and both oracles remain evaluated on every head-step in shadow mode.
- If any shadow run at 8K/16K shows an NX candidate reading under 95% of pages (mean), that candidate gets its own live run at that context.

**Order of remaining work, by decision value:** M1 timing (bf16, exclusive CPU) → M2 validation and M2 8K → M1 8K and 16K → M1 32K → passkey → M2 timing → remaining M2 contexts → adversarial inputs.

No metric, threshold or decision rule is changed.

## Amendment A3 — 2026-11-04 (recorded with M1 4K complete, M1 batch-1 timing complete, M1 8K at 1 of 8 books; no M1 16K/32K, M2 quality, batch-4 or passkey data yet)

**Reasons:**
1. **A harness defect invalidated the first M2 runs.** IBM Granite returned no KV cache, so decode attended only to the current token. The details are in `results/g5r/invalid/README.md`; the fix is `tests/g5r_cache.py` plus cache-length guards. All M2 quality, timing and passkey runs are re-run with the fix.
2. **The batch-1 M1 timing shows a measured Amdahl ceiling of 1.98× at 32K.** So, for M1 at batch 1, no attention-only method can meet the 2× criterion on this platform. The only setting that can still change that conclusion is batch > 1, where KV traffic grows relative to the weight traffic. The batch-4 timing therefore moves up the order.
3. **Runtime.**

**Scope changes:**
- M1 16K: 4 books instead of 8.
- M1 32K: 2 books instead of 4.
- M1 batch-4 timing: contexts 8K, 16K and 32K.
- M2: 8K with 4 books, 16K with 2 books, 4K with 4 books.

**New order:** M1 8K (running) → M1 batch-4 timing → M2 8K → M2 batch-1 timing → M1 16K → M1 32K → passkey M1 (16K, 32K, 8K) → passkey M2 8K → adversarial inputs → remaining M2 contexts.

No threshold, metric or decision rule is changed.

## Amendment A4 — 2026-11-04 (recorded during M1 batch-4 32K timing; no M1 16K/32K traces, M2 traces or timing, passkey or adversarial-input data yet)

**Reason:** runtime. M1 batch-4 timing at 32K is taking more than 45 minutes, because NX runs at a few percent of dense speed. M2 has 40 query heads and would be slower still.

**Changes (scope and order only):**
- **M2 timing:** 3 reps × 8 steps (instead of 5 × 16), methods `{dense, quest:1024, quest:2048, nx_box, nx_one}`. `nx_sph` is dropped because it matched `nx_box` within CI at every timed M1 context.
- **New order:**
  1. M2 8K traces
  2. M1 16K traces
  3. M1 32K traces
  4. M2 timing at 8K and 16K
  5. adversarial inputs (M1 4K)
  6. passkey M1 16K
  7. passkey M1 32K
  8. passkey M2 8K
  9. M2 16K traces
  10. passkey M1 8K
  11. M2 4K traces

Thresholds and decision rules are unchanged.

## Amendment A5 — 2026-11-04 (order only; recorded before any valid M2 trace, M2 timing or passkey result exists)

**Reason:** a second and a third Granite defect were found and fixed:
- an empty hybrid cache reports length = batch size, which offset RoPE positions;
- the dense timing baseline was handicapped by `repeat_kv`.

Both are documented in `results/g5r/invalid/README.md`. All prior M2 trace and timing outputs are invalid. The M2 evidence that the decision rule needs ("both families") is therefore moved ahead of the passkey and optional runs.

**New order, after the M1 16K passkey that is running:**
1. M2 decode-consistency check (4 books)
2. M2 8K traces
3. M2 timing
4. M1 decode-consistency check
5. passkey M2 8K
6. passkey M1 32K
7. M2 16K traces
8. passkey M1 8K
9. M2 4K traces

Thresholds and decision rules are unchanged.

## Notice N1 — 2026-11-05: platform changed mid-study (a fact, not a design choice)

The cloud container was restarted between 00:09 and 00:13 UTC on 2026-11-05, and came back on a different CPU.

| | Before (host A) | After (host B) |
|---|---|---|
| CPU | Xeon family 6 model 207 | Xeon family 6 model 85 |
| AMX-BF16 / AVX512-BF16 | yes | no (AVX-512F only) |
| L3 | 260 MiB | 33 MiB |

**Rules applied from this point:**
- Speed ratios are taken only within one run, on one host.
- Attention shares, Amdahl ceilings and absolute speeds are reported per host and never pooled.
- The family-ceiling bound uses f and r from the same model, with the host stated.

`results/g5r/host_log.json` maps every result file to its host. No threshold or decision rule is changed.
