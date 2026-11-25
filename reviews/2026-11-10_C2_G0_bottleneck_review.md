# C2-G0 review — Batch-1 decode bottleneck re-selection (cycle 2)

**Version/date:** 1.0 / 2026-11-10
**Charter:** `specs/C2_G0_batch1_decode_bottleneck_charter.md`, pre-registered with amendments A1–A3 (`results/c2/preregistration.json`).
**Data:** `results/c2/` (index and reproduction commands in `results/c2/README.md`).
**Lenses applied:** A1, A3, A4, A6, A8, A10, A11 (prior art named only; no new search in this cycle).
**Missing:** independent review; GPU/accelerator, energy and DRAM counters.
**Owner:** this AI-assisted session. It is not an independent sign-off.

## 1. Decision, evidence, unresolved risk, next gate

**Decision: C2-G0 = HOLD. Not PASS.**

**What the pre-registered rule printed.** `tests/c2_analyze.py` printed "eligible: mlp". MLP is eligible only under rule 2, the byte view. One thing stops that output from being accepted as the gate decision (item 1). A second limits what it would mean even if accepted (item 2):

1. **The charter §3 precondition for Granite is not met as written.** The standing decode-consistency check (KL ≤ 1e-3 and argmax 1.0) fails for Granite bf16 on 3 of 4 books on this host. This is the same residual G5-R recorded.
   - Under the charter text, Granite results are not admissible.
   - Without Granite, no class meets the two-family condition of rules 1 and 2, and rule 5 applies.
   - A post-hoc fp32 diagnostic on genuine truncated Granite (§3) bears on the *cause* of the residual. It shows the cache logic exact to fp32 rounding (KL ≤ 1.3e-11), while the same network in bf16 also fails the criterion. This supports bf16 rounding, not a cache defect, as the cause. It cannot retroactively satisfy the precondition.
2. **Even if Granite is admitted, the only eligible class is a byte-share class whose eligibility depends on the baseline.**
   - No class is time-eligible in both families, under any pairing of the repeated dense runs (§2, §5).
   - The measured time ceilings for MLP are 1.86× (Granite 4K) and 1.54× (Qwen 4K).
   - MLP byte-eligibility holds for the dense bf16 C*. Dense bf16 is C* only because no faster int8 path met matched quality on this CPU.
   - Under a derived weight-only baseline (int8 or int4) that keeps lm_head in bf16, Qwen's MLP byte share stays below 0.5 at every context (§5.3).

**Evidence:**
- Every number below is CPU wall clock on one VM (host C), or derived bytes.
- 2 genuine models: Qwen2.5-0.5B and Granite-4.0-micro-base.
- PG-19 text, contexts 4K–32K, batch 1.

**Unresolved risks:**
- CPU kernel efficiency dominates the time breakdown. Effective bandwidth is only 24–72% of the measured triad (§2.1–2.2), so time shares here are not those of a bandwidth-bound accelerator.
- B1 is weak on CPU: W8A8 fails quality, and weight-only int8 is slow.
- Granite admissibility (§3).
- Two small models.

**Next gate:** a decision for the user (§9). My recommendation is a dated charter change (a C3 charter) rather than G1 for "MLP weight traffic".

## 2. Results under the pre-registered rule

C* is the fastest configuration meeting matched quality (PPL ratio ≤ 1.01 and top-1 ≥ 0.98). It is **dense bf16 at every context for both models**:
- every torch.ao W8A8 variant fails quality (§4);
- torchao int8 weight-only is slower than dense everywhere (§4);
- dense fp32 is slower than dense bf16.

The shares below are s_up = min(1, s_hooked · (1 + o)) (Amendment A1). Byte shares are **[Derived]**. Time is **[Measured]**.

### 2.1 Qwen2.5-0.5B, C* = dense bf16 (host C)

| T | step ms (p50) | attn core | attn proj | MLP | lm_head | other | KV byte share | MLP byte share | eff. GB/s (÷ triad) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 48.4 | 0.142 | 0.149 | 0.351 | 0.122 | 0.281 | 0.048 | 0.604 | 21.4 (0.43) |
| 8192 | 60.5 | 0.246 | 0.125 | 0.285 | 0.099 | 0.272 | 0.092 | 0.576 | 18.0 (0.36) |
| 16384 | 82.7 | 0.383 | 0.096 | 0.213 | 0.074 | 0.257 | 0.169 | 0.528 | 14.4 (0.29) |
| 32000 | 117.7 | **0.516** | 0.069 | 0.154 | 0.052 | 0.257 | 0.285 | 0.454 | 11.7 (0.24) |

### 2.2 Granite-4.0-micro-base, C* = dense bf16 (host C; admissibility in §3)

| T | step ms (p50) | attn core | attn proj | MLP | lm_head | other | KV byte share | MLP byte share | eff. GB/s (÷ triad) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 218.0 | 0.142 | 0.167 | 0.463 | 0.043 | 0.183 | 0.047 | 0.705 | 32.8 (0.72) |
| 8192 | 276.1 | 0.279 | 0.137 | 0.379 | 0.035 | 0.215 | 0.090 | 0.673 | 27.1 (0.59) |
| 16384 | 389.4 | 0.386 | 0.092 | 0.260 | 0.024 | 0.235 | 0.165 | 0.618 | 20.9 (0.46) |

### 2.3 Rule outcomes

| Class | Rule 1 time (Qwen / Granite) | Rule 2 bytes (Qwen / Granite) | Max Amdahl ceiling, time (Qwen / Granite) |
|---|---|---|---|
| attention core | yes (0.516 at 32K) / no (≤ 0.386) | no (≤ 0.285) / no (≤ 0.165) | 2.06× / 1.63× |
| attention projections | no / no | no / no | 1.17× / 1.20× |
| MLP | no (≤ 0.351) / no (≤ 0.463) | **yes** (0.604 at 4K) / **yes** (0.705 at 4K) | 1.54× / 1.86× |
| lm_head | no / no | no / no | 1.14× / 1.05× |
| other (not a target, rule 3) | no / no | — | 1.39× / 1.31× |

- **Rule 4.** B2 (typical) is slower than plain greedy, at 0.457× (95% CI 0.410–0.463), and its tokens are not identical. So B2 does not raise the bar. Granite's B2 was refused by the library (§4).
- **Rule 5.** It is not triggered if Granite is admitted, because MLP is byte-eligible. It is triggered if Granite is not admitted.

**Observation, outside the rule [Interpretation].** At Qwen 32K, attention-core time share (0.516) is 1.8× its byte share (0.285), and effective bandwidth is 24% of the triad. The CPU SDPA decode kernel is therefore far from bandwidth-bound. A faster kernel would be a software fix, not a processor mechanism.

For these GQA models at ≤ 32K, KV bytes are at most 28.5% (Qwen) and 16.5% (Granite) of the derived bytes per step. This weakens the project's original starting hypothesis (KV traffic in long-context batch-1 decode) for models of this shape and these context lengths. It is not a statement about longer contexts or MHA models.

## 3. Granite admissibility (charter §3 precondition)

**[Measured]** `decode_consistency_granite4micro_T8192_hostC.json`:
- argmax agreement 1.0 on all 4 books;
- max KL 1.61e-3, 2.7e-4, 5.13e-3 and 1.70e-3;
- so **3 of 4 books fail** the KL ≤ 1e-3 criterion;
- max |Δlogit| 0.25–0.54, which is 2–4.3 bf16 ulps at the largest logit.

These values are identical to the G5-R host-A values (`results/g5r/decode_consistency_granite4micro_T8192.json`). Both are CPU model 207 with the same software.

**Charter-design error (mine, recorded as defect C2-D8).** G5-R had already shown that this fp32-calibrated criterion fails for bf16 Granite, and fails 1 of 4 books for bf16 Qwen, whose path is fp32-exact. Even so, I wrote "must pass … before its results are used" into the C2 charter without a bf16 provision. The charter is not amended after the data to admit Granite.

**Post-hoc diagnostic (exploratory)** — `tests/c2_granite_trunc_consistency.py`, output `diag_granite_trunc12_consistency_T8192.json`:
- Full fp32 Granite (about 13.6 GB) does not fit in this VM. So the genuine first 12 decoder layers, embedding, tied lm_head and final norm are loaded in fp32 and in bf16.
- Loaded tensors were checked bit-equal to the checkpoint, and there were no missing keys.
- The unchanged standing check was run at 8K on the same 4 books.

| Network (genuine weights) | dtype | max KL per book (4 books) | argmax agreement | max \|Δlogit\| |
|---|---|---|---|---|
| Full, 40 layers (standing check) | bf16 | 1.6e-3, 2.7e-4, 5.1e-3, 1.7e-3 (3 fail) | 1.0 | 0.25–0.54 |
| Layers 0–11 | **fp32** | 1.3e-11, 6.3e-12, 8.5e-12, 6.9e-12 (all pass) | 1.0 | 1.4e-5–1.7e-5 |
| Layers 0–11 | bf16 | 1.9e-3, 1.7e-3, 8.1e-4, 1.1e-3 (3 fail) | 1.0 | 0.16–0.22 |

Integrity checks on the truncated model:
- 224 checkpoint tensors (layers 12–39, 8 each) were unused, as expected;
- no missing keys;
- 5 sampled tensors were bit-equal to the safetensors checkpoint in both dtypes.

**[Supported hypothesis]** The cached-decode path for this model class is exact to fp32 rounding on the genuine truncated network. The same network in bf16 exceeds the criterion on 3 of 4 books. So the full-model failure is consistent with bf16 rounding and not with a cache-logic defect of the D4b kind, which would also appear in fp32.

Limits of this evidence:
- Layers 12–39 are not exercised in fp32. They use the same code and layer type.
- This is post-hoc evidence, run after the precondition failed.
- **It does not change admissibility under the charter as written.**
- Admitting Granite would take a dated amendment that is explicitly labelled post-hoc. That is a decision for review, not for this session to take silently.

## 4. Baselines

**B1, Qwen fp32, torch.ao dynamic int8 (W8A8)** (`breakdown_b1_qwen05_fp32.json`, A3 rerun, sequential variants):

| T | dense fp32 ms | int8dyn | int8dyn_pc | int8dyn_pc_nolm | Quality (PPL ratio / top-1) |
|---:|---:|---:|---:|---:|---|
| 4096 | 71.9 | 46.7 (1.54×) | 48.1 (1.49×) | 54.7 (1.31×) | 2.68/0.54, 2.70/0.50, 2.67/0.48 — all FAIL |
| 16384 | 127.6 | 90.7 (1.41×) | 91.4 (1.40×) | 104.1 (1.23×) | 3.56/0.44, 3.54/0.42, 3.53/0.45 — all FAIL |
| 32000 | 172.8 | 160.4 (1.08×) | 167.7 (1.03×) | 175.1 (0.99×) | 3.62/0.41, 3.62/0.40, not evaluated (slower) — FAIL |

- Against dense **bf16** (48.4 / 82.7 / 117.7 ms), int8dyn (46.7 ms) and int8dyn_pc (48.1 ms) are marginally faster at 4K, and every variant is slower at 16K and 32K. At 4K, therefore, the quality failure, not speed, is what excludes them from C*.
- Run 1 (interleaved, OOM-killed at 32K) agrees with the sequential rerun within −7.8% to +3.5% at 4K/16K (`sensitivity_exploratory.json`).

**B1, torchao int8 weight-only (`int8wo`)** (A2):
- slower than dense in every arm: Qwen bf16 0.40–0.61× and Granite bf16 0.17–0.29×;
- run-1 Qwen fp32 at 4K/16K: 0.39× and 0.55× (0.392 and 0.545);
- its quality was not evaluated, because it cannot be C* (A2 item 3).

**Baseline weakness, stated plainly.** C* = dense bf16 is the strongest configuration *measured* here. It is probably not the strongest CPU baseline that exists. Optimized CPU weight-only kernels, for example llama.cpp-style Q8/Q4 GEMV, were not installed or tested. On a GPU, weight-only int4/int8 kernels are standard. Every conclusion here is conditional on this baseline set.

**B2, prompt lookup** (Qwen bf16, 8K prefix, 64 new tokens, 3 reps):
- **typical:** 0.457× (CI 0.410–0.463).
  - 55 target forwards vs 64; few drafts are accepted, and multi-token verification is costlier on this CPU.
  - Tokens differ from plain greedy from position 5.
- **favorable** (repeated 2K chunk): 4.39× (CI 3.68–4.49), 6 forwards, identical tokens.
- **Granite:** `ValueError: assisted generation is not supported with stateful models` (transformers 4.57.1), so it was not measured.

## 5. Robustness (exploratory; `tests/c2_sensitivity.py`)

### 5.1 Repeated dense runs

Dense bf16 was measured twice per model: in the breakdown run and in the B1 run. Run-to-run difference: Qwen 3–5% at 4K/16K and 13% at 32K (117.7 vs 132.9 ms); Granite 4–6%.

The analyzer takes the faster run. Recomputing eligibility for all 4 pairings changes nothing:
- attention-core maximum: Qwen 0.516 / 0.508, Granite 0.385 / 0.386;
- MLP maximum: Granite 0.463 / 0.462;
- **no class is time-eligible in any pairing.**

Byte shares are configuration-derived and identical across runs.

### 5.2 Hook overhead

Hook overhead is above the §4 limit of 5% in 5 configurations, with a maximum of 6.3% (Qwen dense fp32 4K). It is at most 5.1% for every C* row. A1's upper bound s_up covers it. Shares near 0.5 (Qwen attention at 32K, 0.516 vs a hooked 0.491) depend on that bound.

### 5.3 MLP byte share under derived weight-only quantization

**[Derived]** These configurations were not run with quality. Bytes per parameter: int8 = 1 (scales neglected, < 0.5%); int4 with group 128 = 0.531. KV is bf16. The table gives the highest MLP byte share over contexts (4K).

| Weights | Qwen | Granite |
|---|---:|---:|
| bf16 (C* here) | 0.604 | 0.705 |
| int8, lm_head int8 | 0.576 | 0.673 |
| int8, lm_head bf16 | **0.461** | 0.630 |
| int4 g128, lm_head int4 | 0.533 | 0.624 |
| int4 g128, lm_head bf16 | **0.325** | 0.530 |

So MLP byte-eligibility survives a stronger weight baseline for Granite. For Qwen, it survives only if lm_head is quantized too. The reason is that Qwen's tied 136M-parameter lm_head is 28% of its bf16 weight bytes.

## 6. Prompt-lookup divergence (task #13; exploratory)

`results/c2/diag_lookup_divergence_qwen05_T8192.json` (v1.0) and `results/c2/diag_lookup_divergence_qwen05_T8192_v1.1.json`:

**[Measured]** v1.1 reproduces B2 token for token. Its argmax reference equals B2's verified manual-greedy path (64 of 64 tokens), and its prompt-lookup output equals B2's lookup output.

**bf16:**
- **First divergence at position 5,** where the top-2 margin is **exactly 0**.
- Tokens 1513 and 2776 have the same bf16 logit (17.0; one bf16 ulp there is 0.125). `argmax` returns 1513, the first index. Prompt lookup emitted 2776, the other tied token.
- Exact ties occur at 4 of 64 positions (5, 21, 25, 55). 11 of 64 positions have a margin of ≤ 0.125.

**fp32:** no exact ties (minimum margin 4.4e-3), and prompt lookup is **identical** to greedy over all 64 tokens.

**[Untested hypothesis]** The multi-token verification forward produces logits that differ in their last bits from one-token decode, so the tie resolves the other way. The verification logits were not captured.

**Implications:**
- On bf16 text inputs, "identical tokens to plain greedy" can fail at exact ties even when the lookup logic is correct.
- This does not affect rule 4 here, because lookup is slower on typical text anyway.
- Scope: one book and 64 tokens.

**Defect C2-D7.** v1.0 took its reference token from `torch.topk`, whose order among exactly tied values differs from `torch.argmax` (first maximal index). HF greedy uses `argmax`. At a bf16 tie, v1.0's reference was therefore not plain greedy: its first divergence (14) was measured against a different path from B2's (5). v1.1 uses `argmax`. The v1.0 file is kept unchanged.

## 7. Defects and incidents in this cycle (none hidden)

| # | Defect / incident | Effect | Status |
|---|---|---|---|
| C2-D1 | B1 v1.0: the module-name qconfig spec also mapped `nn.Embedding` | Crash, no data | Fixed (Linear-only mapping); regression test with an embedding |
| C2-D2 | First queue's `step()` masked exit codes (G5-R D8 pattern) | The D1 crash logged rc=0 | Later queues capture rc first; every output checked for completeness |
| C2-D3 | torchao `Int8Tensor`: `element_size()` reports the outer dtype, and `data_ptr()` is 0 | Byte counts and dedupe wrong | Fixed (count inner tensors); test |
| C2-D4 | My commit message claimed "222 tests pass" while 6 failed in a full-suite run (global float64 default set by another test module) | False claim in the record | Corrected in a follow-up commit; autouse fixture |
| C2-D5 | `generate` with prompt lookup overshoots `max_new_tokens` by 1–2 | False "not identical" in the smoke test | Compare the first n tokens; overshoot recorded |
| C2-D6 | Qwen fp32 B1 run 1 OOM-killed at 32K | Partial data | Kept; A3 rerun with sequential variants |
| C2-D7 | Divergence diagnostic v1.0 used `topk` tie order | Wrong reference at ties | v1.1 (§6) |
| C2-D8 | Charter §3 precondition written without a bf16 provision despite the known G5-R result | Granite inadmissible as written | Recorded; not amended after the data (§3) |

## 8. Limitations, and what this review does not show

- **Scope:** CPU only (one VM, 4 threads); two small models (0.5B and about 3.4B); one timing book; teacher-forced quality on 4 books × 32 tokens; B2 on one book and 64 tokens.
- **Bytes:** derived bytes, not counters. No energy data.
- **Attribution:**
  - "Other" (0.18–0.28) includes MLP activation and element-wise products, norms, RoPE, residuals, cache update and Python dispatch.
  - Only the Linear layers are attributed to MLP. So the MLP time shares are lower bounds of the MLP block's cost, and the attention-core share excludes RoPE and the cache update.
- **Missing points:** fp32 int8wo at 32K is not measured (A3). Granite W8A8 is not run (fp32 Granite does not fit in this VM). No KV-cache quantization backend was installed (the charter's optional item was not run).
- **Host dependence:** time shares depend on the host. G5-R measured Granite attention at 56–70% of decode on host B; host C gives ≤ 39%. The cause was not investigated, and the hosts are not pooled.
- **Not novelty:** eligibility is not novelty. Batch-1 weight streaming is among the most heavily addressed problems in the field, by weight-only quantization (e.g. GPTQ arXiv:2210.17323, AWQ arXiv:2306.00978), activation sparsity and near/in-memory designs. Earlier branches in this record that touched it (weight batching, near-memory compute, progressive bitplanes) closed on G1 novelty.

## 9. Options for the next step (user decision)

1. **[Recommended] Charter change (C3), with a dated rationale.** Re-measure the breakdown on hardware whose decode is closer to bandwidth-bound, with the strongest available kernels: fused attention decode, and weight-only int8/int4 GEMV with measured quality.
   - The free option is a Kaggle GPU notebook (T4/P100) on the connected account. It uses the account's free weekly GPU quota and needs the user's explicit go-ahead.
   - Pre-register a bf16-appropriate consistency criterion, based on the Qwen bf16 control, before any data.
   - Optionally add an optimized CPU baseline (llama.cpp) here.
2. **G1 prior-art pass on "MLP weight traffic at batch 1", conditional on admitting Granite.** Its outcome is likely to be dominated by known methods (§8). It is listed because the rule printed it, not because I expect a novel opening.
3. **Stop cycle 2 at HOLD** and keep the measurements as a reference.

I recommend option 1. On this CPU, every class is below the 2× time ceiling except attention at Qwen 32K (2.06×, a single-family, kernel-inefficiency effect). The byte view, which is the only route to eligibility here, is meaningful only on a bandwidth-bound device with a strong weight baseline. That device and baseline cannot be measured in this VM.
