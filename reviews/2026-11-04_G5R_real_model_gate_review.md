# G5-R Gate Review — NEXUS candidates on genuine pretrained transformers

**Date:** 2026-11-04 · **Version:** 1.0 (DRAFT, filled in as runs complete)
**Governing protocol:** `AGENTS.md` v1.0, read in full before work, unchanged.
**Contract:** `specs/G5R_real_model_benchmark_contract.md` v1.0 + amendments A1, A2. SHA-256 and UTC times in `results/g5r/preregistration.json`; v1.0 was recorded before any comparison data existed.
**Review lenses applied:** A1, A2, A3, A4, A6, A8, A10, A11.
**Missing:** any independent human review, A5/A7/A9/A12 (not applicable to a CPU software study), and GPU or energy measurement (impossible on this platform). All work was done by one AI-assisted session. This is a "signed-off-style" report, **not** an independent sign-off.

---

## 1. Decision

*(to be completed from data; see §8)*

## 2. What was tested, exactly

| ID | Candidate (archived definition) | Tested how |
|---|---|---|
| NX-SPH | mean/radius page bound + mass certificate `2U/(Z+U) <= 0.02`, greedy | shadow on every layer/head/step; live; timed |
| NX-BOX | min/max page bound + same certificate | shadow, live, timed |
| NX-ONE | one-page admission or dense fallback | shadow (admission rate), live, timed |
| NX-VAL | value-geometry certificate with box bounds | shadow; live at M1 4K |
| family bound | OR-SCORE: the exact minimum page count for **any** valid-bound mass-certificate method (covers anchor-delta reuse and guarded mixed precision) | shadow |

**Baselines:** dense (stock PyTorch CPU SDPA in transformers 4.57.1) and Quest-style top-k pages (page 16, budgets 512–4096, first 2 layers dense).

**Models** (genuine, pretrained):
- **M1:** Qwen2.5-0.5B base (Kaggle `qwen-lm/qwen2.5/transformers/0.5b` v1).
- **M2:** IBM Granite-4.0-micro-base (Kaggle `ibm-research/granite-4.0/transformers/granite-4.0-micro-base` v1).

**Data:** PG-19 test split (8 books chosen by a fixed rule); a passkey needle in PG-19 haystacks; random-token and repetitive adversarial inputs.

## 3. Provenance and validity checks (all measured)

**M1 Qwen2.5-0.5B**
- All 10 files are recorded with SHA-256 (`results/g5r/provenance_qwen2.5-0.5b.json`), and byte counts equal Kaggle's listing.
- No publisher signature is available, so provenance rests on the QwenLM Kaggle organisation, the config (consistent with the published architecture), and function:
  - PG-19 perplexity over 2,048 tokens: 13.1–25.9 across the 8 books.
  - On the same 512 tokens: pretrained 28.2, against **189,356** for a random-init control of the same config (≈ vocabulary 151,936).
- Hugging Face revision cross-check: **not possible** (huggingface.co blocked by network policy).

**M2 Granite-4.0-micro-base**
- The Sigstore bundle `model.sig` **verifies cryptographically**:
  - DSSE signature checked with sigstore-python 4.5.1, offline embedded trust root.
  - Identity `Granite-verify@ibm.com`, issuer `https://sigstore.verify.ibm.com/oauth2`.
- All 11 signed file digests equal the local SHA-256 values.
- The subject digest recomputed from local hashes equals the signed subject digest (`55da5cb2…aec`).
- Limitation: offline trust root, so no online revocation check. See `results/g5r/provenance_granite_signature_verification.json`.

**Capture validity (M1)**
- `g5r` dense logits are bit-identical to stock SDPA (max |Δ| = 0.0).
- The float64 shadow reconstruction matches the model's SDPA output: max abs 8.3e-6 at 1,000-token decode (fp32 rounding level). Across the 4K runs, max abs ≤ 2.5e-5.

**PG-19 data:** 8 test books, with SHA-256 and listed sizes (`results/g5r/provenance_pg19_selected.json`).

## 4. Defects and discrepancies found and investigated (none discarded)

1. **Archived certificate evaluation underflows on real data (G3-relevant).**
   - The archived code shifts by the maximum upper bound and then applies `exp`. On real Qwen2.5 layer-0 keys, loose bounds push the remaining terms below float64's range, giving `U = 0`, so `cert = 0` while the true error is > 0.
   - In the first smoke run this produced 4 "violations". Root cause: evaluation arithmetic, not the theorem.
   - Fix: log-domain evaluation (`g5r_core` v1.1, regression test added).
   - Naive-evaluation false early stops are still counted on every run (e.g. 361 NX-BOX and 984 NX-SPH head-steps of 86,016 at M1 4K).
   - In fp32 or bf16 hardware arithmetic, where `exp` underflows near −103 instead of −745, this defect would be far more frequent.
2. **Kaggle served one PG-19 file as a zip container** (book 3608: 538,473-byte zip instead of 1,434,234-byte text). Detected through a size mismatch, then extracted and verified. Without the check, binary data would have been tokenised as text.
3. **Two of my own synthetic test constructions were wrong:**
   - The stale-metadata witness used random keys that never stop early.
   - The fast-path screening test aligned only one query head per group.
   Both are corrected and documented in the test files.
4. **Kaggle rate limit (HTTP 429)** when re-listing 11,600 dataset files. Handled by reusing sizes from the original listing call.

## 5. Results

*(tables generated by `tests/g5r_analyze.py` → `results/g5r/summary_tables.md`; to be inserted)*

## 6. Adversarial and failure analysis

*(to be completed)*

## 7. What remains unverified

*(to be completed)*

## 8. Gate decision

*(to be completed)*
