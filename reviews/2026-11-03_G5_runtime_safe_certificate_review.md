# Gate review — G3/G5 runtime-compatible logit safety certificate

**Date:** 2026-11-03  
**Rules:** `AGENTS.md` v1.0 was read before work; heightened verification applied.  
**Lenses:** A2 math, A3 ML, A4 runtime, A8 verification, A10 systems, A11 prior art. **No independent human or silicon review.**

## Decision

**Narrow certificate arithmetic: CONDITIONAL PASS** for exact arithmetic in fixed-weight, six-layer residual MLPs with honest coordinatewise preactivation noise bounds. **G3 overall: IN PROGRESS; G5 pretrained validation: BLOCKED; G6 hardware: NOT AUTHORIZED. Novelty/breakthrough: NOT ESTABLISHED.**

The previous verification rule used exact logits, which an approximate implementation would normally not have. The new guard uses only approximate logits, known per-layer error budgets and precomputed, fixed-weight sensitivity coefficients. It certifies identical top-1 to the exact model when the approximate top-1 lower logit bound exceeds all other upper bounds. Abstentions are **unknown**, not errors.

## Executed verification

- Two independently executed seeded matrices, each **9,216** random residual-MLP input comparisons across ReLU/GELU, 2 widths, 3 gains, 3 seeds, and 4 noise budgets. Per-case certificate decisions and computed error diagnostics matched exactly on repetition; timing did not need to match.
- **Zero** observed coordinatewise bound violations and **zero** false top-1 certificates in both runs. These are tests, **not** a proof of rounding safety, full Transformers, or a statistical guarantee.
- **54/54 combined automated tests passed** (13 new tests + 41 prior G5 tests). Counterexample deliberately dropping a layer's budget produces a falsely reassuring certificate; correct budget accounting rejects it.
- No pretrained weights accessed. A free public file download was attempted separately and failed; no paid Hugging Face job launched, no model claimed executed.

## Positive and unfavorable regimes — first run

| Per-layer coordinate error limit | Exact-model comparisons | Certified unchanged | Actual class flips | Interpretation |
|---:|---:|---:|---:|---|
| 0 | 2,304 | 2,304 (100%) | 0 | Trivial zero-error control |
| 0.0001 | 2,304 | 2,203 (95.6%) | 0 | Tight, synthetic error |
| 0.001 | 2,304 | 1,566 (68.0%) | 2 | Mixed certification; some actual flips |
| 0.01 | 2,304 | 193 (8.4%) | 8 | Mostly abstains; poor utility |

Certification fractions depend heavily on weight scale and network structure: for GELU at 0.001 noise, 0.05 gain certified 87.2%, but 0.25 gain certified 31.8%. All models here were **randomly initialized**, not representative of real LLMs.

## Cost and implementation gaps

1. The local error budget `eps_l` is **assumed known and correct**; a deployable accelerator would have to compute/guarantee it, potentially eliminating any savings.
2. The offline `C` matrix is fixed-weight and can be reused for this narrow architecture; it still requires preprocessing and storage, is sensitive to weight updates, and may be quite loose in deeper/larger networks.
3. Measured Python certificate time is not comparable to a competitive accelerated dense or sparse attention kernel. A certificate adds runtime overhead; **no inference-speedup claim is justified**.
4. Model scope excludes LayerNorm/RMSNorm, attention, softmax, variable sequence length, mixed precision, KV-cache coherence, GPU numerical behavior, and all long-context issues.
5. This tests **argmax equivalence**, not accuracy, output distribution, sampling or autoregressive generation. A tied/close logit margin often yields an abstention.
6. Numerical checks in float64 are not a formal outward-rounded interval certificate; true finite-precision robustness remains unproved.
7. IBP (Gowal et al., ICCV 2019) and CROWN (Zhang et al., NeurIPS 2018) establish close prior art; a claimed invention would require a new mechanism and a fair advantage over modern verification systems.

## Artifact provenance

- Source: `tests/g5_deployable_logit_certificate.py`
- Unit tests: `tests/test_g5_deployable_logit_certificate.py`
- Contract: `specs/G5_deployable_logit_certificate_protocol.md`
- Raw first/repeated results: `results/g5_deployable_logit_certificate.json`, `results/g5_deployable_logit_certificate_repeat.json`
- Test log: `results/g5_certificate_combined_pytest.txt`
- Source/baseline review: `reviews/2026-11-03_G5_local_to_global_safety_review.md`

## Next justified test

Real-model G5 is **still blocked**. Stop expanding mock model harnesses as substitutes. Obtain an authorized, revision-pinned pretrained checkpoint, integrate correct full-model capture and end-to-end quality evaluation, then **measure quality, overhead, and accuracy** with strong production baselines. If resources remain inaccessible, pivot to a distinct architecture-level hypothesis rather than treating certificate research as a processor breakthrough.
