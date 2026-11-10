# Project 01 — G3 reduced precision gate review

**Date:** 2026-11-03 · **Review version:** 1.0 · **ACT lenses:** A2/A5/A8 lead; A1/A3/A10 constraints. **Independent external review:** none.

## Decision

**G3 IN PROGRESS.** Reject a hardware datapath that computes page-score certificates with naive serial FP16 or BF16 dot-product accumulations. This is a valid safety counterexample for that hypothetical datapath, NOT proof that the current FP64 software has a bug. Research into a novel processor remains unsubstantiated.

## Test results

| Workload | Naive claimed normalized error bound | Actual normalized error | Guarded behavior |
|---|---:|---:|---|
| Synthetic FP16 cancellation | 0.00619843 | 1.93929782 | Dense fallback |
| Synthetic BF16 cancellation | 0.00619843 | 0.47926312 | Dense fallback |

Verification: 9/9 pytest checks passed. Of 2,000 randomized dot-bound checks, 1,998 were evaluated, two declined; no observed failures. Of 1,000 additional two-page attention tests, 499 were guarded admissions, with zero detected certificate violations. Repeat of prior deterministic witness evaluation was identical. These are finite **empirical** checks, not a universal correctness proof.

### Source of failure

FP16 and BF16 serial summation can erase a middle term in a cancellation sequence; exact attention on the stored values still assigns the omitted key substantial probability. The naive metadata upper bound is thus wrong. A classical conservative gamma error envelope makes the selected page no longer certifiable in the two examples. The envelope itself has domain restrictions and may be loose enough to erase any advantage.

## Prior art and external evidence

Conservative floating-point `gamma_n` analysis is established numerical analysis, not new architecture: Higham & Mary, *A New Approach to Probabilistic Rounding Error Analysis* (2019), https://doi.org/10.1137/18M1226312 . The Quest 2024 approach already uses page summaries for query-aware attention, https://proceedings.mlr.press/v235/tang24l.html .

**G5 model provenance warning:** the Pythia-14M model card now states that prior to 2026-02-27 a different (deduplicated) model had occupied its name, https://huggingface.co/EleutherAI/pythia-14m . Before running a pinned checkpoint in `tests/g5_pythia_capture.py`, confirm the pinned revision corresponds to the intended model variant. This change was verified from the model card, not from downloaded weights. Existing 14M models are diagnostic only; the G5 charter calls for stronger families and 8K+ contexts.

## Reproducibility and artifacts

- `specs/G3_reduced_precision_certificate_protocol.md`: assumptions, equations, negative case, limitations, commands.
- `tests/g3_reduced_precision_certificate_audit.py`: executable witnesses and property tests.
- `results/g3_reduced_precision_audit.json`: machine-readable measured numbers.
- `results/g3_reduced_precision_pytest.txt`: 9 passing tests.
- `results/g3_reduced_precision_stdout.txt`: summarized run output.

## Remaining unproved and next smallest useful test

Full certified attention with finite-precision min/max metadata, online radius updates, softmax, value accumulation, and concurrent decode is NOT proven. A next step is to define a specific mixed-precision datapath with outward rounding or provable guard and compare fallback rate and latency against dense, on representative pretrained queries when available. G5 is **BLOCKED** by no accessible pretrained checkpoint in this workspace (local DNS resolution and direct download failed); no paid cloud job was run. No chip, no RTL, no claimed breakthrough.
