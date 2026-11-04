# 2026-11-03 G5 Readiness Audit — Engineering Gate Review

**Version:** 1.0
**Owner/review lenses:** A1/A2/A3/A4/A6/A8/A10/A11 (AI-assisted checks only; no independent specialist sign-off)
**Governance:** Read `/mnt/data/AGENTS.md` before execution. Rules, goal and gate criteria unchanged.

## Exact work completed

1. Inspected `AGENTS.md`, prior project log, previous G4 mathematics, earlier tests, and the G0 benchmark contract before designing this test.
2. Verified public model card of `EleutherAI/pythia-14m`, confirmed model is pretrained with 2,048 token context (not sufficiently long for G5).
3. Checked local environment: PyTorch 2.10 CPU, NumPy 2.3.5, no Transformers installation, no cached weights, local external HTTPS resolution fails. Web fetch of model file cannot provision binary weights here. Hugging Face Jobs are charged by usage; did not start a job. Public pricing source: https://huggingface.co/docs/hub/en/jobs-pricing
4. Implemented exact-probability attention mass oracle, exact-score value-ball certificate, and greedy hindsight comparison for a single [probabilities, V] head/query pair, `tests/g5_trace_oracle.py`.
5. Implemented a future pretrained GPT-NeoX/Pythia runner with fusion-layout and pre-projection context reconstruction checks, `tests/g5_pythia_capture.py`. It has **not** run against pretrained weights.
6. Added `tests/test_g5_trace_oracle.py` and `tests/test_g5_capture_shapes.py`; ran both locally using available Python/Torch. **11 test functions passed** (including 250 fractional-box brute-force cases and 120 randomized adversarial shapes; number of tests functions is not number of individual property trials).
7. Ran `g5_trace_oracle.py --smoke` on **five artificial** fixtures; exported `results/g5_trace_pipeline_smoke.json`. In the fixtures, median mass oracle page fraction was 1.0, median exact-score value page fraction 0.78125, min value page fraction 0.03125, max 1.0. No pretrained trace tested. These values are not achievable page skips without paying for full attention data.

## Disconfirming points, model and pipeline limitations

- The probability vector for each query already requires the full QK attention calculation. The value-aware certificate inputs also read V. Consequently, **none of the oracle page-count reductions demonstrate traffic savings**.
- Baselines are idealized and not hardware comparable; no inference time, throughput, energy or full-model quality measured.
- Current toy data and Pythia-14M pilot context cannot satisfy the >=8K target.
- The GPT-NeoX extraction path was only unit-tested on mock fused projection/attention tensors; it may fail when Transformers APIs change. If context reconstruction fails, the runtime aborts, rather than silently accepting incorrect V traces.
- The pilot input text will need a clearly licensed, pinned, representative dataset rather than an arbitrary snippet.
- Numeric float64 tests are not proofs of BF16/FP16 hardware guarantees.
- Model card changed in February 2026 for Pythia 14M; the audit pins a specific current revision and avoids mixing deduplicated and standard Pile variants.

## Evidence/outputs

- `tests/g5_trace_oracle.py`, `tests/g5_pythia_capture.py` (source)
- `tests/test_g5_trace_oracle.py`, `tests/test_g5_capture_shapes.py` (verification)
- `results/g5_trace_pipeline_smoke.json` (machine-readable synthetic results)
- `results/g5_harness_environment.json` (environment/availability)
- `specs/G5_pretrained_trace_protocol.md` (contract)
- `reviews/2026-11-03_G5_harness_readiness_review.md` (this gate decision)

## Gate decision

**G5 BLOCKED — no real pretrained evaluation performed.** The pipeline passed only mathematical/shape smoke tests. No chip or genuinely novel computing principle established. Next authorized step requires pretrained-model access and representative data before a claimed G5 comparison. Do not spend money or begin G6 hardware design without renewed documented evidence/authorization.