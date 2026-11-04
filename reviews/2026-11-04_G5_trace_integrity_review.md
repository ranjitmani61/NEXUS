# NEXUS G5 — Recorded-trace integrity gate

**Version:** 1.0 · **Date:** 2026-11-04 · **Status:** G5 `BLOCKED`
**Governing discipline:** `AGENTS.md` v1.0. Review lenses A2 (numerical), A3 (AI systems), A4 (runtime), A8 (verification), A10 (performance). **No independent human review.**

## Decision

**PASS for one narrow G3 measurement-integrity subtest:** a new standalone fail-closed verifier is integrated into the Pythia-14M pilot output writer. **G5 remains BLOCKED.** This work is not a new processor, an algorithmic speedup, a pretrained-model test, or formal proof of certificate correctness.

## Falsification result: self-reported summary flag was unsafe to trust

`tests/g5_trace_oracle.py::summarize()` unconditionally emits `all_value_certificates_verified_numerically: true`. It calculates aggregate metrics but does not verify each row's error/certificate inequality. A deliberately corrupted oracle row with normalized error 0.8 can therefore receive that flag when `summarize()` is rerun. This is **not a claim that the original `evaluate_one()` returned the corrupted result**; it demonstrates that a subsequently edited or incomplete JSON artifact could fool downstream report readers who trust only the flag.

`tests/g5_result_validator.py` now independently rejects this fixture and validates a complete trace record before it is written by `tests/g5_pythia_capture.py`:

- pinned model and resolved revision, metadata, nonempty query/layer/head grid;
- missing, duplicate, misplaced, or contradictory per-head/per-query records;
- token prefix `q+1` and page-count/fraction arithmetic;
- nonnegative finite measurements, mass and value certificates, hindsight diagnostic, and tolerance bounds;
- explicit `oracle` caveat and constrained guarantee wording;
- all measured query positions covered by each layer reconstruction check;
- independently recalculated min/median/max summaries and count;
- duplicate JSON fields, `NaN`, `Infinity`, malformed JSON and unpaired Unicode surrogate rejection by independent CLI.

The G5 runner calls `validate(result)` before saving and writes JSON with `allow_nan=False`. This is **structural and arithmetic consistency**, not proof of original attention probabilities, source model weights, trace truthfulness, or floating-point hardware safety. Attackers who coordinate all fabricated numbers can still produce an internally consistent but false record.

## Reproducible execution

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=tests \
 python -m pytest -q \
 tests/test_g5_result_validator.py \
 tests/test_g5_pythia_capture_integrity.py \
 tests/test_g5_trace_oracle.py \
 tests/test_g5_capture_shapes.py
# 58 passed in 1.70s in this workspace

# After obtaining a genuine, pinned pretrained result file:
python tests/g5_result_validator.py results/g5_pythia14m_traces.json
```

Recorded output: `results/G5_trace_validator_pytest.txt`; detailed machine-readable audit: `results/G5_trace_validator_audit.json`. Adversarial inputs are generated within tests, with no outside data or trained checkpoint. Tested the existing pilot integration fixtures together with the new validator.

## Current unavoidable evidence gap

- No pretrained weights cached; Hugging Face DNS lookup failed in this workspace and Transformers package was absent. No billed remote compute launched.
- No source-model weight-file checksums or independent model forward pass verified.
- Pythia-14M causal prefill oracle is not true autoregressive cached decode; not representative of 8K+ context.
- No optimized kernel baseline, model-output task quality, time-to-token, memory traffic, power, GPU or silicon evidence.
- G1 novel-mechanism case remains unsatisfied; do not authorize hardware based on this work.

**Next discriminator:** obtain verified pinned pretrained weights and tokenizer files, record their SHA-256 checksums, execute controlled prefill+decode trace validation and full-quality comparisons. If no checkpoint is accessible, G5 remains blocked rather than repeatedly labeling synthetic tests a breakthrough.
