# NEXUS — G5 Pythia capture integrity audit

**Version/date:** 1.0 / 2026-11-04  
**Authority:** `AGENTS.md` v1.0; **status:** G5 BLOCKED  
**Engineering review lenses:** A2 numerical analysis, A3 AI systems, A4 runtime, A8 verification, A10 performance. No independent specialist review performed.

## Decision (first)

**Safety regression found and repaired in the diagnostic instrumentation.** The earlier Pythia pilot checked QKV-to-attention-context reconstruction at three positions (`0`, middle, last), yet collected oracle statistics for multiple *other* query positions. Corruption in those other positions could escape the integrity check. This is a software data-integrity flaw, **not** a processor-performance result.

**G5 is still BLOCKED.** No genuine pretrained checkpoint, full-model benchmark, competitive kernel comparison, GPU energy, or hardware execution was completed.

## Falsifiable experiment

- Source under test: `tests/g5_pythia_capture.py` from the original G5 harness ZIP, updated in the loose repository tree.
- Baseline integrity check: only `q = [0, T//2, T-1]`.
- Counterexample: seeded causal attention of length `T=65`, `H=2`, `D=4`; replace the attention distribution at queried position `q=T-2` with a unit spike at token 0 while leaving the recorded dense attention-context input unchanged.
- Before fix: maximum reconstruction error across original three checked positions: **0.0**.
- When tested at the corrupted query: maximum absolute context difference: **1.8470681306847547** (the precise value is fixture-specific; not a real-model error).
- New check validates **all oracle-evaluated query positions**, plus first/middle/last sentinels. It checks causal masking, finite nonnegative weights and row normalization, then verifies reconstructed V-weighted attention against pre-dense inputs.
- A deliberately corrupted one-layer GPT-NeoX-style **untrained synthetic integration fixture** now fails capture. The same consistent fixture passes. The fixture is not a pretrained network.
- A fixed 40-character checkpoint revision must match both the requested and the model config's resolved commit, or the script aborts. This is a Git revision identity check, **not** a cryptographic hash of all checkpoint tensor contents.

## Commands run

```bash
cd NEXUS
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q \
  tests/test_g5_pythia_capture_integrity.py \
  tests/test_g5_capture_shapes.py \
  tests/test_g5_trace_oracle.py
```

**Observed:** `19 passed in 1.93s`. Complete log: `results/G5_capture_hardening_pytest.txt`; machine-readable provenance: `results/G5_capture_hardening_audit.json`.

## Pretrained pilot remains unexecuted

The following is a **future execution command, not a result**:

```bash
python -m pip install torch numpy transformers safetensors pytest
python tests/g5_pythia_capture.py \
  --text-file /path/to/public_text.txt \
  --max-tokens 1024 --last-queries 16 \
  --local-files-only \
  --output results/g5_pythia14m_traces.json
```

`--local-files-only` requires that the pinned checkpoint and tokenizer have already been cached. Without it the script can attempt a normal public Hugging Face download. The current environment has `torch` and `pytest` but not `transformers`; the pinned checkpoint is not cached, and `huggingface.co` DNS resolution fails. No cloud GPU or paid job was launched.

The diagnostic uses full attention, QKV and V; page counts are *impractical oracles*. The pilot would not establish long-context 8K+ efficiency even if it ran. Need two unrelated real model families and fair optimized baselines for G5, plus end-to-end quality, traffic, latency and energy where measurable.

## Unresolved risks

1. No genuine pretrained-model capture executed; current Transformers ABI integration not tested.
2. Exact checkpoint-file SHA-256 values and tokenizer file hashes not yet recorded; configuration `_commit_hash` alone is insufficient for byte-level provenance.
3. The `0.005` absolute reconstruction threshold is a pilot diagnostic rather than a proved safe finite-precision tolerance.
4. All-head full-attention capture is expensive; no decode/KV runtime overhead measured.
5. This fixes measurement integrity, not the previously identified lack of novel hardware mechanism.

**Gate status:** G3 capture-instrumentation regression PASS on synthetic fixtures; **G5 BLOCKED**; G6+ NOT AUTHORIZED; breakthrough NOT DEMONSTRATED.
