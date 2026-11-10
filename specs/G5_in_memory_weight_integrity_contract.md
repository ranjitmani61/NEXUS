# G5 in-memory checkpoint consistency contract — v1.3

**Date:** 2026-11-05. **Gate:** G5 BLOCKED (supporting verification only).
**Engineering ACTs:** A3 AI/ML, A4 runtime, A8 verification. No independent human review.

## Objective

Close a gap in the Pythia pilot: local snapshot SHA-256 metadata alone does not demonstrate that the actual runtime model tensors equal the staged `model.safetensors` data. A mismatched or modified in-memory weight can produce a trace with perfectly consistent-looking checkpoint hashes.

## Invariant and method

Immediately **before** and **after** the diagnostic forward pass, `verify_loaded_model_weights(model, staged_dir)` shall:

1. Open the staged `model.safetensors` using `safetensors.safe_open` on CPU; reject absent, symlinked, invalid, or zero-tensor files.
2. For **every** safetensors tensor, match a same-name `model.state_dict()` tensor of identical shape. Reject non-finite tensors, missing keys, and value changes.
3. Allow only *lossless* floating-point promotions (`float16/bfloat16 → float32/float64`, `float32 → float64`) or identical dtypes. Reject unrecorded precision reductions or cross-half conversions.
4. Require every runtime `named_parameters(remove_duplicate=False)` parameter to be checkpoint-backed. A tied parameter may omit its own on-disk key only if it has the **same storage address, shape, dtype, and current values** as a verified parameter.
5. Return a numerical audit summary, and include the before/after audit status in pilot metadata. This is an invariant of the current process at two sampling times, **not continuous monitoring**.
6. Reject traces advertising `version=1.3` that lack the self-declared in-memory audit; older `v1.2` traces remain parseable for historical reproducibility. A JSON-only validator **cannot authenticate** those self-declarations.

This does not verify tokenizer in-memory behavior, forward hooks, all nonparameter buffers, hardware, model publisher identity, or any content on the pretrained checkpoint unless the real model is executed. It does not detect a transient mutation that is reversed between the two audits.

## Reproducibility

Requirements: Python 3.11+, torch, safetensors, pytest, numpy. Run from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_g5_checkpoint_snapshot.py \
  tests/test_g5_result_validator.py \
  tests/test_g5_capture_shapes.py \
  tests/test_g5_pythia_capture_integrity.py \
  tests/test_g5_trace_oracle.py \
  tests/test_g5_loaded_tensor_integrity.py
```

Existing genuine-model command remains documented in `tests/g5_pythia_capture.py`; no pretrained-model run was possible in the current sandbox.

## Gate criteria and decision

**Reference software integrity:** PASS on synthetic fixtures only. **G5 pretrained model quality/throughput:** BLOCKED. **G6 silicon design:** not authorized. Any successful record says `PILOT ONLY — NOT G5 PASS`.

Independent pretrained source: [EleutherAI Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m), pinned revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`; no local weights obtained.
