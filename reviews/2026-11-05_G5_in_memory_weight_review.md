# G5 loaded-weight verification review — 2026-11-05

**Decision: G5 BLOCKED; instrumentation partially verified.** All statements labeled by evidence type.

## Verified environment and blocker

- Read complete `AGENTS.md`, current project log, the G5 snapshot, capture, validation code and previous review.
- Locally available: PyTorch and safetensors; transformers **not installed**; no cached genuine Pythia model files found.
- Attempted DNS access to `huggingface.co`, GitHub, and `raw.githubusercontent.com` failed (name resolution). No paid job or remote execution was attempted.

## Vulnerability and correction

[Derived] Prior code hashed model bytes on disk but did not compare every loaded `state_dict` parameter against the expected weights. A model with altered in-memory parameters could pass that provenance audit. That does **not** prove any previous real-model data are invalid: no real-model trace was ever captured in this project.

[Implemented] Added `verify_loaded_model_weights`; called before and after the diagnostic inference in the Pythia pilot, and upgraded optional artifact schema from 1.2 to 1.3. Exact tensor equality is enforced after safe promotion, including tied-weight verification and adversarial aliases. Added fail-closed tests for 1.3 metadata.

[Measured software tests] The combined existing/new G5 suite passed **91/91** on synthetic tensors and artificial traces. A mismatched in-memory tensor and a non-identical tied alias were correctly rejected. PyTorch models in these tests are not pretrained LLMs.

## Risks not closed

- The real Pythia checkpoint and installed Transformers runtime remain unavailable locally: **zero** genuine pretrained inference evaluations.
- Verifying local file and model equality does not prove third-party publisher authenticity or that the model is the exact intended training run.
- Full prefill oracle traces cannot establish a runtime sparse-attention implementation, end-to-end quality, speed, energy, or processor novelty.
- The verification only checks model parameters and known staged tensors, not all mutable nonparameter buffers, unusual dynamic reparameterizations, or transient changes that revert between snapshots.
- The current Pythia pilot context (<=2048) is not representative of long-context target systems.

**ACTs used:** A3 model semantics, A4 runtime, A8 adversarial tests, A11 public model information. A5–A7 / A9–A12 not reviewed; no independent human sign-off.

**Next hard gate:** Obtain independently identifiable pretrained files and execute the pilot on real weights, then test end-to-end task quality versus optimized dense and published competitors. Do not advance hardware work before G5 evidence.
