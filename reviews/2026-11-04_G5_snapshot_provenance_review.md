# G5 Provenance Hardening Review — 2026-11-04

**Engineering status:** G5 BLOCKED. This change strengthens the provenance chain; no pretrained model weights were run.

## Baseline failure and correction

The previous `tests/g5_pythia_capture.py` resolved the model and tokenizer in two independent `from_pretrained` calls and checked only the model config's `_commit_hash` against the requested revision. It did not record a content hash of the actual weight and tokenizer files used. The saved trace could therefore not be checked against the files that generated it. This is an evidence/provenance gap, not proof a prior result was falsified.

The updated script calls `snapshot_download` on the pinned commit with a safetensors/JSON allowlist, audits the returned snapshot path, stages the selected files together, records SHA-256 per file in `checkpoint_manifest`, and loads tokenizer and model from the same staged directory. It rehashes stage and source after the capture. The independent trace validator rejects absent/inconsistent manifests for version `1.2`.

## Verification performed

- Pinned model snapshot unavailable in local HF cache (`LocalEntryNotFoundError`); outgoing DNS `huggingface.co` unavailable.
- `78 passed in 2.71s` across the existing G5 suite and newly added adversarial snapshot tests.
- Synthetic tests cover mutation after staging, mutation of original snapshot, corrupted JSON, wrong revision/model type, symlink escape, valid HF-style blob symlink, broken optional symlink, missing safetensors, size cap, and manifest tamper.
- No actual pretrained model was loaded: fixtures contain invalid artificial safetensors bytes.

## Operational instructions

Run the commands in `specs/G5_snapshot_provenance_contract.md`; see captured evidence in `results/G5_snapshot_provenance_pytest.txt` and `results/G5_snapshot_provenance_audit.json`.

## Remaining risks and gate decision

- Captured local hashes cannot authenticate the publisher or prove real-model correctness; independent source hash verification is still required.
- Transformers integration, pinned Pythia architecture hooks, tokenizer behavior, and long-context model quality remain untested.
- The new capture flow is still a full dense-attention analysis with **no** speed/energy improvement measured.
- G3/G4 generally incomplete, G1 novelty unestablished, G5 **BLOCKED**, G6 **not authorized**, G12 **not demonstrated**.

**Engineering ACT lenses:** A3 model identity and tokenization, A4 runtime single-snapshot loading, A8 adversarial integrity verification, A11 source review. No human technical sign-off.
