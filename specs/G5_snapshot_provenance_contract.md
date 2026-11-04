# G5 Checkpoint Snapshot Provenance Contract — NEXUS Project 01

**Version:** 1.0, 2026-11-04  
**Gate:** G5 BLOCKED, provenance subsystem verified only with offline synthetic fixtures  
**Review lenses:** A3 AI/ML, A4 runtime, A8 verification, A11 source provenance; no external specialist review.

## Objective and gate

A G5 capture must use model weights, configuration and tokenizer from **one** pinned model revision and record byte-level SHA-256 hashes of every file actually used. Reject incompletely cached or changed inputs instead of silently falling back to another revision or to pickle-based model weights. This does **not** establish publisher authenticity, model training provenance, or real-model benchmark performance.

## Pinned source

- Repo ID: `EleutherAI/pythia-14m`
- Exact revision: `cf967c0a9a04383db6f7b1108d86b2962634b4ac`
- Required: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`.
- Optional: `special_tokens_map.json`, `generation_config.json`.
- Select from a Hugging Face `snapshot_download` revision path; **no** `pytorch_model.bin` or `.pt` deserialization; no `trust_remote_code`.
- Stage copies of only selected files in a temporary directory; load model and tokenizer from that one directory.
- Record SHA-256 and file byte counts; rehash after use; reject mismatches, missing files, unsafe symlinks, wrong model type, invalid JSON, and outsized files.
- G5 version `1.2` requires the embedded file inventory and fingerprint to pass a separate validator. The trace validator *cannot* read files merely from its JSON payload, so manifest arithmetic is not an independent provenance attestation.

## Local testing and limitations

```bash
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  tests/test_g5_checkpoint_snapshot.py \
  tests/test_g5_pythia_capture_integrity.py \
  tests/test_g5_result_validator.py \
  tests/test_g5_trace_oracle.py \
  tests/test_g5_capture_shapes.py
```

- **Observed:** 78/78 tests passed in offline local CPU environment. The new fixture uses a deliberately invalid `model.safetensors` byte string; it checks staging and integrity only, **not** model loading.
- **Observed:** Pinned snapshot is not in the local Hugging Face cache; local files-only call raises `LocalEntryNotFoundError`. External network DNS fails, so no pretrained model execution occurred.
- **Unverified:** The actual model can load under current Transformers; QKV hook orientation on a trained checkpoint; end-to-end output correctness; independent trusted publisher hashes; remote-code supply-chain guarantees; performance, energy, GPU or silicon results.

## Reproduction when external prerequisites are available

```bash
python -m pip install 'transformers' 'huggingface_hub' 'safetensors' 'torch'
python tests/g5_pythia_capture.py \
  --text-file path/to/approved_public_sample.txt \
  --max-tokens 1024 --last-queries 16 \
  --output results/g5_pythia14m_traces.json
python tests/g5_result_validator.py results/g5_pythia14m_traces.json
```

Pythia-14M is a diagnostic, not a representative long-context model; real G5 requires unrelated model families at 8K+ context under the preexisting benchmark contract. Do not treat this code as a paid-cloud-work authorization.

## Threat model and weaknesses

- Local hashes expose unintentional or detectable malicious file changes. They do not prove the weights were originally published by EleutherAI; to establish external authenticity, independently compare the recorded hashes with a trustworthy pinned repository source.
- A compromised user or process that can rewrite code, source files, and recorded manifests can fabricate self-consistent output.
- Bytes can still be altered by a higher-privilege attacker while being loaded; snapshots are staged, and reverified, but Python is not a secure enclave.
- CPU capture still uses full attention and oracle/hindsight counts, not real acceleration. The checkpoint does not unblock long-context inference.

## Sources consulted

- Pythia-14M model files: https://huggingface.co/EleutherAI/pythia-14m/tree/main (main was inspected; exact revision hash remains unverified independently).
- Hugging Face `snapshot_download` behavior: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
- NEXUS previous protocol: `specs/G5_pretrained_trace_protocol.md`.

**Decision:** PASS for **offline provenance-path validation only**. **G5 remains BLOCKED.** No novel hardware or architecture claims.
