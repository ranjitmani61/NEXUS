# results/g5r — G5-R real-model study (2026-11-04)

Contract (pre-registered, with dated amendments A1/A2): `specs/G5R_real_model_benchmark_contract.md`.
Pre-registration hashes: `preregistration.json`.
Gate report: `reviews/2026-11-04_G5R_real_model_gate_review.md`.

## Environment actually used

- Cloud VM: 4 vCPU Intel Xeon (family 6, model 207; AVX-512, AMX-BF16), 15 GiB RAM, no GPU, no RAPL.
- Python 3.13.18, PyTorch 2.15.0 (PyPI wheel, MKL + oneDNN 3.13.3; CUDA not used), transformers 4.57.1, NumPy 2.5.4, sigstore 4.5.1.
- `huggingface.co` was blocked by the network policy, so weights and data come from Kaggle.

## Reproduce

All commands are run from the repository root, with `V=<venv>/bin/python`, `M=<model dir>`, `B=<pg19 dir>/test/test`.

```bash
# 0. fetch (official Kaggle publishers; writes provenance manifests with SHA-256)
$V tests/g5r_kaggle_fetch.py model qwen-lm qwen2.5 0.5b 1 --out $M/qwen2.5-0.5b --manifest results/g5r/provenance_qwen2.5-0.5b.json
$V tests/g5r_kaggle_fetch.py model ibm-research granite-4.0 granite-4.0-micro-base 1 --out $M/granite-4.0-micro-base --manifest results/g5r/provenance_granite-4.0-micro-base.json
$V tests/g5r_kaggle_fetch.py dataset tunguz the-pg19-language-modeling-benchmark-dataset \
   --files $(python -c "import json;print(' '.join(json.load(open('results/g5r/pg19_selection.json'))['selected']))") \
   --out <pg19 dir> --manifest results/g5r/provenance_pg19_selected.json --listed-sizes-json results/g5r/pg19_selection.json

# 1. unit/property tests (synthetic; math and timed-path correctness only)
PYTHONPATH=tests $V -m pytest -q -p no:cacheprovider tests/test_g5r_core.py tests/test_g5r_fast_attn.py

# 2. model validation (pretrained vs random-init control, capture validity)
$V tests/g5r_validate_models.py --model-dir $M/qwen2.5-0.5b --model-name qwen2.5-0.5b --dtype fp32 --books $B/*.txt --output results/g5r/validate_qwen2.5-0.5b.json

# 3. traces + live quality (one context per call); see queue order in the review
$V tests/g5r_run_quality.py --model-dir $M/qwen2.5-0.5b --model-name qwen05 --dtype fp32 --context 4096 --steps 32 \
   --books $B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt $B/3608.txt $B/3754.txt $B/4047.txt $B/5396.txt \
   --raw-dir <raw dir> --output results/g5r/quality_qwen05_T4096.json

# 4. timing (exclusive CPU)
$V tests/g5r_timing.py --model-dir $M/qwen2.5-0.5b --model-name qwen05 --dtype bf16 --contexts 4096 8192 16384 32000 \
   --methods dense quest:1024 quest:2048 quest:4096 nx_box nx_sph nx_one --steps 16 --reps 5 --books $B/860.txt \
   --output results/g5r/timing_qwen05_bf16_b1.json

# 5. passkey, independent check, adversarial metadata, aggregation
$V tests/g5r_passkey.py --model-dir $M/qwen2.5-0.5b --model-name qwen05 --dtype fp32 --context 8192 --books $B/*.txt --output results/g5r/passkey_qwen05_T8192.json
$V tests/g5r_independent_check.py <raw dir>/*_samples.npz --output results/g5r/independent_check.json
$V tests/g5r_adversarial_meta.py <raw dir>/*_samples.npz --output results/g5r/adversarial_metadata.json
$V tests/g5r_analyze.py
```

The exact queue used is reproduced in the gate review. Raw per-record traces and captured q/K/V samples (hundreds of MB) are **not** committed. They are regenerated deterministically by steps 3 and 5.

## Labels

- `quality_*`, `validate_*`, `passkey_*`: **[Measured]** model outputs on CPU.
- `timing_*`: **[Measured]** wall-clock on this VM only.
- Oracle columns: **[Derived/diagnostic]**; they need full attention and are not executable savings.
- No GPU, energy or silicon numbers exist in this directory.
