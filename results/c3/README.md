# results/c3 — Cycle 3, G0: batch-1 decode bottleneck on a Kaggle T4 GPU (2026-11-10)

**Charter:** `specs/C3_G0_gpu_decode_bottleneck_charter.md`, with amendments A1 (after the smoke runs, before any main data) and A2 (a post-data defect fix after main run 1, before run 2). The hashes are in `preregistration.json`.
**Gate review:** `reviews/2026-11-10_C3_G0_gpu_bottleneck_review.md`.

## Environment actually used

- **Hardware:** Kaggle "GPU T4 x2": 2× Tesla T4 (sm_75, 15 GB, 70 W), driver 580.178.04, CUDA 13.0, 4 vCPU, 31 GB RAM. The docker image digest is in each `c3_run_record.json`.
- **Software:** torch 2.11.0+cu130 as shipped; transformers **4.57.1** and bitsandbytes 0.50.2, both pip-installed in the session.
- **Inputs:** Kaggle Models `qwen-lm/qwen2.5/transformers/0.5b/1` and `ibm-research/granite-4.0/transformers/granite-4.0-micro-base/1`, and dataset `tunguz/the-pg19-language-modeling-benchmark-dataset`. Every file used passed a SHA-256 check against `results/g5r/provenance_*.json` inside the session (P1).
- **Device use:** timing on cuda:0 only. cuda:1 only holds half of the fp32 Granite quality reference.

## Files

| Path | What | Label |
|---|---|---|
| `preregistration.json` | Charter and A1/A2 hashes, code hashes, UTC times, quota at registration | record |
| `probe/c3_probe_partial_inventory.json` | Feasibility probe that stopped at its 900 s cap (my hashing-filter bug); environment inventory only | not a result |
| `smoke/` | Smoke runs 1 and 2 (T = 512/1024) and the kernels as pushed | **not results** |
| `main_run1/` | Main run 1, kept unchanged. **Graph-mode timings invalid** (C3-D4); many OOMs; not used for the decision (see its README) | [Measured], superseded |
| `main_run2/c3_main_*.json`, `c3_run_record.json`, `log_*.txt` | **The decision run** | [Measured] time and quality; [Derived] bytes |
| `main_run2/decision.json`, `summary_tables.md` | Pre-registered rule (`tests/c3_analyze.py` v1.2; KV bytes recomputed, C3-D5) | [Derived] |
| `main_run2/sensitivity_exploratory.json` | Roofline projection, attention efficiency, run-1 vs run-2 eager reproducibility, clock snapshots | exploratory |
| `microbench/c3_attn_microbench.json` | Attention-only SDPA variants on the same T4 | exploratory (not pre-registered) |
| `model_configs/*/config.json` | The two model configs, SHA-256-equal to the G5-R manifests; used to recompute KV bytes | record |
| `run_record/kernel_main*.py` | The exact main kernels pushed (code embedded in base64; SHA-256 recorded) | record |

Note on `main_run2/c3_main_qwen05.json`: its stored `bytes_derived` has `kv_read = 0`, because of defect C3-D5. The analyzer recomputes the bytes from `model_configs/` and keeps the recorded values under `bytes_derived_as_recorded` in its output.

## Reproduce

```bash
# local logic tests (synthetic; no hardware claim)
PYTHONPATH=tests python -m pytest -q -p no:cacheprovider tests/
# build and push the main kernel (Kaggle API via tests/c3_kaggle.py; credentials are injected by the session proxy)
python tests/c3_build_kernel.py --plan main --output kernel_main.py
python -I tests/c3_kaggle.py push kernel_main.py --title nexus-c3-main2 --gpu --internet --timeout 3000 \
  --model qwen-lm/qwen2.5/transformers/0.5b/1 ibm-research/granite-4.0/transformers/granite-4.0-micro-base/1 \
  --dataset tunguz/the-pg19-language-modeling-benchmark-dataset
python -I tests/c3_kaggle.py output <user>/nexus-c3-main2 --dest results/c3/main_run2
# decision and exploratory checks
python -I tests/c3_analyze.py --dir results/c3/main_run2
python -I tests/c3_sensitivity.py
# attention micro-benchmark (no inputs needed)
python -I tests/c3_kaggle.py push tests/c3_attn_microbench.py --title nexus-c3-attn-microbench --gpu --timeout 600
```

## Labels

- Time: **[Measured]** wall clock with device synchronization, on this T4 only.
- Bytes: **[Derived]** from configuration and storage.
- Copy and triad bandwidth: **[Measured]**.
- nvidia-smi power values are snapshots, **not** energy.
- No silicon, RTL or energy results exist in this directory.
