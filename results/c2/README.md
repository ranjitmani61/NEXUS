# results/c2 — Cycle 2, G0: batch-1 decode bottleneck re-selection (2026-11-10)

**Charter:** `specs/C2_G0_batch1_decode_bottleneck_charter.md`, pre-registered with dated amendments A1–A3. The hashes and times are in `preregistration.json`.
**Gate review:** `reviews/2026-11-10_C2_G0_bottleneck_review.md`.

## Environment actually used ("host C")

- Cloud VM: 4 vCPU Intel Xeon @ 2.10 GHz (family 6, model 207; AVX-512, AMX-BF16), kernel 6.18.44-fc-v338, 15 GiB RAM.
- No GPU, no RAPL energy counters, no DRAM counters.
- It is a different VM from G5-R hosts A and B. Nothing here is pooled with G5-R numbers.
- Software: Python 3.13.18, PyTorch 2.15.0 (CPU execution), transformers 4.57.1, torchao 0.18.0.
- Weights and data are the G5-R files (`results/g5r/provenance_*.json`, `results/g5r/pg19_selection.json`).

## Files

| File | What | Label |
|---|---|---|
| `preregistration.json` | Charter and A1–A3 hashes and UTC times | record |
| `decode_consistency_qwen05_fp32_T8192_hostC.json` | Cached decode vs full forward, Qwen fp32: passes (max KL ≤ 4.7e-12) | [Measured] |
| `decode_consistency_granite4micro_T8192_hostC.json` | Same for Granite bf16: argmax 1.0, but max KL 1.6e-3–5.1e-3 on 3 of 4 books, so it **fails the charter §3 precondition as written** (see review §3) | [Measured] |
| `diag_granite_trunc12_consistency_T8192.json` | (pending: running at commit time) Post-hoc: same check on genuine Granite layers 0–11 in fp32 and bf16 | [Measured], exploratory |
| `breakdown_qwen05_bf16.json`, `breakdown_granite4micro_bf16.json` | M1 time breakdown, M2 derived bytes, M3 triad, dense bf16 | [Measured] time; [Derived] bytes |
| `breakdown_b1_qwen05_fp32.json` | B1 (A3 rerun, sequential variants): dense fp32 plus three torch.ao dynamic-int8 (W8A8) variants; timing and quality | [Measured] |
| `breakdown_b1_qwen05_fp32_run1_oomkilled.json` | B1 run 1 (interleaved), OOM-killed at 32K; kept unchanged; no quality, no 32K | [Measured], partial |
| `breakdown_b1_qwen05_bf16.json`, `breakdown_b1_granite4micro_bf16.json` | B1 (A2): dense bf16 plus torchao int8 weight-only; the latter was slower than dense everywhere, so its quality was not evaluated (A2 item 3) | [Measured] |
| `b2_prompt_lookup_*_T8192.json` | B2 prompt-lookup decoding vs plain greedy; Granite refused by the library | [Measured] |
| `diag_lookup_divergence_qwen05_T8192.json` | v1.0 divergence diagnostic. Its reference used `topk` tie order (defect, review §6); kept unchanged | [Measured], exploratory |
| `diag_lookup_divergence_qwen05_T8192_v1.1.json` | (pending: queued at commit time) v1.1: argmax reference as in `generate`, exact ties recorded | [Measured], exploratory |
| `decision.json`, `summary_tables.md` | Output of the pre-registered rule (`tests/c2_analyze.py`) | [Derived] |
| `sensitivity_exploratory.json` | Run-choice pairings, Amdahl ceilings, derived weight-only-quantization byte shares, fp32 interleaved vs sequential, hook overhead, bandwidth vs triad | [Derived], exploratory |
| `run_record/` | The queue scripts actually executed, plus their logs | record |

## Reproduce

Run from the repository root, with `V=<venv>/bin/python`, `Q=<dir>/qwen2.5-0.5b`, `G=<dir>/granite-4.0-micro-base`, `B=<pg19 dir>/test/test` and `F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"`. Set `PYTHONDONTWRITEBYTECODE=1`.

```bash
# unit tests (synthetic instrumentation only; no model evidence)
PYTHONPATH=tests $V -m pytest -q -p no:cacheprovider tests/

# preconditions
$V tests/g5r_decode_consistency.py --model-dir $Q --model-name qwen05 --dtype fp32 --books $F4 --T 8192 --output results/c2/decode_consistency_qwen05_fp32_T8192_hostC.json
$V tests/g5r_decode_consistency.py --model-dir $G --model-name granite4micro --dtype bf16 --books $F4 --T 8192 --output results/c2/decode_consistency_granite4micro_T8192_hostC.json

# M1-M3 dense breakdown
$V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype bf16 --variants dense --contexts 4096 8192 16384 32000 --timing-book $B/860.txt --output results/c2/breakdown_qwen05_bf16.json
$V tests/c2_breakdown.py --model-dir $G --model-name granite4micro --dtype bf16 --variants dense --contexts 4096 8192 16384 --timing-book $B/860.txt --output results/c2/breakdown_granite4micro_bf16.json

# B1 family (A2/A3)
$V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype fp32 --variants dense int8dyn int8dyn_pc int8dyn_pc_nolm --contexts 4096 16384 32000 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --sequential-variants --output results/c2/breakdown_b1_qwen05_fp32.json
$V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype bf16 --variants dense int8wo --contexts 4096 16384 32000 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --output results/c2/breakdown_b1_qwen05_bf16.json
$V tests/c2_breakdown.py --model-dir $G --model-name granite4micro --dtype bf16 --variants dense int8wo --contexts 4096 8192 16384 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --sequential-variants --output results/c2/breakdown_b1_granite4micro_bf16.json

# B2 prompt lookup
$V tests/c2_prompt_lookup.py --model-dir $Q --model-name qwen05 --dtype bf16 --context 8192 --new-tokens 64 --reps 3 --book $B/860.txt --output results/c2/b2_prompt_lookup_qwen05_bf16_T8192.json
$V tests/c2_prompt_lookup.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --new-tokens 64 --reps 1 --book $B/860.txt --output results/c2/b2_prompt_lookup_granite4micro_bf16_T8192.json

# decision (pre-registered rule) and exploratory sensitivity
$V -I tests/c2_analyze.py
$V -I tests/c2_sensitivity.py

# exploratory diagnostics
$V tests/c2_lookup_divergence.py --model-dir $Q --book $B/860.txt --context 8192 --new-tokens 64 --output results/c2/diag_lookup_divergence_qwen05_T8192_v1.1.json
$V tests/c2_granite_trunc_consistency.py --model-dir $G --books $F4 --layers 12 --T 8192 --output results/c2/diag_granite_trunc12_consistency_T8192.json
```

## Notes on the run record

- `run_record/c2_queue.sh` is the first queue. Its `step()` logs `rc=$?` after a `$(date …)` substitution, so its exit codes are not reliable (G5-R defect D8). This hid the B1 v1.0 crash.
- `run_record/c2b_queue.sh`, `c2c_queue.sh` and `c2d_queue.sh` capture `rc` first. One step's rc=137 is the OOM kill recorded in Amendment A3.
- Every output was additionally checked for completeness: contexts, variants and quality entries present.
- `c2_queue_out.log` is the unfiltered stdout and stderr of all queues. It contains the transformers warning "attention mask is not set". At batch 1 with no padding this is harmless: plain `generate` equals the verified manual greedy loop in every B2 run (`plain_generate_consistent_with_verified_path: true`).
- The paths in the scripts refer to the session scratchpad. They are a record, not a portable script.

## Labels

- Time: **[Measured]** wall clock, on this VM only, CPU execution.
- Bytes per step: **[Derived]** from configuration and dtype. No counters exist.
- The triad is **[Measured]** host bandwidth.
- No GPU, energy or silicon numbers exist in this directory.
