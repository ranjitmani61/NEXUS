# G5-R defect-localisation diagnostics (2026-11-04)

These scripts were run to investigate anomalies. They are kept because they are the evidence for the defect findings in `results/g5r/invalid/README.md`. Outputs are in `results/g5r/diag/`.

The scripts hard-code the scratchpad paths of the session that ran them. Edit `S=` to reproduce elsewhere.

| Script | Question | Finding |
|---|---|---|
| `diag_granite_3340.py` | Is Granite book-3340 perplexity (290) real? | Full forward without cache: 8.32. Cached decode: 298. |
| `diag_decode_paths.py` | Does cached decode match the full forward? | Qwen fp32: yes (1.7e-5 logits). Granite bf16: no (KL 4.05), for both M=1 decode and M=2 chunks. |
| `diag_granite_sensitivity.py` | Is BF16 Granite numerically chaotic? Is the g5r wrapper to blame? | Full forward bit-identical across lengths. Stock `sdpa` shows the same gap, so the wrapper is not the cause. |
| `diag_granite_probe.py` | Where do the paths diverge? | Layer-0 prefix keys differ by 15.9% while values are identical, so RoPE is implicated. Root cause: an empty hybrid cache reports length = batch size. |
| `diag_dense_baseline.py` | Is the dense timing baseline fair for GQA? | Granite got a 4D mask, so the wrapper used `repeat_kv`: 37.1 ms against 7.9 ms native GQA per layer at 8K. Qwen got mask=None (fair). |
