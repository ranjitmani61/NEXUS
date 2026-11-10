# G5 Pretrained Attention Trace Protocol — Project 01

**Version/date:** 1.0, 2026-11-03  
**Status:** G5 BLOCKED for representative real-model evaluation; pilot capture and offline verification code prepared.  
**Review lenses:** A1 architecture, A2 numerical mathematics, A3 AI/ML systems, A4 compiler/runtime, A6 memory, A8 verification, A10 performance, A11 prior art. No independent engineering review.

## Decision and falsifiable objective

The previous value-aware page-screening hypothesis failed to skip pages for the **untrained** transformer experiment. It must not be promoted to a novel processor. The next experiment must inspect **real pretrained** attention and V distributions to determine whether (a) the same failure persists, (b) an impossible exact-information oracle has headroom, and (c) any future computable selector could plausibly justify its metadata and arithmetic overhead.

**Important:** This package **does not execute G5** on pretrained weights in this workspace. The workspace has PyTorch and NumPy, but no installed Transformers package, locally cached pretrained weights, GPU, or externally accessible model downloads. Hugging Face remote Jobs are pay-per-use; no remote paid job has been authorized or launched.

## Model ladder (not interchangeable evidence)

1. **Pythia-14M, diagnostic pilot only**: `EleutherAI/pythia-14m`; pinned revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`; GPT-NeoX; 6 layers, 4 heads, model width 128, max 2,048 tokens. It tests source integration, capture correctness, and expected attention behavior on a genuinely trained (but very small) model. **It cannot validate long-context 8K+ efficiency.** Source: https://huggingface.co/EleutherAI/pythia-14m and https://github.com/EleutherAI/pythia
2. **Long-context model A (required for representative G5):** `HuggingFaceTB/SmolLM2-135M`, trained Llama architecture with grouped-query attention, claimed context 8,192 tokens. Requires a *different, separately verified QKV capture adapter*. Source: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
3. **Long-context model B (required for representative G5):** `Qwen/Qwen2.5-0.5B` or instruction-tuned variant, Qwen2 architecture, grouped-query attention, 32,768 maximum context. Requires a separate capture adapter and model-quality measurement. Source: https://qwen.ai/blog?id=qwen2.5-llm and https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

No claims about accuracy or performance transfer across these families from the diagnostic pilot.

## Source verification and capture integrity

- Pin full commit ID, framework/runtime versions, tokenizer, precision, data hash, and evaluation code revision.
- Set `model.eval()`, no dropout; use eager attention to access attention matrices **for analysis only**.
- For GPT-NeoX only, hook the `attention.query_key_value` fused projection and `attention.dense` input. Recover V by splitting fused QKV into `[T, heads, 3*head_dim]` and selecting the last third. Reconstruct actual pre-projection attention context `sum softmax(QK) V` and compare to `dense` input. **Abort** if absolute reconstruction error >0.005 or shapes mismatch. The threshold is a pilot sanity check, not a rigorous low-precision validation tolerance.
- Extract probabilities for selected causal prefix positions, compare oracle bounds from `g5_trace_oracle.py` against the full-attention weighted V output. Store every head/layer/query row and all numerical checks.
- The output is computed from **full attention weights and values**; all page-count metrics are hindsight or oracle diagnostics, **not executable accelerator savings**.

## Oracle measurement contract

For probability vector `p_i>=0`, `sum p_i=1`, value vector `v_i`, and selected tokens `S`:

`y_full=sum_i p_i v_i`; `y_S=(sum_{i in S} p_i v_i)/(sum_{i in S} p_i)`.

Declared error: `||y_S-y_full||_2 / max_i ||v_i||_2`, tolerance **0.02** (arbitrary exploratory metric; **not** end-user model-quality guarantee).

Compare three **infeasible** tests:

- **Exact mass oracle**: page probabilities known; sort by exact mass and include enough mass that `2*(omitted mass)<=0.02`.
- **Exact page-score + value ball oracle**: page score maxima exactly known; each unselected page has bound `U_j = n_j * max_i p_i`, and per-page mean+max-radius summaries. Stop when the exact-real fractional-box upper bound on the omitted contribution is <=0.02 after normalizing by the maximum value norm. Numerically recheck actual output error.
- **Greedy hindsight**: uses the full dense output and exact page probabilities to stop at the first prefix satisfying observed error. This is **not a certified bound**, **not a globally optimal subset**, and **not realizable without full computation**.

For every result report exact page count, fraction of pages, true numerical error and certificate; **never multiply page fraction by physical-speedup estimates**. Costs of QK score computation, V reading, metadata construction, SRAM/DRAM traffic, cache misses, runtime selection, sorting, and compiler integration are unknown. These idealized counts are *not* G5 success metrics.

## Pre-registered coverage requirements for eventual G5

- >=2 unrelated pretrained model families at supported 8K+ contexts; model-specific, validated capture path; available, pinned datasets.
- At least document understanding, multi-hop/needle retrieval, diffuse/flat and adversarial text, dissimilar/repetitive prompts, and early/late decoding.
- Page size 16/32/64/128, head dimension variations, query regimes, KV length and attention-sink behavior, perturbations, and numeric precision stress tests.
- Matched quality vs dense and strongest currently relevant attention kernels with pre-processing, runtime and metadata charged; end-to-end p50/p95, throughput, memory movement by tier, energy, and task quality. A model-output L2 error alone cannot substitute for model task accuracy.
- No hardware implementation until workload, benefit, correctness and prior art justify G6.

## Commands

Offline mathematical/shape checks (run successfully in this workspace):

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_g5_trace_oracle.py tests/test_g5_capture_shapes.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g5_trace_oracle.py --smoke --output results/g5_trace_pipeline_smoke.json
```

To execute the **unrun** pretrained Pythia diagnostic in a separately configured environment with weights access and installed `transformers`:

```bash
python -m pip install 'transformers>=4.40' torch safetensors
python tests/g5_pythia_capture.py --text-file /path/to/public_text.txt --max-tokens 1024 --last-queries 16 --output results/g5_pythia14m_traces.json
```

The latest Transformers APIs may evolve; this capture is not yet integration-tested against a live pretrained checkpoint. It must be verified before using its output as evidence. Do not pass confidential document text to remote jobs without authorization. **Do not run paid cloud compute without approval.**

## Gate status

- G0 exploratory charter: PASS (existing contract).
- G1 prior-art review: IN PROGRESS; novelty not proven.
- G3/G4 synthetic numerical testing: IN PROGRESS; no end-to-end measurements.
- **G5: BLOCKED** (model weights, installed Transformers, long-context representative data and competitive baselines not available). The harness readiness work does not count as a G5 pass.
- G6–G12: NOT AUTHORIZED by current evidence; breakthrough not demonstrated.
