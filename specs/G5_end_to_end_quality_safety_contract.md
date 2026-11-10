# G5 end-to-end quality safety contract — v1.0
**Date:** 2026-11-03  
**Project:** Project 01 AI processor research  
**Governing standard:** `/mnt/data/AGENTS.md` (read before work)  
**Owner:** assistant; review lenses A2 (numerical), A3 (ML), A4 (runtime), A8 (verification), A10 (performance)  
**Status:** Proposed benchmark requirement and checked linear-only counterexample; NOT a pretrained-model evaluation.

## Failure investigated

The existing G5 projection-delta diagnostic (`tests/g5_projection_delta_capture.py`) marks a threshold safe if **each linear projection's local relative L2 output error** is <= 0.02. That metric **cannot imply** bounded final-model logits, token quality or the same argmax. Relative errors can be magnified by downstream maps; the denominator for a final output can be much smaller than the magnitude of upstream activations. Errors can accumulate across layers/decode steps.

## Constructive falsification (fully specified)

Let previous input `x0=(0,0)`, current `x1=(1,0.015)`, projection `W=I₂`, threshold `tau=0.02`. A thresholded temporal-delta update discards `0.015` and returns `y*=(1,0)` instead of `y=(1,0.015)`. The local relative L2 error is about **1.4998%**, under the local **2%** criterion. Let downstream classification head be `H=diag(0.001,1)`. The true logits are `(0.001,0.015)` (class 1) while the approximate logits are `(0.001,0)` (class 0), with about **99.7785% final relative L2 error**. These are deliberately constructed values, not trained weights.

The implementation's existing `evaluate_projection_deltas()` **accepts** this example. A new regression test asserts both that behavior and the different final argmax. Thus the current local-only rule is falsified as a *sufficient* end-to-end safety criterion; it may still serve as an exploratory screening statistic.

## Valid mathematical guard for a restricted linear chain

For an `L`-stage linear chain, `h_i=W_i h_{i-1}` and approximate `h*_i=W_i h*_{i-1}+e_i`, with trusted absolute local error bounds `||e_i||₂ <= epsilon_i`, triangle inequality and submultiplicativity give:

`B_0=0; B_i=||W_i||₂ B_{i-1}+epsilon_i` and `||h*_L-h_L||₂ <= B_L`.

For final dense logits `z`, with top-two gap `gamma=z_(1)-z_(2)`, the true top-1 class is guaranteed unchanged if `gamma > 2 B_L`; this follows because each logit perturbation is bounded in absolute value by `B_L`. **Failure of this test means UNKNOWN, not necessarily incorrect.** The certificate requires a defensible bound and a trusted exact-margin (or separately certified margin lower bound) — computing dense logits first is only an oracle diagnostic, not acceleration.

This derivation does **not** cover nonlinear activations, residual additions, LayerNorm/RMSNorm, attention softmax, rotary positional operations, quantized arithmetic, feedback, or changes in the token sequence. Their effects require explicit Lipschitz/interval/rounding bounds and a full algorithmic quality test; do not reuse this restricted theorem as an LLM hardware guarantee.

## Mandatory G5 experimental acceptance criteria (before any hardware claim)

1. **Actual pinned pretrained checkpoints**, tokenizer, input datasets and reproducibility identifiers. Synthetic/mock results do not satisfy G5.
2. **Matched-quality end-to-end outputs**, including prefill and cached autoregressive decode. Report per-token logit error, top-1 agreement, sequence divergence, cross-entropy/perplexity on representative data, and downstream task quality as relevant. Track both typical and worst cases.
3. **Error accumulation checks** across depth and across autoregressive steps, including adversarial near-tie logits, long contexts, numerical precision and boundary inputs.
4. **Competitive baselines and full costs**: dense optimized kernels, relevant known delta/sparse methods, input sparsity detection, metadata writes, cache consistency, reset/bootstrap, fallback, compilation, memory movement, tail latency, throughput, energy if actually measured.
5. **No proxy-to-claim promotion**: local L2 checks, oracle skipped coordinates/pages, memory estimates and output reconstruction must be labeled as diagnostic only, never as safe inference or chip speedup.

The project has not defined an evidence-backed final-model allowable quality-loss budget for each target task; that must be pre-registered before data-driven tuning. A 2% local projection criterion is *not* such a budget.

## Reproduce

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_g5_end_to_end_error_contract.py \
  tests/test_g5_projection_delta_capture.py tests/test_g5_capture_shapes.py \
  tests/test_g5_trace_oracle.py tests/test_g2_temporal_delta_gemv.py \
  tests/test_g2_delta_transformer_activation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tests/g5_end_to_end_error_contract.py --cases 500 \
  --output results/g5_local_to_global_error_audit.json
```

**Evidence:** `results/g5_local_to_global_error_audit.json`, `results/g5_local_to_global_regression_pytest.txt`. The 500 seeded randomized linear-chain property checks are not formal proofs; the derivation above is restricted to real arithmetic.

## Gate decision

G5 `BLOCKED` (no genuine pretrained weights or competitive runtime benchmarks). Local 2%-error-as-global-safety hypothesis `FAIL`. Correctness instrumentation `PASS` for the constructed counterexample and restricted linear-chain bound. G6–G12 not authorized. No processor novelty or breakthrough established.
