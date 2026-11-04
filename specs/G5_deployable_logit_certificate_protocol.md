# G3/G5 Diagnostic — Deployable Logit-Interval Safety Check

**Version:** 1.0 — 2026-11-03  
**Governing rules:** `AGENTS.md` (read before execution); 20 truth and verification commitments  
**Engineering lenses:** A2 mathematics, A3 AI/ML, A4 runtime, A8 verification, A10 performance, A11 prior art  
**Scope:** untrained, 6-layer random residual MLP, 16 classes; **not** a pretrained transformer, long-context language-model evaluation, finite-precision hardware proof, or novel processor.

## Rationale and decisive question

Last G5 review invalidated the assumption that a <=2% **local** projection error preserves **global** output. It proposed a spectral error bound and a guard based on **exact** logits. Exact logits are not accessible to a runtime that tries to avoid the exact computation. This work tests a certificate requiring **only approximate logits**, fixed pretrained coefficients (the weights here are randomly initialized), and **honest local absolute error bounds**.

## Derivation (exact arithmetic)

For residual layer l:

```
x_(l+1)       = x_l  + phi(W_l x_l  + b_l)
x~_(l+1)      = x~_l + phi(W_l x~_l + b_l + e_l)
|e_l|         <= eps_l * ones                  (coordinate-wise)
B_0           = 0
B_(l+1)       = (I + L_phi*abs(W_l)) B_l + L_phi*eps_l*ones
B_logits      = abs(W_head) B_L
```

Here phi is ReLU (L=1) or the **exact** GELU `x*Phi(x)` (L=1.13, conservative because the largest absolute derivative is about 1.128904). All arrays have fixed widths; weights, biases and exact input are shared by the exact and approximate networks. Noise is injected before each activation and has a *known* componentwise absolute bound, not an estimated average.

For constant weights, precompute one class-specific vector of coefficients per layer, `C[:,l]`, by propagating a unit eps_l through all later layers, then the bound per class online is `B_logits = C @ eps`. If `t = argmax(approx_logits)` and

```
approx_logits[t] - B_logits[t] > max_{j != t}(approx_logits[j] + B_logits[j])
```

then **the true argmax must be t**, under the above mathematical assumptions. If it fails, the result is **UNKNOWN**, not a known misprediction. The selection rule never reads the exact logits.

### What this DOES NOT cover

- LayerNorm/RMSNorm, self-attention, RoPE, QK bilinear products, softmax, quantized weight errors, output-head error, KV coherency, varying tokenization, GPU kernels, or formal IEEE rounding guarantees.
- Computation of trustworthy local eps_l: in a practical accelerator this could be costly and cannot be assumed free. The synthetic experiment *injects* bounded noise and knows eps_l in advance.
- The certificate establishes top-1 *stability versus exact model*, not task-level accuracy, likelihood, perplexity, generation quality, training performance, or speedup.
- The precompute/online benchmark is Python/NumPy wall clock with validation overhead and not a reliable estimate of GPU/ASIC efficiency.

## Prespecified exploratory matrix

- Seeds: 20261103, 20261104, 20261105
- Widths: 16, 32; fixed depth 6; output classes 16
- Weight gains: 0.05, 0.1, 0.25 (scaled as gain/sqrt(width))
- Activations: exact GELU, ReLU
- Per-layer injected coordinate error bound: 0, 1e-4, 1e-3, 1e-2
- 64 independent inputs per combination; random noise drawn uniformly in [-eps,eps]
- **9216 checks** (3*2*3*2*4*64), plus unit tests
- Positive/negative tests: exact data, random bounded perturbations, adversarial false certificate when a layer's error budget is **omitted**, invalid error bounds rejected, and numerical derivative sweep for GELU.

## Baseline and metrics

Compare each approximate prediction against the same network run with zero injected error. Report: fraction certified, actual argmax flips, error-bound violations, false safety admissions, and inspection cost. Any zero-error result is a trivial baseline, not evidence of robustness. CPU timings are diagnostic only; no claim of inference speedups.

## Reproducibility

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_g5_deployable_logit_certificate.py \
  tests/test_g5_end_to_end_error_contract.py \
  tests/test_g5_projection_delta_capture.py \
  tests/test_g5_trace_oracle.py tests/test_g5_capture_shapes.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g5_deployable_logit_certificate.py \
  --samples 64 --output results/g5_deployable_logit_certificate.json
```

A second independent run uses `--output results/g5_deployable_logit_certificate_repeat.json`; counts and error-bound diagnostics must match, while wall-clock durations need not.

## Prior art — no novelty assertion

- Gowal et al., *Scalable Verified Training for Provably Robust Image Classification*, ICCV 2019: interval bound propagation (IBP). https://openaccess.thecvf.com/content_ICCV_2019/html/Gowal_Scalable_Verified_Training_for_Provably_Robust_Image_Classification_ICCV_2019_paper.html
- Zhang et al., *Efficient Neural Network Robustness Certification with General Activation Functions*, NeurIPS 2018: CROWN. https://papers.nips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html
- Pythia-14M model card: https://huggingface.co/EleutherAI/pythia-14m; publicly available, **not downloaded/executed** here. The model card documents a February 2026 change to the undeduplicated vs deduplicated repository, which reinforces the need to pin revisions and hashes.

## Decision rule

`PASS` for a narrowly specified *reference mathematical guard* only if no adversarial unit tests or measured error bounds contradict the guarantee. G3 overall remains `IN PROGRESS` until full-model and numerical requirements are addressed; G5 `BLOCKED` until genuinely pretrained model weights, complete model-quality tests and matched implementation baselines are available. G6 silicon still not authorized.
