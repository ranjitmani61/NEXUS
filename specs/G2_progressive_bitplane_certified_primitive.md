# G2 feasibility: certified progressive bitplane reading (reference, not invention)

**Version/date:** 1.0 / 2026-11-03  
**Gate:** G2 (early concept); G1 novelty is FAILED for generic mechanism  
**Engineering lenses:** A1 architecture, A2 integer mathematics, A3 model evaluation, A4 runtime, A6 memory, A8 verification, A10 systems, A11 prior art. No independent specialist review.  
**Objective:** See whether a classifier can read less than all eight weight bitplanes **and still guarantee the same argmax as full INT8 computation**. Not an LLM attention or silicon experiment.

## Reference and proposed primitive

- Baseline 1: exact dense dot product of all INT8 weights plus integer biases, returning argmax.
- Baseline 2: approximate four- and six-plane midpoint weight reconstruction at exactly the selected bit budget, reporting both test-set accuracy and agreement with full INT8.
- Candidate: store 8-bit weights as independently readable MSB-first packed bitplanes. On each query, unpack one additional plane, update partial outputs, then check a mathematically sufficient class-margin certificate. Only read further planes if necessary. All candidate score arithmetic uses 64-bit signed integers; the reference checks a conservative overflow precondition.
- Benchmark: sklearn **digits** (bundled 1,797 labelled 8x8 images), 70/30 stratified splits with seeds 17 and 29, 10-class logistic regression trained locally on each split, per-tensor symmetric int8 weights, integer-scaled biases and integer pixels. Classification accuracy vs held-out true labels is distinct from argmax agreement with the quantized INT8 reference.
- Negative controls: signed random activations and weights, identical class weights/biases (worst case; all planes needed), very large dominating output bias (best case; early termination), and randomized shape/edge cases.

## Exact-arithmetic certificate

For a weight entry `w in [-128,127]`, let `u=w+128`. After reading its leading `k` binary planes, let `p=floor(u/2^(8-k))`; with `r=8-k`, its unknown exact value is

```
w = -128 + 2^r*p + e,   e in [0, 2^r-1].
```

For integer activation vector `x`, class bias `b_j`, and `R=2^r-1`, define `a_j = (2^r*p_j-128) dot x+b_j`. Since unread residuals are independent but individually constrained,

```
L_j = a_j - R*sum_i max(-x_i,0)
U_j = a_j + R*sum_i max( x_i,0)
```

satisfies `L_j <= exact_logit_j <= U_j`. If a candidate class `c` has `L_c > U_j` for every `j != c`, it must be the exact argmax of the full INT8 classifier. If no class satisfies this sufficient test, read more planes. At `k=8`, all weights are known and the ordinary NumPy first-maximum tie rule applies. There is no approximation at the **quantized classifier** level. This certificate is conservative and may force more reads than necessary.

**Bounded-integer assumption:** The input-constrained reduction and bias fit in signed int64, enforced in the reference by a conservative precheck. `int64` is not an unlimited mathematical integer; unsafe hardware arithmetic, wrapping, and missing checks would invalidate the guarantee. Ordinary model quantization changes the FP64 reference logits and must be independently evaluated.

## Cost model and limitations

- `K*D` INT8 weight values occupy `K*D` bits per plane; when divisible by eight, a single packed plane is `K*D/8` bytes, and all eight planes occupy `K*D` bytes.
- Ideal **logical weight-bit read fraction** per inference is `mean(k)/8`; this is **not a measured DRAM/HBM byte count**. Actual cache-line granularity, banking, burst transfers, bitplane unpacking and reductions, interconnect energy, page coalescing, on-chip caching and initial layout conversion are unmodeled.
- In this reference, candidate Python scans and unpacking are much slower than a NumPy dense matrix-vector operation. This is a real CPU software measurement, but **not** an optimized hardware benchmark.
- Argmax certification is not a guarantee that an entire LLM hidden-state tensor, probability distribution, top-k sampler, or generation sequence is unchanged. This primitive applies only where the contracted output is an argmax decision.

## Prior art (not exhaustive)

1. Precision Batching (2020), Harvard, bitserial decomposition with runtime-selectable precision: https://vlsiarch.eecs.harvard.edu/publications/precision-batching-bitserial-decomposition-efficient-neural-network-inference
2. ProgressiveNN (2021), MSB-first accumulative computation: https://www.jstage.jst.go.jp/article/ijnc/11/2/11_338/_article/-char/en
3. BitSET (2023), bit-serial early termination: https://lacuna.tiptreesystems.com/work/bitset-bit-serial-early-termination-for-computation-reduction-in-convolutional/wrk_9eb0a0a7e0b818e24fa415e5a9ff4272
4. Neural Network Acceleration Using Digit-Plane Computation with Early Termination (2024): https://scholar.nycu.edu.tw/en/publications/neural-network-acceleration-using-digit-plane-computation-with-ea/
5. BitFair (2026), adaptive bit order and early termination: https://arxiv.org/abs/2607.05445
6. SATIN (2026), progressively refined scores with explicit L1 bounds for unread bitplanes (especially close mathematically): https://doi.org/10.1007/s44443-026-01313-1

**Novelty judgement:** The generic incremental bitplane dataflow and residual certificate are strongly overlapped by existing published ideas. This work is **not a new invention claim**, and no strong differentiation has been established.

## Reproduction

```
python tests/g2_progressive_bitplane_audit.py
python -m pytest -q tests/test_g2_progressive_bitplane.py
```

Outputs in `results/g2_progressive_bitplane.json`, `results/g2_progressive_bitplane_stdout.txt`, `results/g2_progressive_bitplane_pytest.txt`.