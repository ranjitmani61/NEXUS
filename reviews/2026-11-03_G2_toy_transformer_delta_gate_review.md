# G2 gate review — trained-toy transformer activation-change density
Date: 2026-11-03. Decision: **NO-GO as a new chip mechanism**; hold temporal-delta branch pending pretrained-model evidence. Engineering lenses: A1/A2/A3/A4/A6/A8/A10/A11, no outside specialist review.

## Prior evidence being tested
The earlier temporal-delta GEMV microbenchmark showed CPU speedups when **the fraction of changed coordinates was artificially controlled** to 1–20%. This experiment asks whether a trained causal model produces such changes naturally. Existing delta networks/Chipmunk already cover the general technique; novelty is not claimed.

## Actual experiment
Two independent initializations/training seeds (20261103 and 20261104), each trained on mixtures of repeat/cycle/alternate toy next-token tasks (320 CPU optimizer steps), plus matched **untrained** control. Evaluated 64 sequences per generator and four generators (repeat/cycle/alternate/random). Each 24-position sequence gives 23 consecutive-token updates. Model: two causal transformer layers, width 48, four heads, vocabulary 32. No public pretrained checkpoint. Measured only the **final output-head projection**. Full-sequence and prefix-by-prefix causal execution agree within 2.1e-6 absolute logit difference. Exact FP64 cached-head reconstruction agrees with dense within 1e-12 relative L2 in every evaluated trace.

## Results, trained toy, cross-token activation deltas
| Pattern | Baseline next-token accuracy | Exactly changed dimensions | Smallest fraction updated under stringent diagnostic | Optimistic *head weight traffic only* speedup cap |
| --- | ---: | ---: | ---: | ---: |
| Repeated token | 96.5–97.5% | 100% | 97.30–97.44% | 1.026–1.028× |
| Modulo increment | 97.1–97.4% | 100% | 99.37% | ~1.006× |
| Alternating | 96.5–97.0% | 100% | 99.39–99.47% | 1.005–1.006× |
| Random control | 2.9–3.5% | 100% | 99.33–99.42% | 1.006–1.007× |

Diagnostic requires **maximum normalized logit L2 error ≤0.02 and top-1 agreement ≥0.999**. A threshold τ=0.01 was best among the preselected sweep values in all four patterns. Such best-possible weight-read reduction assumes zero sparsity-detection/gather overhead; actual end-to-end latency is not measured. At τ=0.2 the repeat pattern selected ~48% of coordinates but permitted maximum relative logit error 0.316–0.355, even though top-1 labels remained unchanged; good top-1 agreement alone cannot certify numerical fidelity. The random control is deliberately unlearnable and should not be interpreted as a successful AI workload.

## Verification
- 25/25 targeted automated tests passed (13 new trace tests + 12 existing GEMV tests).
- Two independent full training/evaluation runs succeeded and produced stored JSON and stdout.
- The script asserts exact recurrence agreement and prefix-consistency; held-out same-generator patterns avoid reusing evaluation arrays but do not constitute real language data.
- One numeric failure mode is cumulative output error after aggressive skipping, and results explicitly report the maximum across sequences and time. No experimental evidence of GPU speedup, BF16 numerical safety, trained LLM quality, energy, compiler mapping, or novel architecture.

## Interpretation
1. In these toy runs, coordinate-level **exact** temporal sparsity is absent; it is incorrect to infer cross-token sparsity from within-token ReLU zeros or from diffusion-step similarity.
2. The controlled original GEMV benchmark's 1–20% changed-coordinate regime does not arise at this final hidden projection in these tests without large approximation error.
3. The optimistic 1/f weight-byte bound applies only to the output head and neglects all overhead; it cannot support any architecture-level speedup claim.
4. Toy model size, patterns, position embeddings, layer choice and sequence length limit generalization. This neither proves nor disproves sparsity in genuine pretrained LLMs.
5. Temporal delta is already in prior work (Delta Networks 2017, Chipmunk 2025). This branch must not be called an invention.

## Gate dispositions
- G1 novelty: FAIL for ordinary temporal-delta execution.
- G2 suitability as autoregressive exact-delta candidate: FAIL **in tested toy regimes**; approximate version offers insufficient *strict-error* head-level work reduction here.
- G3/G4: scoped reference arithmetic/trace experiments complete; no floating-point hardware guarantee.
- G5: BLOCKED until genuine pretrained-model traces and fair optimized baseline comparisons are available.
- G6–G12: NOT AUTHORIZED / NOT DEMONSTRATED.

## Next discriminating evidence
Use the existing `tests/g5_pythia_capture.py` harness or an equivalent to acquire **actual Pythia-14M or larger pretrained** layer activation traces, capture exact/thresholded consecutive-token deltas per layer (including Q/K/V and MLP projections) and benchmark against dense/sparse production kernels. Until that occurs, close this direction as a hardware candidate and do not shift to another invention label without new mechanism/prior-art evidence.

## Reproduction artifacts
- `tests/g2_delta_transformer_activation_audit.py`
- `tests/test_g2_delta_transformer_activation.py`
- `specs/G2_temporal_delta_activation_trace_protocol.md`
- `results/g2_trained_toy_delta_activation_seed20261103.json`
- `results/g2_trained_toy_delta_activation_seed20261104.json`
- `results/g2_trained_toy_delta_seed20261103_stdout.txt`
- `results/g2_trained_toy_delta_seed20261104_stdout.txt`
- `results/g2_delta_next_gate_pytest.txt`
- `results/g2_delta_toy_aggregate.json`

Prior art: https://proceedings.mlr.press/v70/neil17a.html ; https://arxiv.org/abs/2506.03275 ; https://research.google/pubs/on-emergence-of-activation-sparsity-in-trained-transformers/