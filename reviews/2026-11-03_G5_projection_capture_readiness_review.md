# G5 gate review: token-to-token projection capture integrity
**Date:** 2026-11-03. **Decision:** Instrumentation tests PASS; **G5 actual pretrained benchmark remains BLOCKED**.

## What was attempted

Read `AGENTS.md`, latest G2 trained-toy review and code. Probed internet access and model locations. No `EleutherAI/pythia-14m` pretrained weights exist locally; Transformers not installed; `huggingface.co` DNS resolution fails. Did not run billed cloud jobs or pretend to have obtained genuine model traces.

## New artifact

Created `tests/g5_projection_delta_capture.py` for pinned pretrained GPT-NeoX capture (future use) plus `tests/test_g5_projection_delta_capture.py` tests. It instruments the input to all four `nn.Linear` projections per layer and the final LM head; validates shapes, finite values, model architecture and prior-prefix causal consistency; evaluates thresholded linear delta identities in FP64. A failure closes hooks cleanly. All oracle metrics require fully computed dense activations and parameters.

## Verified test evidence

- **56 / 56** targeted G2+G5 tests passed (`results/g5_delta_capture_pytest.txt`). No pretrained weights involved.
- **6 seeded mock fixture runs** (3 seeds × GELU vs ReLU) recorded in `results/g5_delta_instrumentation_mock.json`.
- In GELU mocks, 100% of input coordinates changed across observed consecutive tokens. ReLU mocks showed roughly 84–87% changed because ReLU introduced exact zeros. This is a **negative-control demonstration**: an instrument that uses ReLU mocks cannot infer GELU-model sparsity. It says nothing about actual Pythia activations.
- With threshold 0.01 on these particular mocks, selection averaged about 96% of delta coordinates under GELU, with maximum local projection-relative error across all six fixtures <=0.017; this is *not* an end-to-end quality guarantee or speedup.
- Valid causal-prefix fixture had no detected mismatch; an intentionally noncausal global-context fixture was rejected. This does not guarantee numerical behavior in a real Transformer install, nor validate incremental decode/streaming KV.
- Earlier failures in assumptions about activation-zero density were corrected by matching the mock's default nonlinearity to Pythia's pinned `hidden_act: gelu`. Earlier test failures are retained in the work record; no silent change to pass an unjustified test.

## Critical limitations

Mock is 2 layers, width 8, single-head toy causal attention, random weights; NOT a pretrained model. It tests hooks and recurrence only. The existing 2026-11-03 activation-delta toy model was trained on algorithmic tasks; neither model establishes LLM activation statistics. Since model weights and Transformers are unavailable locally, the new pretrained capture command has not been executed and must be integration-tested before any real-data inference.

Even if local output-error controls pass, they cannot be generalized through GELU, LayerNorm, residual connections, attention/position effects, and multi-layer recurrence. Sparse update discovery, threshold decisions, matrix reading and gather/scatter cost must be measured; the method is not novel.

## Next gate, with no implied authorization for paid actions

1. Obtain pinned pretrained checkpoint + tokenizer and a non-private input sample in a capable environment (no charges incurred here).
2. Run projection-capture integrity checks and compare token positions under true incremental decode using KV cache; check outputs against full-sequence reference, across distinct models and contexts.
3. Benchmark strong dense/sparse kernels with matched quality and all metadata/preprocessing overhead. No G6 hardware authorization until G5 passes and G1 novelty concerns are resolved.

**Sources:** https://huggingface.co/EleutherAI/pythia-14m/blob/cf967c0a9a04383db6f7b1108d86b2962634b4ac/config.json ; https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py

**Provenance:** Code/test/mock JSON/local-resource-probe/pytest-log in package. No external engineer review.