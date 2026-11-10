# G4 Trained-Attention Control — Engineering Review

**Date:** 2026-11-03  
**Protocol:** `AGENTS.md` v1.0 read prior to execution; 20 truth rules apply.  
**Status:** G4 exploratory control **COMPLETED**, not a G4 overall pass; **G5 remains BLOCKED**.  
**ACT review lenses:** A1 architecture, A2 mathematics, A3 AI systems, A4 implementation, A6 memory, A8 verification, A10 performance, A11 prior art. These are analysis lenses, **not independent expert sign-off**.

## Decision first

The earlier untrained-transformer controls were unsuitable to infer what **learned** attention can do. A trainable single-head address-to-address softmax attention toy was therefore optimized from scratch on 128 known addresses. Training produced exceptionally concentrated weights. An idealized full-information oracle then selected **1 of 8 pages** on each trained query, but **8 of 8** with untrained weights. Injecting **2% uniform probability** increased the median mass-certificate requirement to **5 of 8 pages**; injecting 20% increased it to **8 of 8**. This is a **sensitivity diagnostic**, not a new processor mechanism or executable speedup.

## Workload and fairness

- 128 fixed address slots; each query specifies one of the same 128 addresses used in training. This is a small **trained lookup mechanism**, not a transformer language model, trained on English text, or a long-context benchmark.
- Single head, query/key dimension 24; 16-dimensional values, newly sampled independent Gaussian vectors at evaluation.
- Cross-entropy loss trains the query-to-key address mapping. Losses at 500 Adam steps: approximately 1.1–1.3e-6 over three seeds. This is *near-complete memorization* of the 128 address targets, not language generalization.
- Three independently seeded runs (21, 37, 53), **128 evaluation queries each** per regime, five regimes = **1,920 query/regime evaluations**. The same random generator yields fresh values per evaluation. No distinct unseen addresses are evaluated.
- 8 pages × 16 addresses each; error certificate tolerance 0.02 under the existing `g5_trace_oracle.py` normalized attention-output metric. Exact QK probabilities and V tensors were computed in full before oracle selection; there is no implementable page-skipping result.
- No external weights, GPU, paid remote job, language-model context, silicon, or physical measurement used.

## Aggregated observations (median of three seed-level medians)

| Regime | Mean target p (approx) | Median mass-oracle pages | Median value-oracle pages | Median relative retrieval-vector error |
|---|---:|---:|---:|---:|
| Untrained | 0.00784 | 8/8 | 8/8 | 0.999 |
| Trained | 0.9999987 | 1/8 | 1/8 | 0.00000047 |
| Trained + 2% uniform mixture | 0.980155 | 5/8 | 3/8 | 0.01994 |
| Trained + 20% mixture | 0.80156 | 8/8 | 8/8 | 0.1988 |
| Trained + 50% mixture | 0.50391 | 8/8 | 8/8 | 0.4991 |

The relative retrieval-vector error compares `p @ values` to the **intentionally addressed token's value**, a task-specific synthetic metric. It is **not** the normalized dense-vs-sparse output certificate and has no link to natural-language task quality. A 20% mixture makes retrieval itself inaccurate; comparison at matched task quality would reject that workload/model for retrieval rather than imply an algorithm failed.

The oracle mass certificate requires selected mass >= 1 - epsilon/2 = **0.99**. Consequently, a one-page mass-certificate result is possible only when that one page holds at least 99% of attention probability. This threshold is implied by the chosen certificate, **not a universal necessity for sparse-attention algorithms**. Value-aware bounds sometimes need fewer pages, but require expensive full attention/value information in this test.

## Test integrity, negative results, and limitations

1. `tests/g5_learned_control_audit.py` completed with 3 seeds and exported `results/g5_learned_control_audit.json`.
2. Full local regression suite: **13 pytest functions passed** (existing 11 plus 2 added tests). First run had **one test failure** because it unrealistically required float32 softmax normalization within 1e-9. The threshold was changed to 2e-7 and the suite passed; this was a **test precision bug**, not a certificate violation. Do not conceal the initial failed test.
3. The full 1,920-evaluation JSON result was reproduced a second time with **byte-identical output** (SHA-256 `c4658e9a46cd709807d8db8caf5e95eb0e3879b13cc88ef59b58db000929bbaf`).
4. The learner is tailored to precisely the same address vocabulary it was trained on. Its 99.9999% concentration is **not evidence** that normal pretrained LLM attention behaves so sharply. Treat it as an artificial upper-control.
5. Uniform probability mixing is an externally imposed intervention, not the output of a separately trained language model. The strong sensitivity is a falsification of naive robustness assumptions for this *certificate*, not a claim about natural attention distributions.
6. Oracle selection requires full probabilities and V values, while software, memory traffic, metadata, runtime, energy and task quality are not measured. **Never** report these page counts as speedups or KV-byte reductions.
7. A genuine pretrained Pythia-14M checkpoint remains inaccessible in the container; even if acquired it would be only a 2K diagnostic, not the two 8K+ family G5 exit criterion. The model card notes a corrected standard-vs-deduplicated Pile change dated Feb 27, 2026; pinned revisions must be verified against data provenance. Source: https://huggingface.co/EleutherAI/pythia-14m

## Reproduce

```bash
cd /mnt/data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g5_learned_control_audit.py --steps 500 --queries 128 --output results/g5_learned_control_audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_g5_trace_oracle.py tests/test_g5_capture_shapes.py tests/test_g5_learned_control.py
```

## Gate decision and next discriminating experiment

- **G4:** remains `IN PROGRESS` for any potentially useful architecture. This controlled trained-attention diagnostic is complete, but offers no implementable speedup.
- **G5:** `BLOCKED`. Requires externally accessible, pinned pretrained-model weights, model-specific verified QKV capture, diverse task data, multiple 8K+ model families, and matched-quality end-to-end baselines. No paid cloud work authorized.
- **G6–G12:** not authorized. **No breakthrough demonstrated.**
- Next authorized activity: obtain *genuine* pretrained attention/V traces, check reconstruction, measure the oracle gap and task quality, then either drop this line of work or propose an implementable mechanism subject to prior-art comparison.
