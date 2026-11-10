# G2 — FFN weight-traffic feasibility screen (2026-11-10)

**Decision: FAIL for postactivation SwiGLU down-projection-only skipping as a 2× mechanism.** G1 novelty not shown, no new G5 inference, G6+ unauthorized. This is a dated analytic screen, not hardware performance.

**Baseline:** Claude G5-R real-model CPU observations at immutable commit [8510d899](https://github.com/ranjitmani61/NEXUS/blob/8510d89980816e06a511ab0d665188cbe7da3d6c/results/g5r/summary.json), currently on an unmerged research branch. We used these secondary measurements, **not** newly acquired pretrained-model weights.

## Shape-only weight-I/O calculation
For a tied-embedding SwiGLU transformer of L layers, hidden d, inner width m, query heads hq, KV heads hkv, vocabulary V and head width dh=d/hq:

- FFN gate/up = 2Ldm weights; FFN down = Ldm.
- Attention projections = L(2d²+2d·hkv·dh).
- Tied LM head/embedding = Vd stored weights counted **once**.
- Counted total = sum. Omit biases/norm, KV traffic and all runtime overhead. Assume every counted weight read once/token (an unmeasured streaming abstraction).

**Even skipping 100% of down weights at zero cost** yields the following optimistic maximum:

| Model | Counted weights | Down share | Maximum weight-stream speedup |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 493,961,216 | 21.17% | **1.269×** |
| Granite-4.0-micro-base | 3,402,629,120 | 24.65% | **1.327×** |

The gate/up matrices have already been read and SwiGLU neurons are generally nonzero. The hypothetical 100% skip is **not** a feasible claim of exact accuracy.

## Independent time-based obstruction
Let f be the measured fraction of decode time spent in attention, a the fraction of attention time saved and w the fraction of all remaining time saved. By Amdahl, under zero overhead, speedup = 1/(1−f·a−(1−f)·w). With attention untouched (a=0), removing **all** nonattention computation is bounded by 1/f.

| G5-R CPU case | Attention fraction f | Perfect nonattention-elimination ceiling |
|---|---:|---:|
| Qwen batch 4, 32K tokens | 0.58612 | **1.706×** |
| Granite batch 1, 16K tokens | 0.70521 | **1.418×** |

These conditional ceilings remain below 2× even if f is reduced by a hypothetical 0.05 absolute. Qwen batch 1 at 32K is an explicit exception: f≈0.494 means perfect elimination of everything nonattention could mathematically reach 2.023×, but no overhead is allowed.

With OR-SCORE's **non-executable hindsight** score oracle, a hypothetical *joint* optimizer must still eliminate 53.75% (Qwen batch 4/32K) or 50.55% (Granite batch 1/16K) of the **nonattention runtime** to reach 2×. OR-SCORE is per-query-head; actual GQA shared KV pages are the *union* of heads, so treating it as real saved bytes would exaggerate savings.

**Review lenses:** A1/A2/A3/A4/A6/A8/A10/A11. Local full regression package passed 68/68 automated checks; initial expected algebra value in one test was corrected (0→1/3), not a model change. The full local test implementation and machine-readable outputs are available in the originating ChatGPT package; this GitHub branch includes a minimal stdlib replay only, not the full 68-test suite. No independent human review or new power/silicon results.

**Prior art:** [DejaVu ICML 2023](https://proceedings.mlr.press/v202/liu23am.html) predicts conditional MLP sparsity ahead of weight loading; [PyTorch 2025 sparse inference](https://pytorch.org/blog/beyond-quantization-bringing-sparse-inference-to-pytorch/) documents activation long tails; [Sparsing Law ICML 2025](https://proceedings.mlr.press/v267/luo25i.html), [Turbo Sparse](https://arxiv.org/abs/2406.05955), and [vLLM EAGLE3](https://docs.vllm.ai/projects/speculators/en/stable/user_guide/algorithms/eagle3/) block generic novelty claims for FFN gating or speculative decoding.

**Next gate:** A genuinely distinct joint attention/weight/sequence mechanism must first satisfy f·a+(1−f)·w≥0.5 with fully counted work, then distinguish itself from DejaVu, PowerInfer and speculative verification before any hardware development. No seventh processor is demonstrated.
