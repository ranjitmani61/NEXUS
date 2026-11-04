# INVALID results — kept only as evidence of a harness defect (2026-11-04)

These files were produced before `tests/g5r_cache.py` existed. Do **not** use any number in them.

**Defect:** IBM Granite 4.0 (`GraniteMoeHybrid` in transformers 4.57.1) returns **no** KV cache when it is called with `use_cache=True` and `past_key_values=None`. It logs a warning, which the run log filter hid. Every "decode" step therefore attended only to the current token:
- every shadow record has `T = 1`;
- the dense decode perplexity is 2,967;
- the float64 reconstruction is "exactly 0.0", because softmax over a single token returns its value vector unchanged.

**What remains valid in `validate_granite4micro_v1_NO_CACHE.json`:** the full-sequence perplexity check, which uses no cache (PG-19 perplexity 8.5–15.9 against 116,416 for random init), and the bit-identical dense-against-SDPA logits. Only its shadow-reconstruction fields are invalid.

**Fix:** `tests/g5r_cache.py` builds `HybridMambaAttentionDynamicCache` explicitly. All drivers now assert the cache length after prefill and the attention length at every decode step, so a wrong length is a hard error.

**Qwen2.5-0.5B is unaffected:** its decode records show `T = context + step + 1`, and its timing `kv_bytes` equal the analytic full-length values.

## Second Granite defect: off-by-batch-size RoPE positions (found 2026-11-04, after the first fix)

`quality_granite4micro_T8192_v2_ROPE_OFFSET.json` is **INVALID**.

**Defect:** transformers 4.57.1 `HybridMambaAttentionDynamicCache.get_seq_length()` returns `batch_size`, not 0, for an empty cache. Its placeholder tensors have shape (batch, 0), and the method reads `shape[-2]`. A prefill into a fresh cache therefore starts its positions at `batch_size`, and every cached key carries RoPE positions shifted relative to later queries.

**How it was found:**
1. Book 3340's decode perplexity was 290, against 15.3 for Qwen-0.5B on the same text.
2. A cache-free full forward gave 8.3, while cached decode gave 298.
3. Both stock `sdpa` and the `g5r` wrapper reproduced the gap, so the wrapper was not the cause.
4. The full forward was bit-identical across sequence lengths, so the model is not numerically chaotic in BF16.
5. A per-layer probe showed layer-0 prefix **keys** differing by 15.9% while **values** and the new token's q/k were bit-identical. That pattern points to RoPE.
6. `get_seq_length()` on an empty cache returned 1 at batch 1 and 4 at batch 4.

**Fix:** `tests/g5r_cache.py::_fixed_hybrid_cache_class`. A new standing check, `tests/g5r_decode_consistency.py`, requires cached decode to match the full forward (KL ≤ 1e-3, argmax agreement 1.0) before any decode result is used.

**Scope:** this affected every Granite decode run before the fix. Qwen is unaffected: its cached decode matches the full forward to 1.7e-5 in logits.

## Handicapped Granite dense baseline in timing (found 2026-11-04)

`timing_granite4micro_bf16_b1_v1_HANDICAPPED_DENSE.json` is **INVALID as a baseline comparison**.

**What happened:** under the custom attention name `g5r_fast`, Granite's legacy mask builder passed a 4D mask at every decode step. The builder skips the mask only for the literal name `"sdpa"`. The transformers SDPA wrapper then took its `repeat_kv` path, copying K/V 5× for GQA, plus a masked kernel.

**Measured cost:** at 8K, the attention op took 37.1 ms per layer, against 7.9 ms with native GQA (`diag_dense_baseline`).

**What it distorted:** Granite's dense attention share (0.727 / 0.824), its Amdahl ceilings (3.66× / 5.69×), and Quest's apparent speedups (2.13× / 2.78×). NX-BOX still lost (0.43× / 0.24×).

**Fix:** `g5r_fast_attn._dense` v1.1 calls torch SDPA with `enable_gqa=True` when the decode mask is a no-op. Unit tests check equality with the wrapper and the fallback when a mask is real. Timing JSON now records `dense_native_gqa_calls`.

**Qwen is unaffected:** its decode mask was `None` in every call (24/24 at batch 4; verified for batch 1 as well), so the wrapper already used native GQA.
