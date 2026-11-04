# INVALID results — kept only as evidence of a harness defect (2026-11-04)

These files were produced before `tests/g5r_cache.py` existed. Do **not** use any number in them.

**Defect:** IBM Granite 4.0 (`GraniteMoeHybrid` in transformers 4.57.1) returns **no** KV cache when it is called with `use_cache=True` and `past_key_values=None`. It logs a warning, which the run log filter hid. Every "decode" step therefore attended only to the current token:
- every shadow record has `T = 1`;
- the dense decode perplexity is 2,967;
- the float64 reconstruction is "exactly 0.0", because softmax over a single token returns its value vector unchanged.

**What remains valid in `validate_granite4micro_v1_NO_CACHE.json`:** the full-sequence perplexity check, which uses no cache (PG-19 perplexity 8.5–15.9 against 116,416 for random init), and the bit-identical dense-against-SDPA logits. Only its shadow-reconstruction fields are invalid.

**Fix:** `tests/g5r_cache.py` builds `HybridMambaAttentionDynamicCache` explicitly. All drivers now assert the cache length after prefill and the attention length at every decode step, so a wrong length is a hard error.

**Qwen2.5-0.5B is unaffected:** its decode records show `T = context + step + 1`, and its timing `kv_bytes` equal the analytic full-length values.
