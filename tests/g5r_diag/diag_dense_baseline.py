import sys, time, torch, json
sys.path.insert(0, '/home/user/NEXUS/tests')
import g5r_fast_attn as F, g5r_cache as KC
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4)
S = '/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad'
F.register()
seen = {}
orig = F._dense
def spy(module, query, key, value, attention_mask, scaling, **kw):
    seen.setdefault('mask_none', set()).add(attention_mask is None)
    return orig(module, query, key, value, attention_mask, scaling, **kw)
F._dense = spy
res = {}
for name, mdir in [('qwen05', S + '/models/qwen2.5-0.5b'), ('granite4micro', S + '/models/granite-4.0-micro-base')]:
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, attn_implementation='g5r_fast').eval()
    ids = torch.tensor(tok(open(S + '/data/pg19/test/test/860.txt').read(), add_special_tokens=False)['input_ids'][:301])
    seen.clear(); F.reset('dense')
    with torch.no_grad():
        pre = m(input_ids=ids[:300].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1)
        seen.clear()
        m(input_ids=ids[300:301].view(1, 1), past_key_values=pre.past_key_values, use_cache=True)
    cfg = m.config
    hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    res[name] = {'decode_mask_is_None': sorted(seen['mask_none'])}
    # attention-op microbenchmark at T=8192 (bf16): HF wrapper with 4D mask (repeat_kv) vs native GQA, no mask
    T = 8192
    q = torch.randn(1, hq, 1, d, dtype=torch.bfloat16); k = torch.randn(1, hkv, T, d, dtype=torch.bfloat16); v = torch.randn_like(k)
    mask = torch.zeros(1, 1, 1, T, dtype=torch.bfloat16)
    mod = m.model.layers[0].self_attn
    def bench(fn, n=30):
        for _ in range(3): fn()
        t = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - t) / n * 1e3
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    res[name]['ms_hf_wrapper_with_mask'] = bench(lambda: sdpa_attention_forward(mod, q, k, v, mask, scaling=1 / d ** .5))
    res[name]['ms_hf_wrapper_mask_none'] = bench(lambda: sdpa_attention_forward(mod, q, k, v, None, scaling=1 / d ** .5))
    res[name]['ms_torch_sdpa_enable_gqa'] = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1 / d ** .5, enable_gqa=True))
    print(name, json.dumps(res[name]), flush=True)
    del m
json.dump(res, open(S + '/diag_dense_baseline.json', 'w'), indent=1)
