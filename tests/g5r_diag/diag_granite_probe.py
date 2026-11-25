import sys, torch, json
sys.path.insert(0, '/home/user/NEXUS/tests')
import g5r_cache as KC
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
torch.set_num_threads(2)
S = '/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad'
REC = {}
def probe(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    REC.setdefault(module.layer_idx, []).append({'q': query[0, :, -1].float().clone(), 'k': key[0].float().clone(), 'v': value[0].float().clone(),
                                                 'qlen': query.shape[2], 'mask_none': attention_mask is None,
                                                 'is_causal_attr': getattr(module, 'is_causal', None)})
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kw)
AttentionInterface.register('probe', probe)
try:
    from transformers import AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionMaskInterface.register('probe', sdpa_mask)
except Exception:
    pass
mdir = S + '/models/granite-4.0-micro-base'
tok = AutoTokenizer.from_pretrained(mdir)
m = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, attn_implementation='probe').eval()
T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
ids = torch.tensor(tok(open(f'{S}/data/pg19/test/test/3340.txt').read().replace('\r\n', '\n'), add_special_tokens=False)['input_ids'][:T + 1])
with torch.no_grad():
    REC.clear(); m(input_ids=ids.view(1, -1), use_cache=False, logits_to_keep=1); A = {l: r[0] for l, r in REC.items()}
    REC.clear(); pre = m(input_ids=ids[:T].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1)
    m(input_ids=ids[T:T + 1].view(1, 1), past_key_values=pre.past_key_values, use_cache=True); B = {l: r for l, r in REC.items()}
out = []
for l in sorted(A):
    a, bp, bd = A[l], B[l][0], B[l][1]
    rel = lambda x, y: float((x - y).abs().max() / y.abs().max().clamp_min(1e-30))
    out.append({'layer': l, 'k_prefix_rel': rel(bp['k'], a['k'][:, :T]), 'v_prefix_rel': rel(bp['v'], a['v'][:, :T]),
                'q_newtoken_rel': rel(bd['q'], a['q']), 'k_newtoken_rel': rel(bd['k'][:, T], a['k'][:, T]),
                'decode_klen': bd['k'].shape[1], 'full_qlen': a['qlen'], 'full_mask_none': a['mask_none'], 'prefill_mask_none': bp['mask_none'],
                'decode_mask_none': bd['mask_none'], 'is_causal_attr': a['is_causal_attr']})
for r in out[:6] + out[-2:]:
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
first = next((r['layer'] for r in out if max(r['k_prefix_rel'], r['q_newtoken_rel'], r['k_newtoken_rel']) > 1e-2), None)
print('first layer with >1% relative divergence:', first)
json.dump(out, open(f'{S}/diag_granite_probe_T{T}.json', 'w'), indent=1)
