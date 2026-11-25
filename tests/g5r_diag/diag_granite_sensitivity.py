import sys, torch, json
sys.path.insert(0, '/home/user/NEXUS/tests')
import g5r_hf as H, g5r_cache as KC
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(2)
S = '/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad'
H.register()
mdir = S + '/models/granite-4.0-micro-base'
tok = AutoTokenizer.from_pretrained(mdir)
res = {}
for impl in ['g5r', 'sdpa', 'eager']:
    m = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, attn_implementation=impl).eval()
    T = 8192
    ids = torch.tensor(tok(open(f'{S}/data/pg19/test/test/3340.txt').read().replace('\r\n', '\n'), add_special_tokens=False)['input_ids'][:T + 4])
    with torch.no_grad():
        if impl == 'eager':   # eager materialises T x T attention: only the cached variants below are affordable
            outs = {}
        else:
            outs = {'full_len_T+2': m(input_ids=ids[:T + 2].view(1, -1), use_cache=False, logits_to_keep=2).logits[0, 0].double(),
                    'full_len_T+3': m(input_ids=ids[:T + 3].view(1, -1), use_cache=False, logits_to_keep=3).logits[0, 0].double()}
        pre = m(input_ids=ids[:T].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1) if impl != 'eager' else None
        if pre is not None:
            outs['cached_prefillT_decode1'] = m(input_ids=ids[T:T + 1].view(1, 1), past_key_values=pre.past_key_values, use_cache=True).logits[0, -1].double()
            pre2 = m(input_ids=ids[:T - 100].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1)
            outs['cached_prefillT-100_chunk101'] = m(input_ids=ids[T - 100:T + 1].view(1, -1), past_key_values=pre2.past_key_values, use_cache=True).logits[0, -1].double()
    ref = outs.get('full_len_T+2')
    r = {}
    for k, v in outs.items():
        lp = torch.log_softmax(v, -1)
        r[k] = {'nll_true_next': float(-lp[ids[T + 1]]), 'argmax': int(v.argmax())}
        if ref is not None:
            lr = torch.log_softmax(ref, -1)
            r[k].update({'max_abs_logit_diff_vs_fullT+2': float((v - ref).abs().max()), 'kl_vs_fullT+2': float((lr.exp() * (lr - lp)).sum())})
    res[impl] = r
    print(impl, json.dumps(r), flush=True)
    del m
json.dump(res, open(S + '/diag_granite_sensitivity.json', 'w'), indent=1)
