import sys, torch, json
sys.path.insert(0, '/home/user/NEXUS/tests')
import g5r_hf as H, g5r_cache as KC
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(2)
S = '/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad'
H.register()
name, mdir, dt = sys.argv[1], sys.argv[2], {'bf16': torch.bfloat16, 'fp32': torch.float32}[sys.argv[3]]
tok = AutoTokenizer.from_pretrained(mdir)
m = AutoModelForCausalLM.from_pretrained(mdir, dtype=dt, attn_implementation='g5r').eval()
res = {}
T = 8192
for book in ['3340', '860']:
    ids = torch.tensor(tok(open(f'{S}/data/pg19/test/test/{book}.txt').read().replace('\r\n', '\n'), add_special_tokens=False)['input_ids'][:T + 3])
    with torch.no_grad():
        full = m(input_ids=ids[:T + 2].view(1, -1), use_cache=False, logits_to_keep=2).logits[0].double()   # logits at positions T, T+1
        pre = m(input_ids=ids[:T].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1)
        d1 = m(input_ids=ids[T:T + 1].view(1, 1), past_key_values=pre.past_key_values, use_cache=True).logits[0, -1].double()   # M=1 decode at pos T
        pre2 = m(input_ids=ids[:T].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1)
        d2 = m(input_ids=ids[T:T + 2].view(1, 2), past_key_values=pre2.past_key_values, use_cache=True).logits[0].double()   # M=2 chunk, pos T,T+1
    f0 = full[0]
    def stats(x):
        lpx, lpf = torch.log_softmax(x, -1), torch.log_softmax(f0, -1)
        return {'max_abs_logit_diff': float((x - f0).abs().max()), 'kl_full_to_x': float((lpf.exp() * (lpf - lpx)).sum()),
                'nll_true_x': float(-lpx[ids[T + 1]]), 'nll_true_full': float(-lpf[ids[T + 1]]), 'argmax_same': bool(x.argmax() == f0.argmax())}
    res[book] = {'decode_M1_vs_full': stats(d1), 'chunk_M2_vs_full': stats(d2[0])}
    print(name, book, json.dumps(res[book]), flush=True)
json.dump(res, open(f'{S}/diag_decode_paths_{name}.json', 'w'), indent=1)
