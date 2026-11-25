import sys, torch, json, time
sys.path.insert(0, '/home/user/NEXUS/tests')
import g5r_hf as H, g5r_cache as KC
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(2)
S = '/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad'
H.register()
tok = AutoTokenizer.from_pretrained(S + '/models/granite-4.0-micro-base')
m = AutoModelForCausalLM.from_pretrained(S + '/models/granite-4.0-micro-base', dtype=torch.bfloat16, attn_implementation='g5r').eval()
out = {}
for book in ['3340', '860']:
    ids = torch.tensor(tok(open(f'{S}/data/pg19/test/test/{book}.txt').read().replace('\r\n', '\n'), add_special_tokens=False)['input_ids'][:8192 + 33])
    T = 8192
    with torch.no_grad():
        full = m(input_ids=ids.view(1, -1), use_cache=False, logits_to_keep=33)
    lp = torch.log_softmax(full.logits[0, :-1].double(), -1)       # predicts ids[T+1 .. T+32]
    nll_full = -lp[torch.arange(32), ids[T + 1:T + 33]]
    with torch.no_grad():
        pre = m(input_ids=ids[:T].view(1, -1), past_key_values=KC.make_cache(m, 1), use_cache=True, logits_to_keep=1)
        c = pre.past_key_values; nll_dec = []
        for i in range(32):
            o = m(input_ids=ids[T + i].view(1, 1), past_key_values=c, use_cache=True); c = o.past_key_values
            nll_dec.append(float(-torch.log_softmax(o.logits[0, -1].double(), -1)[ids[T + i + 1]]))
    out[book] = {'ppl_full_nocache': float(torch.exp(nll_full.mean())), 'ppl_cached_decode': float(torch.exp(torch.tensor(nll_dec).mean())),
                 'max_abs_nll_diff': float((nll_full - torch.tensor(nll_dec, dtype=torch.float64)).abs().max())}
    print(book, out[book], flush=True)
json.dump(out, open(S + '/diag_granite_3340.json', 'w'), indent=1)
