#!/usr/bin/env python3
"""C2 diagnostic: why prompt-lookup output diverges from plain greedy (Qwen bf16, typical, 8K).

Version 1.1, 2026-11-10. Investigation required by the charter (§4 B2). Exploratory: it does
not enter the decision rule.

v1.1 fix: v1.0 took the reference token from torch.topk, whose order among EXACTLY tied logits
differs from torch.argmax (first maximal index), which HF greedy `generate` uses. At bf16 ties
the v1.0 reference therefore was not plain greedy, and its first_divergence (14) was measured
against the wrong path. v1.1 takes the token from argmax on float32 logits (as `generate` does),
keeps the margin from topk, and records every exact tie.

[Hypothesis] The divergence is a near-tie flipped by finite-precision arithmetic: the
multi-token verification forward uses different kernels from one-token decode.
Test, per dtype in (bf16, fp32):
  1. On the verified plain path (manual greedy, one token per forward), record the top-2
     logit margin at every generated position.
  2. Run prompt lookup and find the first position where it differs from plain greedy.
  3. Report the plain-path margin at that position, next to the bf16 ulp at that logit.
Prediction if the hypothesis holds: the bf16 divergence sits at a margin within a few bf16 ulps,
and fp32 lookup is identical to fp32 greedy (or diverges only at a far smaller margin).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_cache as KC  # noqa: E402
import g5r_host  # noqa: E402


def bf16_ulp(x: float) -> float:
    t = torch.tensor(abs(x), dtype=torch.bfloat16)
    return float(torch.nextafter(t, torch.tensor(float("inf"), dtype=torch.bfloat16)) - t)


@torch.no_grad()
def greedy_with_margins(model, cache, last_tok, n):
    toks, margins, top1, ties = [], [], [], []
    nxt = last_tok
    for _ in range(n):
        o = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        lg = o.logits[0, -1].float()
        v, _ = torch.topk(lg, 2)
        nxt = int(lg.argmax())  # first maximal index, as HF greedy generate
        toks.append(nxt); margins.append(float(v[0] - v[1])); top1.append(float(v[0]))
        ties.append(torch.nonzero(lg == v[0]).flatten().tolist() if float(v[0] - v[1]) == 0.0 else None)
    return toks, margins, top1, ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--book", type=Path, required=True)
    ap.add_argument("--context", type=int, default=8192); ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--lookup-tokens", type=int, default=10); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = tok(a.book.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"][:a.context]
    T, n = a.context, a.new_tokens
    x = torch.tensor([ids])
    res = {"meta": {"version": "1.1", "date": "2026-11-10", "label": "exploratory diagnostic (charter §4 B2 investigation)",
                    "context": T, "new_tokens": n, "lookup_tokens": a.lookup_tokens, "book": a.book.name,
                    "torch": torch.__version__, "transformers": transformers.__version__, "host": g5r_host.host_info()},
           "dtypes": {}}
    for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="sdpa").eval()
        with torch.no_grad():
            pre = model(input_ids=x[:, :T - 1], past_key_values=KC.make_cache(model, 1), use_cache=True, logits_to_keep=1)
        cache0 = pre.past_key_values
        KC.require_len(cache0, T - 1, "divergence prefill")
        ref, margins, top1, ties = greedy_with_margins(model, copy.deepcopy(cache0), ids[-1], n)
        with torch.no_grad():
            out = model.generate(input_ids=x, past_key_values=copy.deepcopy(cache0), max_new_tokens=n, min_new_tokens=n,
                                 do_sample=False, num_beams=1, prompt_lookup_num_tokens=a.lookup_tokens,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        lk = out[0, T:T + n].tolist()
        div = next((k for k in range(min(n, len(lk))) if lk[k] != ref[k]), None)
        srt = sorted(margins)
        e = {"greedy_tokens": ref, "lookup_tokens": lk, "first_divergence": div,
             "margin_at_divergence": margins[div] if div is not None else None,
             "top1_logit_at_divergence": top1[div] if div is not None else None,
             "bf16_ulp_at_that_logit": bf16_ulp(top1[div]) if div is not None else None,
             "margin_median": srt[len(srt) // 2], "margin_min": srt[0],
             "rank_of_divergence_margin_among_positions": (sorted(range(n), key=lambda k: margins[k]).index(div) + 1) if div is not None else None,
             "exact_tie_positions": [k for k in range(n) if ties[k] is not None],
             "exact_tie_token_sets": {str(k): ties[k] for k in range(n) if ties[k] is not None},
             "lookup_token_at_divergence_in_tie_set": (lk[div] in ties[div]) if (div is not None and ties[div] is not None) else None,
             "margins": margins}
        res["dtypes"][name] = e
        print(json.dumps({"dtype": name, "first_divergence": div, "margin_at_div": e["margin_at_divergence"],
                          "bf16_ulp": e["bf16_ulp_at_that_logit"], "margin_median": e["margin_median"], "margin_min": e["margin_min"],
                          "rank": e["rank_of_divergence_margin_among_positions"]}), flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        del model, pre, cache0


if __name__ == "__main__":
    main()
