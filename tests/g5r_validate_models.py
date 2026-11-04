#!/usr/bin/env python3
"""G5-R model/capture validation (v1.0, 2026-11-04).

1. Genuine-pretrained functional check: PG-19 perplexity of the downloaded weights versus
   a randomly initialised network of the SAME config (negative control only; the random
   network is never used as evidence about pretrained behaviour).
2. Capture validity: logits with attn_implementation="g5r" (dense mode) vs stock "sdpa".
3. Shadow reconstruction: float64 dense attention recomputed from captured q/K/V vs the
   model's own SDPA output at decode steps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_hf as H  # noqa: E402


@torch.no_grad()
def ppl(model, ids):
    out = model(input_ids=ids.view(1, -1), use_cache=False)
    lp = torch.log_softmax(out.logits[0, :-1].double(), -1)
    nll = -lp[torch.arange(ids.numel() - 1), ids[1:]]
    return float(torch.exp(nll.mean())), out.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--dtype", default="fp32"); ap.add_argument("--books", nargs="+", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=2048); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    H.register()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    m_g5r = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="g5r").eval()
    res = {"model": a.model_name, "dtype": a.dtype, "tokens": a.tokens, "books": {}}
    for b in a.books:
        ids = torch.tensor(tok(b.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"][: a.tokens])
        p, logits_g5r = ppl(m_g5r, ids)
        res["books"][b.name] = {"ppl_pretrained": p}
        print(b.name, "ppl", round(p, 3), flush=True)
    first = a.books[0]
    ids = torch.tensor(tok(first.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"][: a.tokens])
    # capture validity vs stock sdpa
    _, lg = ppl(m_g5r, ids[:512])
    del m_g5r
    m_sdpa = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="sdpa").eval()
    pp, ls = ppl(m_sdpa, ids[:512])
    res["g5r_dense_vs_sdpa_logits_maxabs"] = float((lg.double() - ls.double()).abs().max())
    del m_sdpa, lg, ls          # free before building the control (3.4B bf16 models: RAM limit)
    # negative control: random init, same config
    cfg = AutoConfig.from_pretrained(a.model_dir)
    torch.manual_seed(0)
    m_rand = AutoModelForCausalLM.from_config(cfg, dtype=dt, attn_implementation="sdpa").eval()
    pr, _ = ppl(m_rand, ids[:512])
    res["control_random_init_ppl_512"] = pr
    res["pretrained_ppl_512_same_tokens"] = pp
    res["vocab_size"] = cfg.vocab_size
    del m_rand
    # shadow reconstruction at decode
    m = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="g5r").eval()
    with torch.no_grad():
        pre = m(input_ids=ids[:1000].view(1, -1), use_cache=True, logits_to_keep=1)
        H.CTL.mode = "shadow"; H.CTL.records = []
        cache = pre.past_key_values
        for i in range(3):
            H.CTL.step = i
            out = m(input_ids=ids[1000 + i].view(1, 1), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        H.CTL.mode = "dense"
    rel = [r["capture_maxabs"] / max(r["capture_outnorm"], 1e-30) for r in H.CTL.records]
    res["shadow_reconstruction_max_abs"] = max(r["capture_maxabs"] for r in H.CTL.records)
    res["shadow_reconstruction_max_rel_to_outnorm"] = max(rel)
    res["shadow_layers_seen"] = sorted({r["layer"] for r in H.CTL.records})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "books"}, indent=1))


if __name__ == "__main__":
    main()
