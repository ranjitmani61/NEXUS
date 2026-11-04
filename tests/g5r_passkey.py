#!/usr/bin/env python3
"""G5-R passkey retrieval in PG-19 haystacks (real pretrained models, CPU).

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md
(Amendment A1 item 4): the prompt is prefilled (dense, shared by all methods) only up
to the end of the haystack; the question tokens and the greedy answer tokens are
decoded one at a time WITH THE METHOD UNDER TEST, so retrieval depends on it.

Prompt structure follows Mohtashami & Jaggi (2023, "Landmark Attention") passkey
format, with natural PG-19 text as the haystack instead of repeated filler lines.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_hf as H  # noqa: E402
import g5r_cache as KC  # noqa: E402

INTRO = ("There is an important info hidden inside a lot of irrelevant text. "
         "Find it and memorize it. I will quiz you about the important information there.\n\n")
QUESTION = "\n\nWhat is the pass key? The pass key is"


def needle(key):
    return f"\n\nThe pass key is {key}. Remember it. {key} is the pass key.\n\n"


@torch.no_grad()
def answer(model, tok, cache, q_ids, n_new=7):
    gen = []
    nxt = None
    for i, t in enumerate(list(q_ids)):
        H.CTL.step = i
        out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())
    for j in range(n_new):
        gen.append(nxt)
        H.CTL.step = len(q_ids) + j
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--dtype", default="fp32"); ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--books", nargs="+", type=Path, required=True)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--keys-per-depth", type=int, default=2)
    ap.add_argument("--methods", nargs="+", default=["dense", "nx_box", "quest:1024", "quest:2048", "quest:4096"])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    H.register()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="g5r").eval()
    rng = np.random.default_rng(20261104 + a.context)
    intro_ids = tok(INTRO, add_special_tokens=False)["input_ids"]
    q_ids = tok(QUESTION, add_special_tokens=False)["input_ids"]
    trials = []
    ti = 0
    for depth in a.depths:
        for _ in range(a.keys_per_depth):
            key = int(rng.integers(10000, 100000))
            book = a.books[ti % len(a.books)]; ti += 1
            hay = tok(book.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"]
            n_ids = tok(needle(key), add_special_tokens=False)["input_ids"]
            n_hay = a.context - len(intro_ids) - len(n_ids)
            pos = int(depth * n_hay)
            prefill = intro_ids + hay[:pos] + n_ids + hay[pos:n_hay]
            assert len(prefill) == a.context
            H.CTL.mode = "dense"
            t0 = time.perf_counter()
            with torch.no_grad():
                pre = model(input_ids=torch.tensor([prefill]), past_key_values=KC.make_cache(model, 1), use_cache=True, logits_to_keep=1)
            tp = time.perf_counter() - t0
            KC.require_len(pre.past_key_values, a.context, "passkey prefill")
            row = {"depth": depth, "key": key, "book": book.name, "needle_token_pos": len(intro_ids) + pos,
                   "prefill_s": tp, "answers": {}}
            for m in a.methods:
                if m == "dense":
                    H.CTL.mode, H.CTL.method = "dense", ""
                else:
                    H.CTL.mode, H.CTL.method = "live", m
                H.CTL.live_stats = []
                txt = answer(model, tok, copy.deepcopy(pre.past_key_values), q_ids)
                H.CTL.mode = "dense"
                mnum = re.search(r"\d{5}", txt)
                row["answers"][m] = {"text": txt, "correct": bool(mnum and int(mnum.group()) == key),
                                     "pages_frac": float(np.mean([s["pages_frac"] for s in H.CTL.live_stats])) if H.CTL.live_stats else 1.0}
            trials.append(row)
            print(json.dumps({"T": a.context, "depth": depth, "key": key,
                              **{m: v["correct"] for m, v in row["answers"].items()}}), flush=True)
            summary = {m: float(np.mean([t["answers"][m]["correct"] for t in trials])) for m in a.methods}
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({"meta": {"version": "1.0", "date": "2026-11-04", "model": a.model_name,
                                                     "dtype": a.dtype, "context": a.context, "methods": a.methods,
                                                     "seed": 20261104 + a.context},
                                            "accuracy": summary, "n_trials": len(trials), "trials": trials}, indent=1) + "\n")


if __name__ == "__main__":
    main()
