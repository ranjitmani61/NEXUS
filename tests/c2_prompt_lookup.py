#!/usr/bin/env python3
"""C2 B2: prompt-lookup speculative decoding vs plain greedy (batch 1, decode phase only).

Version 1.0, 2026-11-10. Charter: specs/C2_G0_batch1_decode_bottleneck_charter.md (§4 B2).

Per input kind: one dense prefill of T-1 tokens (verified cache path; Granite uses the fixed
hybrid cache), shared by all methods. Each timed call continues from a deep copy of that cache
via `generate` (greedy, exactly `new_tokens` new tokens), so the prefill is not in the timing.
Consistency: plain `generate` output must equal a manual greedy loop on the G5-R-verified
cached-decode path. Prompt lookup must emit the same tokens as plain greedy, or the speedup is
not counted. Forward calls are counted to give tokens per target forward pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2_common as C  # noqa: E402
import g5r_cache as KC  # noqa: E402
import g5r_host  # noqa: E402


@torch.no_grad()
def manual_greedy(model, cache, last_tok, n):
    out, nxt = [], last_tok
    for _ in range(n):
        o = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        out.append(nxt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--context", type=int, default=8192); ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--lookup-tokens", type=int, default=10)
    ap.add_argument("--favorable-chunk", type=int, default=2048)
    ap.add_argument("--book", type=Path, required=True); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kinds", nargs="+", default=["typical", "favorable"])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="sdpa").eval()
    calls = {"n": 0}
    model.register_forward_pre_hook(lambda m, args, kwargs: calls.__setitem__("n", calls["n"] + 1), with_kwargs=True)
    res = {"meta": {"version": "1.1", "date": "2026-11-10", "charter": "specs/C2_G0_batch1_decode_bottleneck_charter.md",
                    "model": a.model_name, "dtype": a.dtype, "context": a.context, "new_tokens": a.new_tokens,
                    "reps": a.reps, "lookup_tokens": a.lookup_tokens, "threads": a.threads, "attn_implementation": "sdpa",
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(), "host": g5r_host.host_info()}, "kinds": {}}
    T, n = a.context, a.new_tokens
    book_ids = tok(a.book.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"]
    rng = np.random.default_rng(20261110)
    for kind in a.kinds:
        if kind == "typical":
            ids = book_ids[:T]
        else:
            chunk = book_ids[:a.favorable_chunk]
            ids = (chunk * (T // len(chunk) + 1))[:T]
        x = torch.tensor([ids])
        with torch.no_grad():
            t0 = time.perf_counter()
            pre = model(input_ids=x[:, :T - 1], past_key_values=KC.make_cache(model, 1), use_cache=True, logits_to_keep=1)
            t_pre = time.perf_counter() - t0
        cache0 = pre.past_key_values
        KC.require_len(cache0, T - 1, "c2 B2 prefill")
        ref = manual_greedy(model, copy.deepcopy(cache0), ids[-1], n)
        entry = {"prefill_s": t_pre, "manual_greedy_tokens": ref, "runs": {"plain": [], "lookup": []}}
        gen_kw = dict(max_new_tokens=n, min_new_tokens=n, do_sample=False, num_beams=1, use_cache=True,
                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        for rep in range(a.reps + 1):  # rep 0 = warm-up, not counted
            for mth in [("plain", "lookup")[i] for i in rng.permutation(2)]:
                c = copy.deepcopy(cache0)
                kw = dict(gen_kw)
                if mth == "lookup":
                    kw["prompt_lookup_num_tokens"] = a.lookup_tokens
                calls["n"] = 0
                err = None
                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        out = model.generate(input_ids=x, past_key_values=c, **kw)
                    new = out[0, T:].tolist()
                except Exception as e:  # recorded, never hidden
                    new, err = [], f"{type(e).__name__}: {e}"[:500]
                dtm = time.perf_counter() - t0
                if rep == 0:
                    continue
                # generate() with prompt lookup can emit more than n tokens when the last accepted
                # draft overruns the limit (observed in the smoke test, transformers 4.57.1); only the
                # first n are compared and the overshoot is recorded.
                entry["runs"][mth].append({"seconds": dtm, "forward_calls": calls["n"], "tokens": new,
                                           "n_new": len(new), "overshoot": len(new) - n if new else None,
                                           "equals_manual_greedy": len(new) >= n and new[:n] == ref, "error": err})
                del c
        for mth in ("plain", "lookup"):
            r = entry["runs"][mth]
            ok = [x for x in r if x["error"] is None]
            entry[mth] = {"seconds_median": float(np.median([x["seconds"] for x in ok])) if ok else None,
                          "tokens_per_s": (n / float(np.median([x["seconds"] for x in ok]))) if ok else None,
                          "forward_calls_median": float(np.median([x["forward_calls"] for x in ok])) if ok else None,
                          "all_equal_manual_greedy": bool(ok) and all(x["equals_manual_greedy"] for x in ok),
                          "errors": [x["error"] for x in r if x["error"]]}
        p, l = entry["plain"], entry["lookup"]
        if p["seconds_median"] and l["seconds_median"]:
            ps = [x["seconds"] for x in entry["runs"]["plain"] if x["error"] is None]
            ls = [x["seconds"] for x in entry["runs"]["lookup"] if x["error"] is None]
            entry["lookup_speedup"] = p["seconds_median"] / l["seconds_median"]
            entry["lookup_speedup_ci95"] = list(C.boot_ratio(ps, ls))
            entry["lookup_tokens_identical_to_plain"] = all(x["tokens"][:n] == entry["runs"]["plain"][0]["tokens"][:n]
                                                           for x in entry["runs"]["lookup"] + entry["runs"]["plain"])
        entry["plain_generate_consistent_with_verified_path"] = p["all_equal_manual_greedy"]
        res["kinds"][kind] = entry
        print(json.dumps({"kind": kind, "prefill_s": round(t_pre, 1), "plain_tok_s": p["tokens_per_s"], "lookup_tok_s": l["tokens_per_s"],
                          "speedup": entry.get("lookup_speedup"), "identical": entry.get("lookup_tokens_identical_to_plain"),
                          "plain_consistent": p["all_equal_manual_greedy"], "fwd_calls": [p["forward_calls_median"], l["forward_calls_median"]],
                          "errors": p["errors"] + l["errors"]}), flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        del cache0, pre


if __name__ == "__main__":
    main()
