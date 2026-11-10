#!/usr/bin/env python3
"""C2 M1/M2/M3 + B1: batch-1 decode breakdown, derived bytes, host bandwidth, int8 baseline.

Version 1.0, 2026-11-10. Charter: specs/C2_G0_batch1_decode_bottleneck_charter.md (pre-registered).

Per context T: each variant's own dense prefill (timed, separate), then `reps` rounds in a
seeded shuffled order over (variant, hooks on/off). Each round deep-copies the variant's
prefill cache and runs `steps` teacher-forced decode steps; every step is timed.
- hooks off: official step time (no instrumentation).
- hooks on : per-class attribution (attention core via the g5r_fast timer; Linear classes via
  forward hooks; "other" = step - attributed).
Quality (B1): teacher-forced continuation per book; each variant uses its own prefill;
compared with the `dense` variant (same dtype).
Measured on this VM only; no GPU, no energy, no DRAM counters.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c2_common as C  # noqa: E402
import g5r_cache as KC  # noqa: E402
import g5r_fast_attn as F  # noqa: E402
import g5r_host  # noqa: E402


@torch.no_grad()
def steps_timed(model, cache, toks, timer=None):
    rows = []
    for i in range(toks.shape[1]):
        if timer is not None:
            timer.reset()
        a0 = F.ST.attn_seconds
        t0 = time.perf_counter()
        out = model(input_ids=toks[:, i:i + 1], past_key_values=cache, use_cache=True)
        dt = time.perf_counter() - t0
        cache = out.past_key_values
        r = {"step_s": dt, "attn_core": F.ST.attn_seconds - a0}
        if timer is not None:
            for c in ("attn_proj", "mlp", "lm_head", "linear_other"):
                r[c] = timer.acc.get(c, 0.0)
            r["other"] = dt - sum(r[c] for c in ("attn_core", "attn_proj", "mlp", "lm_head", "linear_other"))
        rows.append(r)
    return rows


@torch.no_grad()
def prefill(model, ids, T, batch=1):
    F.reset("dense"); F.ST.timing = False
    t0 = time.perf_counter()
    pre = model(input_ids=ids[:, :T], past_key_values=KC.make_cache(model, batch), use_cache=True, logits_to_keep=1)
    tp = time.perf_counter() - t0
    KC.require_len(pre.past_key_values, T, "c2 prefill")
    return pre, tp


@torch.no_grad()
def teacher_forced(model, ids, T, n):
    pre, _ = prefill(model, ids, T)
    cache = pre.past_key_values
    lps = [torch.log_softmax(pre.logits[0, -1].double(), -1)]
    for i in range(n - 1):
        out = model(input_ids=ids[:, T + i:T + i + 1], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        lps.append(torch.log_softmax(out.logits[0, -1].double(), -1))
    KC.require_len(cache, T + n - 1, "c2 quality decode")
    return torch.stack(lps)  # [n, V]; row i predicts ids[T+i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--variants", nargs="+", default=["dense"], choices=["dense", "int8dyn", "int8dyn_pc", "int8dyn_pc_nolm"])
    ap.add_argument("--contexts", type=int, nargs="+", required=True)
    ap.add_argument("--timing-book", type=Path, required=True)
    ap.add_argument("--quality-books", nargs="*", type=Path, default=[])
    ap.add_argument("--quality-steps", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if any(v.startswith("int8") for v in a.variants) and a.dtype != "fp32":
        raise SystemExit("int8dyn requires --dtype fp32 (torch.ao dynamic quantization)")
    torch.set_num_threads(a.threads)
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    F.register()
    proc = psutil.Process()
    res = {"meta": {"version": "1.0", "date": "2026-11-10", "charter": "specs/C2_G0_batch1_decode_bottleneck_charter.md",
                    "model": a.model_name, "dtype": a.dtype, "variants": a.variants, "steps": a.steps, "reps": a.reps,
                    "threads": a.threads, "torch": torch.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(), "host": g5r_host.host_info(),
                    "energy": "NOT MEASURED", "dram_counters": "NONE"},
           "triad_before": C.triad_gbps(threads=a.threads), "contexts": {}, "quality": {}}
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    base = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="g5r_fast").eval()
    models = {}
    for v in a.variants:
        if v == "dense":
            models[v] = base
        else:
            models[v] = C.quantize_variant(base, v)
    res["meta"]["weight_bytes"] = {v: C.linear_weight_bytes(m) for v, m in models.items()}
    rng = np.random.default_rng(20261110)
    text = a.timing_book.read_text().replace("\r\n", "\n")
    all_ids = tok(text, add_special_tokens=False)["input_ids"]
    for T in a.contexts:
        ids = torch.tensor([all_ids[:T + a.steps]])
        toks = ids[:, T:T + a.steps]
        entry = {"timing_book": a.timing_book.name, "variants": {}}
        caches = {}
        for v, m in models.items():
            pre, tp = prefill(m, ids, T)
            caches[v] = pre.past_key_values
            entry["variants"][v] = {"prefill_s": tp, "kv_bytes": KC.kv_bytes(caches[v])}
            del pre
        timers = {}
        combos = list(itertools.product(models, [False, True]))
        for v, hooks in combos:  # warm-up, untimed
            F.reset("dense"); F.ST.timing = True
            tm = C.LinearTimer(models[v]) if hooks else None
            steps_timed(models[v], copy.deepcopy(caches[v]), toks[:, :2], tm)
            if tm:
                tm.remove()
        raw = {f"{v}|{'on' if h else 'off'}": [] for v, h in combos}
        for rep in range(a.reps):
            for i in rng.permutation(len(combos)):
                v, hooks = combos[i]
                c = copy.deepcopy(caches[v])
                F.reset("dense"); F.ST.timing = True
                tm = C.LinearTimer(models[v]) if hooks else None
                rows = steps_timed(models[v], c, toks, tm)
                if tm:
                    tm.remove()
                KC.require_len(c, T + a.steps, f"c2 timing {v}")
                raw[f"{v}|{'on' if hooks else 'off'}"].append({"rows": rows, "dense_native_gqa_calls": F.ST.dense_native_gqa_calls,
                                                               "rss_bytes": proc.memory_info().rss})
                del c
        for v in models:
            off = np.concatenate([[r["step_s"] for r in run["rows"][1:]] for run in raw[f"{v}|off"]])
            on_rows = [r for run in raw[f"{v}|on"] for r in run["rows"][1:]]
            on = np.array([r["step_s"] for r in on_rows])
            shares = {c: float(np.median([r[c] / r["step_s"] for r in on_rows])) for c in C.CLASSES}
            secs = {c: float(np.median([r[c] for r in on_rows])) for c in C.CLASSES}
            wb = res["meta"]["weight_bytes"][v]
            kvb = entry["variants"][v]["kv_bytes"]
            tot_b = sum(wb.values()) + kvb
            med_off = float(np.median(off))
            e = entry["variants"][v]
            e.update({"steady_step_s_p50": med_off, "steady_step_s_p95": float(np.quantile(off, .95)),
                      "tokens_per_s": 1.0 / med_off, "hooked_step_s_p50": float(np.median(on)),
                      "hook_overhead_frac": float(np.median(on) / med_off - 1.0),
                      "time_share_median": shares, "time_s_median": secs,
                      "bytes_derived": {"weights_by_class": wb, "weights_total": sum(wb.values()), "kv_read": kvb,
                                        "kv_share": kvb / tot_b, "lm_head_share": wb.get("lm_head", 0) / tot_b,
                                        "attn_proj_share": wb.get("attn_proj", 0) / tot_b, "mlp_share": wb.get("mlp", 0) / tot_b},
                      "effective_gbps_derived": tot_b / med_off / 1e9,
                      "raw": {"off": raw[f"{v}|off"], "on": raw[f"{v}|on"]}})
        if "dense" in models:
            d_off = np.concatenate([[r["step_s"] for r in run["rows"][1:]] for run in raw["dense|off"]])
            for v in models:
                if v == "dense":
                    continue
                q_off = np.concatenate([[r["step_s"] for r in run["rows"][1:]] for run in raw[f"{v}|off"]])
                entry["variants"][v]["speedup_vs_dense_step"] = float(np.median(d_off) / np.median(q_off))
                entry["variants"][v]["speedup_vs_dense_ci95"] = list(C.boot_ratio(d_off, q_off))
        res["contexts"][str(T)] = entry
        print(json.dumps({"T": T, **{v: {"step_ms": round(entry["variants"][v]["steady_step_s_p50"] * 1e3, 2),
                                         "hook_ovh": round(entry["variants"][v]["hook_overhead_frac"], 3),
                                         "shares": {k: round(x, 3) for k, x in entry["variants"][v]["time_share_median"].items()},
                                         "kv_byte_share": round(entry["variants"][v]["bytes_derived"]["kv_share"], 3),
                                         "eff_GBps": round(entry["variants"][v]["effective_gbps_derived"], 1)} for v in models},
                          **{f"{v}_speedup": round(entry["variants"][v]["speedup_vs_dense_step"], 3) for v in models if v != "dense"}}), flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        del caches
    if a.quality_books and len(models) > 1:
        for T in a.contexts:
            q = {"books": []}
            for b in a.quality_books:
                bid = tok(b.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"]
                ids = torch.tensor([bid[:T + a.quality_steps]])
                tgt = ids[0, T:T + a.quality_steps]
                lp = {v: teacher_forced(m, ids, T, a.quality_steps) for v, m in models.items()}
                row = {"book": b.name, "methods": {}}
                nll_d = -lp["dense"][torch.arange(len(tgt)), tgt]
                for v in models:
                    nll = -lp[v][torch.arange(len(tgt)), tgt]
                    kl = (lp["dense"].exp() * (lp["dense"] - lp[v])).sum(-1)
                    row["methods"][v] = {"nll_per_token": nll.tolist(), "delta_nll_mean": float((nll - nll_d).mean()),
                                         "top1_agree": float((lp["dense"].argmax(-1) == lp[v].argmax(-1)).double().mean()),
                                         "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}
                q["books"].append(row)
            for v in models:
                d = np.concatenate([np.array(r["methods"][v]["nll_per_token"]) - np.array(r["methods"]["dense"]["nll_per_token"]) for r in q["books"]])
                ag = np.mean([r["methods"][v]["top1_agree"] for r in q["books"]])
                bs = []
                g = np.random.default_rng(0)
                per = [np.array(r["methods"][v]["nll_per_token"]) - np.array(r["methods"]["dense"]["nll_per_token"]) for r in q["books"]]
                for _ in range(2000):
                    pick = g.integers(0, len(per), len(per))
                    bs.append(np.exp(np.concatenate([per[k] for k in pick]).mean()))
                q[v] = {"ppl_ratio": float(np.exp(d.mean())), "ppl_ratio_ci95_book_bootstrap": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                        "top1_agree": float(ag), "matched_quality": bool(np.exp(d.mean()) <= 1.01 and ag >= 0.98)}
            res["quality"][str(T)] = q
            print(json.dumps({"quality_T": T, **{v: {k: q[v][k] for k in ("ppl_ratio", "top1_agree", "matched_quality")} for v in models}}), flush=True)
            a.output.write_text(json.dumps(res, indent=1) + "\n")
    res["triad_after"] = C.triad_gbps(threads=a.threads)
    res["meta"]["host_after"] = g5r_host.host_info()
    a.output.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
