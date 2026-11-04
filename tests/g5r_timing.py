#!/usr/bin/env python3
"""G5-R CPU decode latency/throughput/memory benchmark (real pretrained models).

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md.

For each context length: one dense prefill (timed, reported separately), then for
each repetition and each method in a seeded shuffled order: a fresh deep copy of the
prefill cache, `steps` teacher-forced decode steps, wall-clock time of each step
(torch.no_grad, 4 threads). Step 1 includes the method's initial metadata build.

Measured on this VM only. NOT GPU, NOT energy (no counter available), NOT llama.cpp.
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
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_fast_attn as F  # noqa: E402


def kv_bytes(cache):
    n = 0
    for layer in getattr(cache, "layers", []):
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if t is not None:
                n += t.numel() * t.element_size()
    return n


def meta_bytes():
    return sum(v.numel() * v.element_size() for m in F.ST.meta.values() for k, v in m.items() if torch.is_tensor(v))


@torch.no_grad()
def run_steps(model, cache, toks):
    times = []
    for i in range(toks.shape[1]):
        t0 = time.perf_counter()
        out = model(input_ids=toks[:, i:i + 1], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        times.append(time.perf_counter() - t0)
    return times, out.logits[:, -1].float()


def boot_ratio(a, b, n=2000, seed=0):
    """95% bootstrap CI of median(a)/median(b) resampling each independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    r = [np.median(rng.choice(a, a.size)) / np.median(rng.choice(b, b.size)) for _ in range(n)]
    return float(np.quantile(r, .025)), float(np.quantile(r, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--contexts", type=int, nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["dense", "quest:1024", "quest:2048", "nx_box", "nx_sph", "nx_one"])
    ap.add_argument("--books", nargs="+", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    F.register()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=dt, attn_implementation="g5r_fast").eval()
    proc = psutil.Process()
    res = {"meta": {"version": "1.0", "date": "2026-11-04", "model": a.model_name, "dtype": a.dtype,
                    "batch": a.batch, "steps": a.steps, "reps": a.reps, "threads": a.threads,
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(), "cpu": "4 vCPU Xeon fam6 model207 (AVX512, AMX-BF16)",
                    "energy": "NOT MEASURED (no RAPL/powercap on this VM)",
                    "baseline": "stock transformers SDPA on CPU; not llama.cpp, not GPU"}, "contexts": {}}
    rng = np.random.default_rng(20261104)
    for T in a.contexts:
        ids = []
        for bi in range(a.batch):
            text = a.books[bi % len(a.books)].read_text().replace("\r\n", "\n")
            x = tok(text, add_special_tokens=False)["input_ids"]
            off = (bi // len(a.books)) * 1000
            ids.append(x[off: off + T + a.steps])
        ids = torch.tensor(ids)
        F.reset("dense"); F.ST.timing = False
        t0 = time.perf_counter()
        with torch.no_grad():
            pre = model(input_ids=ids[:, :T], use_cache=True, logits_to_keep=1)
        t_pre = time.perf_counter() - t0
        cache0 = pre.past_key_values
        toks = ids[:, T: T + a.steps]
        entry = {"prefill_s": t_pre, "kv_bytes": kv_bytes(cache0), "methods": {}}
        # warm-up each method once (untimed)
        for m in a.methods:
            F.reset(m); run_steps(model, copy.deepcopy(cache0), toks[:, :2])
        raw = {m: [] for m in a.methods}
        logits_ref = None
        for rep in range(a.reps):
            for m in [a.methods[i] for i in rng.permutation(len(a.methods))]:
                c = copy.deepcopy(cache0)
                F.reset(m); F.ST.timing = True
                times, lg = run_steps(model, c, toks)
                rec = {"step_s": times, "attn_s": F.ST.attn_seconds, "build_s": F.ST.build_seconds,
                       "pages_read_frac": (F.ST.pages_read / F.ST.pages_total) if F.ST.pages_total else 1.0,
                       "fallbacks": F.ST.fallbacks, "attn_calls": F.ST.calls, "meta_bytes": meta_bytes(),
                       "rss_bytes": proc.memory_info().rss}
                if m == "dense" and logits_ref is None:
                    logits_ref = lg
                rec["final_logits_maxabs_vs_dense"] = None if logits_ref is None else float((lg - logits_ref).abs().max())
                raw[m].append(rec)
                del c
        dense_tot = [sum(r["step_s"]) for r in raw["dense"]]
        dense_steps = np.concatenate([r["step_s"][1:] for r in raw["dense"]])
        for m in a.methods:
            tot = [sum(r["step_s"]) for r in raw[m]]
            steady = np.concatenate([r["step_s"][1:] for r in raw[m]])
            lo, hi = boot_ratio(dense_tot, tot)
            entry["methods"][m] = {
                "total_s_median": float(np.median(tot)), "first_step_s_median": float(np.median([r["step_s"][0] for r in raw[m]])),
                "steady_step_s_p50": float(np.median(steady)), "steady_step_s_p95": float(np.quantile(steady, .95)),
                "tokens_per_s": float(a.batch * a.steps / np.median(tot)),
                "speedup_vs_dense_total": float(np.median(dense_tot) / np.median(tot)),
                "speedup_vs_dense_total_ci95": [lo, hi],
                "speedup_vs_dense_steady_p50": float(np.median(dense_steps) / np.median(steady)),
                "attn_share_of_decode": float(np.median([r["attn_s"] / sum(r["step_s"]) for r in raw[m]])),
                "build_s_median": float(np.median([r["build_s"] for r in raw[m]])),
                "pages_read_frac": float(np.median([r["pages_read_frac"] for r in raw[m]])),
                "fallbacks_median": float(np.median([r["fallbacks"] for r in raw[m]])),
                "meta_bytes": int(raw[m][0]["meta_bytes"]), "rss_bytes_max": int(max(r["rss_bytes"] for r in raw[m])),
                "raw": raw[m]}
        f = entry["methods"]["dense"]["attn_share_of_decode"]
        entry["amdahl_ceiling_attention_only"] = float(1 / (1 - f)) if f < 1 else float("inf")
        res["contexts"][str(T)] = entry
        print(json.dumps({"T": T, "prefill_s": round(t_pre, 1), "dense_attn_share": round(f, 3),
                          "amdahl_ceiling": round(entry["amdahl_ceiling_attention_only"], 3),
                          **{m: (round(v["speedup_vs_dense_total"], 3), [round(x, 3) for x in v["speedup_vs_dense_total_ci95"]],
                                 round(v["pages_read_frac"], 3)) for m, v in entry["methods"].items()}}), flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(res, indent=1) + "\n")
        del cache0, pre


if __name__ == "__main__":
    main()
