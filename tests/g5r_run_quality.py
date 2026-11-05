#!/usr/bin/env python3
"""G5-R trace + end-to-end quality driver (real pretrained models, CPU).

Version 1.0, 2026-11-04. Contract: specs/G5R_real_model_benchmark_contract.md.

Per (model, context T, PG-19 book):
  1. dense prefill of the first T tokens (stock SDPA)                        [shared]
  2. SHADOW decode of the next `steps` true tokens: model output is dense; every
     candidate/oracle is evaluated on the real q/K/V of every layer/head/step
  3. LIVE decode per method from a copy of the same prefill cache: candidate
     attention replaces dense attention in every layer at every decode step;
     NLL of the true next token, KL(dense || method) and top-1 agreement are
     computed against the dense run on identical inputs (teacher forcing).

Raw per-record traces go to --raw-dir (large, regenerable); a JSON summary goes to
--output. All numbers are CPU software measurements of model outputs, never
hardware speedups.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_host  # noqa: E402
import g5r_hf as H  # noqa: E402
import g5r_cache as KC  # noqa: E402


def load(model_dir: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    H.register()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, attn_implementation="g5r")
    model.eval()
    return tok, model


def book_ids(tok, path: Path, need: int):
    text = path.read_text(encoding="utf-8", errors="strict").replace("\r\n", "\n")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < need:
        raise ValueError(f"{path.name}: {len(ids)} tokens < {need}")
    return torch.tensor(ids[:need]), hashlib.sha256(text.encode()).hexdigest(), len(ids)


@torch.no_grad()
def decode(model, cache, tokens, mode, method=""):
    H.CTL.mode, H.CTL.method = mode, method
    H.CTL.live_stats = []
    lps = []
    for i in range(len(tokens) - 1):
        H.CTL.step = i
        out = model(input_ids=tokens[i].view(1, 1), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        lps.append(torch.log_softmax(out.logits[0, -1].double(), -1))
    H.CTL.mode = "dense"
    return torch.stack(lps), cache, list(H.CTL.live_stats)


def summarize_records(recs):
    out = {}
    def cat(key):
        return np.concatenate([np.atleast_1d(r[key]).astype(float) for r in recs])
    full32 = np.concatenate([np.full(np.atleast_1d(r["box_k"]).shape, r["full32"], float) for r in recs])
    full16 = np.concatenate([np.full(np.atleast_1d(r["box16_k"]).shape, r["full16"], float) for r in recs])
    for name, full in (("box", full32), ("sph", full32), ("val", full32), ("box16", full16)):
        frac = cat(f"{name}_k") / np.maximum(full, 1)
        out[f"{name}_page_frac"] = {"median": float(np.median(frac)), "mean": float(frac.mean()),
                                    "p05": float(np.quantile(frac, .05)), "min": float(frac.min()),
                                    "share_heads_full_read": float(np.mean(frac >= 1 - 1e-12)),
                                    "share_heads_le_1page": float(np.mean(cat(f"{name}_k") <= 1))}
        e = cat(f"{name}_err")
        out[f"{name}_err"] = {"median": float(np.median(e)), "max": float(e.max())}
        out[f"{name}_violations"] = int(sum(r[f"{name}_viol"] for r in recs))
        if f"{name}_boundviol" in recs[0]:
            out[f"{name}_bound_violations"] = int(sum(r[f"{name}_boundviol"] for r in recs))
        if f"{name}_naive_early" in recs[0]:
            out[f"{name}_naive_eval_stops_earlier_heads"] = int(sum(r[f"{name}_naive_early"] for r in recs))
    for name, full in (("ormass", full32), ("orscore", full32), ("ormass16", full16)):
        frac = cat(f"{name}_k") / np.maximum(full, 1)
        out[f"{name}_page_frac"] = {"median": float(np.median(frac)), "mean": float(frac.mean()),
                                    "p05": float(np.quantile(frac, .05)), "min": float(frac.min())}
    for B in H.QUEST_BUDGETS:
        e = cat(f"quest{B}_err"); er = cat(f"quest{B}_errrel")
        out[f"quest{B}_err"] = {"median": float(np.median(e)), "p95": float(np.quantile(e, .95)), "max": float(e.max()),
                                "share_gt_tol": float(np.mean(e > H.C.TOL))}
        out[f"quest{B}_errrel"] = {"median": float(np.median(er)), "p95": float(np.quantile(er, .95))}
    out["capture_maxabs_vs_sdpa"] = float(max(r["capture_maxabs"] for r in recs))
    out["capture_max_outnorm"] = float(max(r["capture_outnorm"] for r in recs))
    tm = cat("top1_page_mass"); p0 = cat("page0_mass")
    out["top1_page_mass_median"] = float(np.median(tm)); out["page0_mass_median"] = float(np.median(p0))
    out["n_head_steps"] = int(full32.size)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--dtype", default="fp32"); ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--books", nargs="+", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--live", nargs="*", default=["nx_box", "nx_sph", "nx_one", "nx_val", "quest:512", "quest:1024", "quest:2048", "quest:4096"])
    ap.add_argument("--capture-prob", type=float, default=0.004)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--input-kind", default="book", choices=["book", "random", "repeat"],
                    help="book: PG-19 text; random: uniform random token ids (anti-structure control); "
                         "repeat: first 256 book tokens tiled (highly repetitive control)")
    ap.add_argument("--raw-dir", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.dtype != "fp32" and "dense64" not in a.live:
        a.live = list(a.live) + ["dense64"]   # precision control for reduced-precision models
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    tok, model = load(a.model_dir, a.dtype)
    import transformers
    meta = {"version": "1.0", "input_kind": a.input_kind, "host": g5r_host.host_info(), "date": "2026-11-04", "model": a.model_name, "model_dir": a.model_dir,
            "dtype": a.dtype, "context": a.context, "steps": a.steps, "live_methods": a.live,
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "threads": a.threads, "page_size_nx": H.CTL.ps,
            "tol": H.C.TOL, "quest_budgets": list(H.QUEST_BUDGETS), "quest_dense_layers": H.QUEST_DENSE_LAYERS,
            "contract": "specs/G5R_real_model_benchmark_contract.md",
            "note": "Shadow statistics use float64 recomputation of the real q/K/V; live runs replace attention."}
    per_book = []
    a.raw_dir.mkdir(parents=True, exist_ok=True)
    for bi, book in enumerate(a.books):
        need = a.context + a.steps + 1
        ids, text_sha, n_tok = book_ids(tok, book, max(need, 300) if a.input_kind == "book" else 300)
        if a.input_kind == "random":
            g = torch.Generator().manual_seed(20261104 + bi)
            ids = torch.randint(0, len(tok), (need,), generator=g)
            text_sha = f"random-token-ids seed {20261104 + bi}"
        elif a.input_kind == "repeat":
            ids = ids[:256].repeat(need // 256 + 1)[:need]
            text_sha = text_sha + " (first 256 tokens tiled)"
        prefix, cont = ids[: a.context], ids[a.context: a.context + a.steps + 1]
        H.CTL.mode = "dense"
        t0 = time.perf_counter()
        with torch.no_grad():
            pre = model(input_ids=prefix.view(1, -1), past_key_values=KC.make_cache(model, 1), use_cache=True, logits_to_keep=1)
        t_prefill = time.perf_counter() - t0
        cache0 = pre.past_key_values
        KC.require_len(cache0, a.context, "after prefill")
        # shadow
        H.CTL.records, H.CTL.capture_samples = [], []
        H.CTL.capture_prob = a.capture_prob
        H.CTL.rng = np.random.default_rng(20261104 + bi)
        t0 = time.perf_counter()
        lp_dense, _, _ = decode(model, copy.deepcopy(cache0), cont, "shadow")
        t_shadow = time.perf_counter() - t0
        recs = H.CTL.records
        bad = [r for r in recs if r["T"] != a.context + r["step"] + 1]
        if bad:
            raise RuntimeError(f"decode attention length guard failed: {len(bad)} records, e.g. T={bad[0]['T']} step={bad[0]['step']}")
        tgt = cont[1:]
        nll_dense = -lp_dense[torch.arange(len(tgt)), tgt]
        book_res = {"book": book.name, "text_sha256": text_sha, "book_tokens": n_tok,
                    "prefill_s": t_prefill, "shadow_s": t_shadow,
                    "dense_nll_mean": float(nll_dense.mean()), "dense_ppl": float(torch.exp(nll_dense.mean())),
                    "shadow": summarize_records(recs), "live": {}}
        # save raw records + captured samples
        keys = [k for k in recs[0] if isinstance(recs[0][k], np.ndarray)]
        np.savez_compressed(a.raw_dir / f"{a.model_name}_T{a.context}_{book.stem}_records.npz",
                            layer=np.array([r["layer"] for r in recs]), step=np.array([r["step"] for r in recs]),
                            T=np.array([r["T"] for r in recs]), **{k: np.stack([r[k] for r in recs]) for k in keys})
        cs = H.CTL.capture_samples
        if cs:
            d = {"n": len(cs)}
            for i, s in enumerate(cs):
                for k, v in s.items():
                    d[f"{k}{i}"] = v
            np.savez_compressed(a.raw_dir / f"{a.model_name}_T{a.context}_{book.stem}_samples.npz", **d)
        book_res["captured_samples"] = len(cs)
        H.CTL.records, H.CTL.capture_samples, H.CTL.capture_prob = [], [], 0.0
        for m in a.live:
            t0 = time.perf_counter()
            lp_m, _, stats = decode(model, copy.deepcopy(cache0), cont, "live", m)
            nll_m = -lp_m[torch.arange(len(tgt)), tgt]
            kl = (lp_dense.exp() * (lp_dense - lp_m)).sum(-1)
            agree = (lp_dense.argmax(-1) == lp_m.argmax(-1)).double()
            pf = np.array([s["pages_frac"] for s in stats])
            book_res["live"][m] = {"nll_mean": float(nll_m.mean()), "delta_nll_mean": float((nll_m - nll_dense).mean()),
                                   "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                                   "top1_agree": float(agree.mean()), "pages_frac_mean": float(pf.mean()),
                                   "violations": int(sum(s.get("viol", 0) for s in stats)),
                                   "admit_mean": float(np.mean([s["admit"] for s in stats])) if stats and "admit" in stats[0] else None,
                                   "nll_per_token": nll_m.tolist(), "seconds": time.perf_counter() - t0}
        book_res["dense_nll_per_token"] = nll_dense.tolist()
        per_book.append(book_res)
        print(json.dumps({"book": book.name, "prefill_s": round(t_prefill, 1), "shadow_s": round(t_shadow, 1),
                          "dense_ppl": round(book_res["dense_ppl"], 3),
                          "box_frac_med": book_res["shadow"]["box_page_frac"]["median"],
                          "ormass_frac_med": book_res["shadow"]["ormass_page_frac"]["median"],
                          "orscore_frac_med": book_res["shadow"]["orscore_page_frac"]["median"],
                          "viol": [book_res["shadow"][f"{n}_violations"] for n in ("box", "sph", "val")],
                          "live_dnll": {m: round(v["delta_nll_mean"], 5) for m, v in book_res["live"].items()}}), flush=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps({"meta": meta, "books": per_book}, indent=1) + "\n")


if __name__ == "__main__":
    main()
