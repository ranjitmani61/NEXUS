#!/usr/bin/env python3
"""G5-R aggregation of trace / quality / timing / passkey results into gate tables.

Version 1.0, 2026-11-04. Reads results/g5r/*.json (+ raw per-record npz for
per-layer breakdowns if --raw-dir given). Writes results/g5r/summary.json and
results/g5r/summary_tables.md. Uncertainty: bootstrap over BOOKS (cluster
bootstrap; tokens within a book are correlated), 4000 resamples, seed 0.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

R = Path("results/g5r")
NX = ["box", "sph", "val", "box16"]


def cluster_boot(values_by_book, stat=np.mean, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    books = [np.asarray(v, float) for v in values_by_book]
    k = len(books)
    if k < 2:
        return None
    out = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        out.append(stat(np.concatenate([books[i] for i in idx])))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def quality_tables(files):
    rows = []
    for f in sorted(files):
        d = json.load(open(f))
        meta, books = d["meta"], d["books"]
        if not books:
            continue
        kind = meta.get("input_kind", "book")
        r = {"file": Path(f).name, "model": meta["model"], "dtype": meta["dtype"], "T": meta["context"],
             "input_kind": kind, "n_books": len(books), "steps": meta["steps"],
             "dense_ppl_decode": float(np.exp(np.mean(np.concatenate([b["dense_nll_per_token"] for b in books])))),
             "head_steps": int(sum(b["shadow"]["n_head_steps"] for b in books))}
        for n in NX:
            fr_mean = [b["shadow"][f"{n}_page_frac"]["mean"] for b in books]
            full = [b["shadow"][f"{n}_page_frac"]["share_heads_full_read"] for b in books]
            le1 = [b["shadow"][f"{n}_page_frac"]["share_heads_le_1page"] for b in books]
            r[f"{n}_frac_mean"] = float(np.mean(fr_mean)); r[f"{n}_frac_mean_minbook"] = float(np.min(fr_mean))
            r[f"{n}_share_full_read"] = float(np.mean(full)); r[f"{n}_share_le1"] = float(np.mean(le1))
            r[f"{n}_violations"] = int(sum(b["shadow"][f"{n}_violations"] for b in books))
            if f"{n}_bound_violations" in books[0]["shadow"]:
                r[f"{n}_bound_violations"] = int(sum(b["shadow"][f"{n}_bound_violations"] for b in books))
            if f"{n}_naive_eval_stops_earlier_heads" in books[0]["shadow"]:
                r[f"{n}_naive_false_early"] = int(sum(b["shadow"][f"{n}_naive_eval_stops_earlier_heads"] for b in books))
        for n in ("ormass", "orscore", "ormass16"):
            r[f"{n}_frac_mean"] = float(np.mean([b["shadow"][f"{n}_page_frac"]["mean"] for b in books]))
            r[f"{n}_frac_median_of_books"] = float(np.median([b["shadow"][f"{n}_page_frac"]["median"] for b in books]))
        for B in (512, 1024, 2048, 4096):
            r[f"quest{B}_local_err_share_gt_tol"] = float(np.mean([b["shadow"][f"quest{B}_err"]["share_gt_tol"] for b in books]))
            r[f"quest{B}_local_err_p95"] = float(np.median([b["shadow"][f"quest{B}_err"]["p95"] for b in books]))
        r["capture_maxabs_vs_sdpa"] = float(max(b["shadow"]["capture_maxabs_vs_sdpa"] for b in books))
        live = {}
        methods = list(books[0]["live"].keys())
        for m in methods:
            dn = [np.array(b["live"][m]["nll_per_token"]) - np.array(b["dense_nll_per_token"]) for b in books]
            dense_all = np.concatenate([b["dense_nll_per_token"] for b in books])
            m_all = np.concatenate([b["live"][m]["nll_per_token"] for b in books])
            ppl_ratio = float(np.exp(m_all.mean() - dense_all.mean()))
            ci = cluster_boot(dn)
            live[m] = {"ppl_ratio": ppl_ratio,
                       "ppl_ratio_ci95_bookboot": [float(np.exp(c)) for c in ci] if ci else None,
                       "delta_nll_mean": float(np.mean(np.concatenate(dn))),
                       "kl_mean": float(np.mean([b["live"][m]["kl_mean"] for b in books])),
                       "kl_max": float(np.max([b["live"][m]["kl_max"] for b in books])),
                       "top1_agree": float(np.mean([b["live"][m]["top1_agree"] for b in books])),
                       "pages_frac_mean": float(np.mean([b["live"][m]["pages_frac_mean"] for b in books])),
                       "live_violations": int(sum(b["live"][m]["violations"] for b in books)),
                       "matched_quality_ppl_top1": bool(ppl_ratio <= 1.01 and np.mean([b["live"][m]["top1_agree"] for b in books]) >= 0.98)}
        r["live"] = live
        rows.append(r)
    rows.sort(key=lambda r: (r["model"], r["input_kind"] != "book", r["input_kind"], r["T"]))
    return rows


def timing_tables(files):
    out = []
    for f in sorted(files):
        d = json.load(open(f))
        for T, e in d["contexts"].items():
            row = {"file": Path(f).name, "model": d["meta"]["model"], "dtype": d["meta"]["dtype"], "batch": d["meta"]["batch"],
                   "T": int(T), "prefill_s": e["prefill_s"], "kv_bytes": e["kv_bytes"],
                   "dense_attn_share": e["methods"]["dense"]["attn_share_of_decode"],
                   "amdahl_ceiling": e["amdahl_ceiling_attention_only"], "methods": {}}
            for m, v in e["methods"].items():
                row["methods"][m] = {k: v[k] for k in ("tokens_per_s", "speedup_vs_dense_total", "speedup_vs_dense_total_ci95",
                                                         "steady_step_s_p50", "steady_step_s_p95", "first_step_s_median",
                                                         "pages_read_frac", "fallbacks_median", "build_s_median",
                                                         "meta_bytes", "rss_bytes_max")}
            out.append(row)
    out.sort(key=lambda r: (r["model"], r["batch"], r["T"]))
    return out


def passkey_tables(files):
    out = []
    for f in sorted(files):
        d = json.load(open(f))
        out.append({"file": Path(f).name, "model": d["meta"]["model"], "T": d["meta"]["context"],
                    "n_trials": d["n_trials"], "accuracy": d["accuracy"]})
    out.sort(key=lambda r: (r["model"], r["T"]))
    return out


def family_ceiling(quality, timing):
    """[Derived] Zero-overhead end-to-end ceiling for ANY valid-bound mass-certificate method:
    1 / (1 - f * (1 - r)), f = measured dense attention share of decode time (timing run),
    r = measured OR-SCORE mean page fraction (trace run) at the same model and context.
    Optimistic for the candidates: it ignores all screening overhead and uses the per-query-head
    fraction, which is <= the per-KV-head union that actually sets traffic."""
    q_by = {(r["model"].replace("qwen05", "qwen05"), r["T"]): r for r in quality if r["input_kind"] == "book"}
    out = []
    for t in timing:
        key = (t["model"], t["T"] if t["T"] != 32000 else 32000)
        qr = q_by.get(key)
        if qr is None:
            continue
        f = t["dense_attn_share"]; r = qr["orscore_frac_mean"]; rb = qr["box_frac_mean"]
        out.append({"model": t["model"], "batch": t["batch"], "T": t["T"], "f_attn_share": f,
                    "r_orscore": r, "family_ceiling": 1 / (1 - f * (1 - r)),
                    "r_nxbox_measured": rb, "nxbox_ceiling_zero_overhead": 1 / (1 - f * (1 - rb)),
                    "amdahl_ceiling": t["amdahl_ceiling"],
                    "nxbox_measured_speedup": t["methods"].get("nx_box", {}).get("speedup_vs_dense_total"),
                    "r_needed_for_2x": max(0.0, 1 - 0.5 / f) if f > 0.5 else None})
    return out


def fmt(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def markdown(summary):
    L = ["# G5-R summary tables (generated by tests/g5r_analyze.py)", ""]
    L += ["## Shadow traces: pages each method must read (mean fraction over all layer×head×step)", "",
          "| model | T | input | books | head-steps | NX-BOX | NX-SPH | NX-VAL | NX-BOX@16 | share NX-BOX reads ALL | OR-SCORE (best possible, any valid-bound mass cert) | OR-MASS | NX violations (box/sph/val) | naive-eval false early stops (box/sph) |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in summary["quality"]:
        L.append(f"| {r['model']} | {r['T']} | {r['input_kind']} | {r['n_books']} | {r['head_steps']} | {fmt(r['box_frac_mean'])} | {fmt(r['sph_frac_mean'])} | {fmt(r['val_frac_mean'])} | {fmt(r['box16_frac_mean'])} | {fmt(r['box_share_full_read'])} | {fmt(r['orscore_frac_mean'])} | {fmt(r['ormass_frac_mean'])} | {r['box_violations']}/{r['sph_violations']}/{r['val_violations']} | {r.get('box_naive_false_early')}/{r.get('sph_naive_false_early')} |")
    L += ["", "## Live end-to-end quality vs dense (teacher-forced decode; book-cluster bootstrap 95% CI)", "",
          "| model | T | input | method | PPL ratio | 95% CI | mean KL | max KL | top-1 agree | KV pages read | matched quality (PPL≤1.01, top1≥0.98) |",
          "|---|---:|---|---|---:|---|---:|---:|---:|---:|---|"]
    for r in summary["quality"]:
        for m, v in r["live"].items():
            ci = v["ppl_ratio_ci95_bookboot"]
            L.append(f"| {r['model']} | {r['T']} | {r['input_kind']} | {m} | {v['ppl_ratio']:.4f} | {('[%.4f, %.4f]' % tuple(ci)) if ci else '—'} | {v['kl_mean']:.5f} | {v['kl_max']:.4f} | {v['top1_agree']:.4f} | {v['pages_frac_mean']:.3f} | {v['matched_quality_ppl_top1']} |")
    L += ["", "## Quest local (per-head) error > 0.02 share (shadow)", "", "| model | T | input | B=512 | B=1024 | B=2048 | B=4096 |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in summary["quality"]:
        L.append(f"| {r['model']} | {r['T']} | {r['input_kind']} | " + " | ".join(fmt(r[f'quest{B}_local_err_share_gt_tol']) for B in (512, 1024, 2048, 4096)) + " |")
    L += ["", "## Measured CPU decode speed (bf16 unless noted; 16 steps × 5 reps; speedup = dense time / method time incl. metadata+fallback)", "",
          "| model | batch | T | dense attn share of decode | Amdahl ceiling (attention-only) | method | tokens/s | speedup vs dense | 95% CI | KV pages read |",
          "|---|---:|---:|---:|---:|---|---:|---:|---|---:|"]
    for r in summary["timing"]:
        for m, v in r["methods"].items():
            ci = v["speedup_vs_dense_total_ci95"]
            L.append(f"| {r['model']} | {r['batch']} | {r['T']} | {r['dense_attn_share']:.3f} | {r['amdahl_ceiling']:.3f} | {m} | {v['tokens_per_s']:.2f} | {v['speedup_vs_dense_total']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {v['pages_read_frac']:.3f} |")
    L += ["", "## [Derived] Ceiling for the whole NX valid-bound mass-certificate family (zero overhead)", "",
          "| model | batch | T | attention share f | OR-SCORE r | family ceiling 1/(1-f(1-r)) | page fraction needed for 2x | NX-BOX zero-overhead ceiling | NX-BOX measured |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in summary.get("family_ceiling", []):
        need = "impossible (f<=0.5)" if c["r_needed_for_2x"] is None else f"<= {c['r_needed_for_2x']:.3f}"
        L.append(f"| {c['model']} | {c['batch']} | {c['T']} | {c['f_attn_share']:.3f} | {c['r_orscore']:.3f} | {c['family_ceiling']:.3f} | {need} | {c['nxbox_ceiling_zero_overhead']:.3f} | {fmt(c['nxbox_measured_speedup'])} |")
    L += ["", "## Passkey retrieval accuracy (question+answer decoded with the method)", "", "| model | T | trials | " + " | ".join(["dense", "nx_box", "quest:1024", "quest:2048", "quest:4096"]) + " |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["passkey"]:
        L.append(f"| {r['model']} | {r['T']} | {r['n_trials']} | " + " | ".join(fmt(r["accuracy"].get(m)) for m in ["dense", "nx_box", "quest:1024", "quest:2048", "quest:4096"]) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=R)
    a = ap.parse_args()
    s = {"version": "1.0", "date": "2026-11-04",
         "quality": quality_tables(glob.glob(str(a.results / "quality_*.json"))),
         "timing": timing_tables(glob.glob(str(a.results / "timing_*.json"))),
         "passkey": passkey_tables(glob.glob(str(a.results / "passkey_*.json")))}
    s["family_ceiling"] = family_ceiling(s["quality"], s["timing"])
    (a.results / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (a.results / "summary_tables.md").write_text(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main()
