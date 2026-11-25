#!/usr/bin/env python3
"""C2 sensitivity and robustness checks on the pre-registered decision (EXPLORATORY).

Version 1.0, 2026-11-10. Reads results/c2/decision.json and the breakdown files; writes
results/c2/sensitivity_exploratory.json. Nothing here changes the charter §5 rule or its
thresholds; it tests how fragile the rule's output is.

1. C* run choice: dense bf16 was measured in two separate runs per model (breakdown run and B1
   run). The analyzer takes the faster one. Here eligibility is recomputed for every pairing
   of runs (Qwen run x Granite run).
2. Amdahl ceiling: 1 / (1 - s_up) per class at C*, i.e. the end-to-end speedup if the class
   cost nothing. [Derived from measured shares]
3. Weight-only quantization byte shares: the MLP byte share at C* if weights were int8
   (1 B/param + one fp32 scale per output row) or int4 (group 128, bf16 scale + zero:
   0.5 + 4/128 B/param), with lm_head quantized or kept in bf16, KV in bf16.
   [Derived] These configurations were NOT run with quality checks; on this CPU, int8
   weight-only was slower than dense, so it could not be C*. The point is only whether
   byte-eligibility survives a stronger weight baseline on a device whose kernels make it fast.
4. Qwen fp32: interleaved run 1 (OOM-killed at 32K) vs the sequential rerun at 4K/16K.
5. Hook overhead above the 5% limit of charter §4.
6. Effective bandwidth of C* as a fraction of the measured triad.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results" / "c2"
CLASSES = ["attn_core", "attn_proj", "mlp", "lm_head", "other"]
DENSE_RUNS = {"qwen05": ["breakdown_qwen05_bf16.json", "breakdown_b1_qwen05_bf16.json"],
              "granite4micro": ["breakdown_granite4micro_bf16.json", "breakdown_b1_granite4micro_bf16.json"]}


def j(name):
    return json.loads((R / name).read_text())


def s_up(ve):
    o = max(ve["hook_overhead_frac"], 0.0)
    return {c: min(1.0, ve["time_share_median"][c] * (1 + o)) for c in CLASSES}


def per_context_dense(f):
    d = j(f)
    return {int(T): e["variants"]["dense"] for T, e in d["contexts"].items()}, d


def main():
    dec = j("decision.json")
    out = {"label": "EXPLORATORY: does not alter the charter §5 rule", "version": "1.0", "date": "2026-11-10"}

    # 1. run-choice pairing. C* is dense bf16 at every context in decision.json (checked below).
    for m, mr in dec["models"].items():
        assert all(v["C_star"] == "dense_bf16" for v in mr["contexts"].values()), (m, "C* is not dense_bf16 everywhere")
    pairing = []
    runs = {m: [per_context_dense(f)[0] for f in fs] for m, fs in DENSE_RUNS.items()}
    for iq, ig in itertools.product(range(2), range(2)):
        sel = {"qwen05": runs["qwen05"][iq], "granite4micro": runs["granite4micro"][ig]}
        maxs = {m: {c: max(s_up(ve)[c] for ve in sel[m].values()) for c in CLASSES} for m in sel}
        elig = {c: all(maxs[m][c] >= 0.5 for m in maxs) for c in CLASSES}
        pairing.append({"qwen_run": DENSE_RUNS["qwen05"][iq], "granite_run": DENSE_RUNS["granite4micro"][ig],
                        "max_s_up": maxs, "time_eligible": elig})
    out["run_choice_pairings"] = pairing
    out["time_eligible_any_pairing"] = {c: any(p["time_eligible"][c] for p in pairing) for c in CLASSES}

    # 2. Amdahl ceilings at the analyzer's C*
    amd = {}
    for m, mr in dec["models"].items():
        amd[m] = {}
        for T, v in mr["contexts"].items():
            amd[m][T] = {c: (1.0 / (1.0 - v["C_star_s_up"][c]) if v["C_star_s_up"][c] < 1 else None) for c in CLASSES}
    out["amdahl_ceiling_at_C_star"] = amd

    # 3. weight-only quantization byte shares (derived)
    def wq_shares(b, wfac, lm_quant):
        w = dict(b["weights_by_class"])
        mlp, ap, lm = w["mlp"] * wfac, w["attn_proj"] * wfac, w["lm_head"] * (wfac if lm_quant else 1.0)
        tot = mlp + ap + lm + b["kv_read"]
        return {"mlp": mlp / tot, "kv": b["kv_read"] / tot, "lm_head": lm / tot}
    wq = {}
    for m, fs in DENSE_RUNS.items():
        dense, _ = per_context_dense(fs[0])
        wq[m] = {}
        for T, ve in sorted(dense.items()):
            b = ve["bytes_derived"]
            wq[m][str(T)] = {"bf16_dense": wq_shares(b, 1.0, True),
                             "int8wo_all": wq_shares(b, 0.5, True), "int8wo_lm_bf16": wq_shares(b, 0.5, False),
                             "int4g128_all": wq_shares(b, (0.5 + 4 / 128) / 2, True),
                             "int4g128_lm_bf16": wq_shares(b, (0.5 + 4 / 128) / 2, False)}
    # int8 factor 0.5 ignores the per-row scale (< 0.5% of weight bytes at these widths); stated.
    out["weight_only_quant_byte_shares_derived"] = wq
    out["mlp_byte_eligible_under_quant"] = {
        cfg: {m: any(v[cfg]["mlp"] >= 0.5 for v in wq[m].values()) for m in wq}
        for cfg in ("bf16_dense", "int8wo_all", "int8wo_lm_bf16", "int4g128_all", "int4g128_lm_bf16")}

    # 4. Qwen fp32 interleaved (run 1) vs sequential (rerun)
    r1, r2 = j("breakdown_b1_qwen05_fp32_run1_oomkilled.json"), j("breakdown_b1_qwen05_fp32.json")
    cmp = {}
    for T in r1["contexts"]:
        cmp[T] = {v: {"run1_interleaved_ms": r1["contexts"][T]["variants"][v]["steady_step_s_p50"] * 1e3,
                      "rerun_sequential_ms": r2["contexts"][T]["variants"][v]["steady_step_s_p50"] * 1e3,
                      "ratio_rerun_over_run1": r2["contexts"][T]["variants"][v]["steady_step_s_p50"] / r1["contexts"][T]["variants"][v]["steady_step_s_p50"]}
                  for v in r2["contexts"][T]["variants"] if v in r1["contexts"][T]["variants"]}
    out["qwen_fp32_interleaved_vs_sequential"] = cmp

    # 5. hook overhead above 5%, 6. effective bandwidth vs triad
    over, bw = [], {}
    for m, mr in dec["models"].items():
        for T, v in mr["contexts"].items():
            for c in v["candidates"]:
                if c["hook_overhead"] > 0.05:
                    over.append({"model": m, "T": int(T), "config": c["config"], "file": c["file"], "hook_overhead": c["hook_overhead"]})
    for m, fs in DENSE_RUNS.items():
        for f in fs:
            d = j(f)
            tri = [x["gbps_median"] for x in (d.get("triad_before"), d.get("triad_after")) if x]
            for T, e in d["contexts"].items():
                g = e["variants"]["dense"]["effective_gbps_derived"]
                bw.setdefault(m, []).append({"file": f, "T": int(T), "effective_gbps": g, "triad_gbps": tri,
                                             "fraction_of_min_triad": g / min(tri) if tri else None})
    out["hook_overhead_above_5pct"] = over
    out["dense_effective_bandwidth_vs_triad"] = bw
    (R / "sensitivity_exploratory.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"time_eligible_any_pairing": out["time_eligible_any_pairing"],
                      "mlp_byte_eligible_under_quant": out["mlp_byte_eligible_under_quant"],
                      "n_hook_overhead_above_5pct": len(over)}, indent=1))


if __name__ == "__main__":
    main()
