#!/usr/bin/env python3
"""C3 exploratory robustness checks on the run-2 decision (NOT pre-registered). Version 1.0, 2026-11-10.

Reads results/c3/main_run2/decision.json (analyzer output), the run-1 files and the attention
micro-benchmark; writes results/c3/main_run2/sensitivity_exploratory.json.
1. Roofline-projected shares at C*: each class's derived bytes / measured device copy bandwidth,
   i.e. the share it would have if every class ran at the measured copy bandwidth (other = 0).
   [Derived] This is rule 2's view expressed as time; it does not alter any rule.
2. Attention efficiency at C*: measured attention removal time vs K/V bytes, as GB/s and as a
   fraction of copy bandwidth; plus the best stock SDPA variant from the micro-benchmark.
3. Reproducibility: run-1 vs run-2 eager p50 for units valid in both (run 1's eager path had no
   autograd defect).
4. Clock/temperature/power snapshots after each context (throttling disclosure).
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results" / "c3"
CLASSES = ("attn_core", "attn_proj", "mlp", "lm_head")


def main():
    dec = json.loads((R / "main_run2" / "decision.json").read_text())
    out = {"label": "EXPLORATORY: does not alter the charter §5 rule", "version": "1.0", "date": "2026-11-10"}
    mb = json.loads((R / "microbench" / "c3_attn_microbench.json").read_text()) if (R / "microbench" / "c3_attn_microbench.json").exists() else None
    proj, attn = {}, {}
    for m in ("qwen05", "granite4micro"):
        d = json.loads((R / "main_run2" / f"c3_main_{m}.json").read_text())
        bw = d["bandwidth"]["copy"]["gbps_median"] * 1e9
        cfg = json.loads((R / "model_configs" / m / "config.json").read_text())
        import c3_analyze as A
        A.recompute_bytes(d, A.kv_bytes_per_slot(cfg))
        proj[m], attn[m] = {}, {}
        for T, ctx in dec["models"][m]["contexts"].items():
            cs = ctx["C_star"]
            if not cs or cs == "BLOCKED":
                continue
            fmt, mode = cs.split("/")
            c = d["formats"][fmt]["contexts"][T]
            b = c["bytes_derived"]
            t_full = c["modes"][mode]["full"]["p50_s"]
            byte_time = {k: (b["kv_read"] if k == "attn_core" else b["weights_by_class"].get(k, 0)) / bw for k in CLASSES}
            tot = sum(byte_time.values())
            proj[m][T] = {"C_star": cs, "roofline_step_ms": tot * 1e3, "measured_step_ms": t_full * 1e3,
                          "measured_over_roofline": t_full / tot, "projected_share": {k: v / tot for k, v in byte_time.items()}}
            t_attn = c["modes"][mode]["share_removed"]["attn_core"] * t_full
            best = None
            if mb:
                rows = [r for r in mb["rows"] if r["model"] == m and r["T"] == int(T) and "per_step_ms" in r]
                if rows:
                    r0 = min(rows, key=lambda r: r["per_step_ms"])
                    best = {"variant": r0["variant"], "backend": r0["backend"], "per_step_ms": r0["per_step_ms"],
                            "c3_path_default_ms": next((r["per_step_ms"] for r in rows if r["variant"] == "c3_path" and r["backend"] == "default"), None)}
            attn[m][T] = {"attn_removed_ms": t_attn * 1e3, "kv_bytes": b["kv_read"], "attn_gbps": b["kv_read"] / t_attn / 1e9 if t_attn > 0 else None,
                          "fraction_of_copy_bw": (b["kv_read"] / t_attn) / bw if t_attn > 0 else None,
                          "kv_at_copy_bw_ms": b["kv_read"] / bw * 1e3, "microbench_best_stock": best}
    out["roofline_projection_at_C_star"] = proj
    out["projected_time_eligible"] = {k: {m: any(v["projected_share"][k] >= 0.5 for v in proj[m].values()) for m in proj} for k in CLASSES}
    out["attention_efficiency_at_C_star"] = attn
    # 3. run-1 vs run-2 eager reproducibility
    rep = {}
    for m in ("qwen05", "granite4micro"):
        d1 = json.loads((R / "main_run1" / f"c3_main_{m}.json").read_text())
        d2 = json.loads((R / "main_run2" / f"c3_main_{m}.json").read_text())
        for fmt, f1 in d1["formats"].items():
            for T, c1 in f1.get("contexts", {}).items():
                e1 = c1.get("modes", {}).get("eager", {}).get("full")
                e2 = d2["formats"].get(fmt, {}).get("contexts", {}).get(T, {}).get("modes", {}).get("eager", {}).get("full")
                if e1 and e2:
                    rep[f"{m}/{fmt}/{T}"] = {"run1_ms": e1["p50_s"] * 1e3, "run2_ms": e2["p50_s"] * 1e3, "ratio_run2_over_run1": e2["p50_s"] / e1["p50_s"]}
    out["eager_reproducibility_run1_vs_run2"] = rep
    rs = [v["ratio_run2_over_run1"] for v in rep.values()]
    out["eager_reproducibility_range"] = [min(rs), max(rs)] if rs else None
    # 4. clocks
    clk = {}
    for m in ("qwen05", "granite4micro"):
        d2 = json.loads((R / "main_run2" / f"c3_main_{m}.json").read_text())
        for fmt, f in d2["formats"].items():
            for T, c in f.get("contexts", {}).items():
                clk[f"{m}/{fmt}/{T}"] = (c.get("nvsmi_after") or "").split("\n")[0]
    out["nvsmi_gpu0_after_each_context"] = clk
    (R / "main_run2" / "sensitivity_exploratory.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"projected_time_eligible": out["projected_time_eligible"], "eager_reproducibility_range": out["eager_reproducibility_range"]}, indent=1))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
