#!/usr/bin/env python3
"""C3 exploratory: batch-1 decode attention cost by SDPA path, on the C3 T4 (NOT pre-registered).

Version 1.0, 2026-11-10. Question: is the attention-core share at C* (stock HF SDPA path on a
StaticCache: explicit boolean mask over max_cache_len and repeat_kv expansion of GQA heads)
a property of the bytes, or of this kernel path? Same device, same torch, same shapes as the
C3 models; no model weights are loaded. Each variant: one layer's decode attention (q_len = 1)
captured in a CUDA graph and replayed; the per-step cost is reported as layers x per-layer time.
Variants (all fp16):
  c3_path      : repeat_kv to all heads + boolean mask over T+16 slots (what C* executed)
  repeat_nomask: repeat_kv + exactly T keys, no mask
  gqa_nomask   : native GQA (enable_gqa=True), exactly T keys, no mask
  each also with the backend forced to EFFICIENT_ATTENTION and to MATH (errors recorded).
Output values are checked against an fp32 reference of the same inputs (max abs error).
Effective KV bandwidth = K+V bytes actually needed (T keys, n_kv heads) / per-step time.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SHAPES = {"qwen05": {"H": 14, "kv": 2, "d": 64, "L": 24, "T": [4096, 8192, 16384, 32000]},
          "granite4micro": {"H": 40, "kv": 8, "d": 64, "L": 40, "T": [4096, 8192, 16384]}}
OUT = Path("/kaggle/working/c3_attn_microbench.json") if Path("/kaggle").exists() else Path("c3_attn_microbench.json")


def repeat_kv(x, n):
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


def timed_graph(fn, iters=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], out


@torch.no_grad()
def main():
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    res = {"label": "EXPLORATORY (not pre-registered); attention-only micro-benchmark", "version": "1.0",
           "torch": torch.__version__, "device": torch.cuda.get_device_name(dev), "rows": []}
    for m, s in SHAPES.items():
        H, kv, d, L = s["H"], s["kv"], s["d"], s["L"]
        for T in s["T"]:
            S = T + 16
            q = torch.randn(1, H, 1, d, device=dev, dtype=torch.float16)
            k = torch.randn(1, kv, S, d, device=dev, dtype=torch.float16)
            v = torch.randn(1, kv, S, d, device=dev, dtype=torch.float16)
            mask = torch.zeros(1, 1, 1, S, dtype=torch.bool, device=dev); mask[..., :T] = True
            kt, vt = k[:, :, :T].contiguous(), v[:, :, :T].contiguous()
            ref = F.scaled_dot_product_attention(q.float(), repeat_kv(kt.float(), H // kv), repeat_kv(vt.float(), H // kv))
            variants = {
                "c3_path": lambda: F.scaled_dot_product_attention(q, repeat_kv(k, H // kv), repeat_kv(v, H // kv), attn_mask=mask),
                "repeat_nomask": lambda: F.scaled_dot_product_attention(q, repeat_kv(kt, H // kv), repeat_kv(vt, H // kv)),
                "gqa_nomask": lambda: F.scaled_dot_product_attention(q, kt, vt, enable_gqa=True),
            }
            kv_bytes = 2 * kv * T * d * 2  # per layer, the K/V actually needed
            for name, fn in variants.items():
                for be in ("default", "efficient", "math"):
                    row = {"model": m, "T": T, "variant": name, "backend": be}
                    try:
                        if be == "default":
                            t, o = timed_graph(fn)
                        else:
                            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION if be == "efficient" else SDPBackend.MATH]):
                                t, o = timed_graph(fn)
                        row.update({"per_layer_us": t * 1e6, "per_step_ms": t * L * 1e3, "kv_gbps": kv_bytes / t / 1e9,
                                    "max_abs_err_vs_fp32": float((o.float() - ref).abs().max())})
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    res["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    OUT.write_text(json.dumps(res, indent=1) + "\n")
            del q, k, v, kt, vt, mask, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
