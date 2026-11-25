#!/usr/bin/env python3
"""C3: batch-1 decode cost by measured removal, on GPU (charter specs/C3_G0_gpu_decode_bottleneck_charter.md).

Version 1.3, 2026-11-10 (A1: C3-D2, C3-D3; A2: C3-D4 no-grad capture, prefill chunk 512, memory/traceback records; C3-D5 KV layer count, after run 2). Self-contained (torch, transformers 4.57.1, numpy; bitsandbytes optional).
Also runs on CPU for logic tests (graph modes are then skipped and recorded as such).

Per model:
1. Reference: dense fp32. Teacher-forced full forward (no cache) on the quality books at
   --quality-T, scoring --quality-steps tokens. Granite fp32 is split over two GPUs when
   --ref-device-map auto is given (fp32 Granite does not fit one 15 GB T4).
2. Precondition: static-cache decode vs full forward (KL and argmax, dtype-aware thresholds).
3. For each weight format (fp16 dense, bnb nf4, bnb int8) and context T:
   chunked prefill (512-token chunks, A2) into a StaticCache (max_cache_len = T + steps), then teacher-forced decode
   steps timed in two modes, eager and one CUDA graph per configuration (GPU only).
   Configurations: full, and with one class removed (attention core / attention projections /
   MLP Linear / lm_head), and with all four removed. Removal replaces the class's work with a
   zero tensor of the right shape. Share s_X = 1 - t(without X) / t(full): the fraction of the
   step that removing X saves, measured in the same mode. [Measured]
   Graph mode counts only if graph logits match eager logits (argmax equal at every step,
   max KL <= 1e-5).
4. Quality of each weight format against the fp32 reference (same books, same tokens).
5. Derived bytes per step: Linear weight storage per class plus K/V read over max_cache_len.
Host-to-device copies are kept out of the timed region: all tokens and positions are staged on
the device before timing.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj", "input_linear", "output_linear")
REMOVABLE = ("attn_core", "attn_proj", "mlp", "lm_head")
ABLATIONS = ("full",) + tuple(f"no_{c}" for c in REMOVABLE) + ("no_all",)
STATE = {"remove_attn": False}
KL_LIMIT = {"fp32": 1e-6, "fp16": 1e-3, "bf16": 1e-2}
GRAPH_KL_LIMIT = 1e-5


def classify(name: str) -> str:
    leaf = name.split(".")[-1]
    if leaf == "lm_head":
        return "lm_head"
    if leaf in ATTN_PROJ:
        return "attn_proj"
    if leaf in MLP:
        return "mlp"
    return "linear_other"


def register_attention():
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def c3_attn(module, query, key, value, attention_mask, **kw):
        if STATE["remove_attn"]:
            b, h, q, d = query.shape
            return torch.zeros(b, q, h, d, dtype=query.dtype, device=query.device), None
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register("c3_sdpa", c3_attn)
    # Defect C3-D1 (found by the CPU logic test): transformers 4.57.1 builds NO attention mask
    # for an attention name missing from the mask interface. With a StaticCache, decode then
    # attends to the unfilled slots (fp32 KL 1.8e-2 vs full forward). Register the stock SDPA mask.
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionMaskInterface.register("c3_sdpa", sdpa_mask)


class Ablator:
    """Replaces the forward of Linear modules of a class by a zero-output stub, and toggles the
    attention-core stub. Restores the original forward exactly."""

    def __init__(self, model):
        self.mods = {"attn_proj": [], "mlp": [], "lm_head": []}
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):  # bnb Linear4bit / Linear8bitLt subclass nn.Linear
                c = classify(n)
                if c in self.mods:
                    self.mods[c].append(m)
        self._orig = {}

    @staticmethod
    def _stub(out):
        def f(x, *a, **k):
            return torch.zeros(*x.shape[:-1], out, dtype=x.dtype, device=x.device)
        return f

    def set(self, removed):
        self.restore()
        STATE["remove_attn"] = "attn_core" in removed
        for c in removed:
            for m in self.mods.get(c, []):
                self._orig[id(m)] = (m, m.__dict__.get("forward"))
                m.forward = self._stub(m.out_features)

    def restore(self):
        for m, f in self._orig.values():
            if f is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = f
        self._orig = {}
        STATE["remove_attn"] = False


def removed_set(ablation):
    if ablation == "full":
        return set()
    if ablation == "no_all":
        return set(REMOVABLE)
    return {ablation[3:]}


def make_static_cache(model, max_len):
    from transformers import StaticCache
    if getattr(model.config, "model_type", "") == "granitemoehybrid":
        class GraniteStaticCache(StaticCache):
            has_previous_state = False  # Granite-4.0-micro: all 40 layers are attention, no Mamba state
        return GraniteStaticCache(config=model.config, max_cache_len=max_len)
    return StaticCache(config=model.config, max_cache_len=max_len)


@torch.no_grad()
def prefill(model, cache, ids, dev, chunk=512):
    """Chunked prefill of ids (list) into the static cache; positions 0..len-1."""
    T = len(ids)
    x = torch.tensor([ids], device=dev)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        model(input_ids=x[:, s:e], past_key_values=cache, cache_position=torch.arange(s, e, device=dev), use_cache=True, logits_to_keep=1)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def nvsmi():
    try:
        r = subprocess.run("nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader",
                           shell=True, capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception as e:
        return f"unavailable: {e!r}"


def weight_bytes(model):
    out, seen = {}, set()
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        w = m.weight
        key = w.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        nb = w.numel() * w.element_size()  # packed storage for bnb 4-bit
        qs = getattr(w, "quant_state", None)
        if qs is not None and getattr(qs, "absmax", None) is not None:
            nb += qs.absmax.numel() * qs.absmax.element_size()
        scb = getattr(w, "SCB", None)
        if scb is not None:
            nb += scb.numel() * scb.element_size()
        if m.bias is not None:
            nb += m.bias.numel() * m.bias.element_size()
        c = classify(n)
        out[c] = out.get(c, 0) + nb
    return out


def kv_read_bytes(model, max_len, elem):
    c = model.config
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    # Defect C3-D5 (fixed in v1.3): Qwen2 configs label layers "full_attention"; counting only
    # "attention" gave 0 KV layers and zero KV bytes for Qwen. Count every attention-type layer.
    L = sum(1 for t in c.layer_types if "attention" in str(t)) if getattr(c, "layer_types", None) else c.num_hidden_layers
    return 2 * L * c.num_key_value_heads * hd * max_len * elem


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [np.median(rng.choice(x, x.size)) for _ in range(n)]
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


@torch.no_grad()
def teacher_forced(model, ids, T, steps, dev):
    """Full forward (no cache) on ids[:T+steps]; returns log-probs (steps x V, float32 CPU)
    for predicting ids[T..T+steps-1], and their NLL."""
    x = torch.tensor([ids[:T + steps]], device=dev)
    lg = model(input_ids=x, use_cache=False, logits_to_keep=steps + 1).logits[0, :steps].float()
    lp = torch.log_softmax(lg, -1).cpu()
    tgt = torch.tensor(ids[T:T + steps])
    nll = (-lp[torch.arange(steps), tgt]).numpy()
    return lp, nll


def quality_vs_ref(ref, cand):
    """ref/cand: list per book of (logprobs, nll). PPL ratio with book-bootstrap CI, top-1, KL."""
    d_nll = [float(np.mean(c[1]) - np.mean(r[1])) for r, c in zip(ref, cand)]
    top1 = [float((c[0].argmax(-1) == r[0].argmax(-1)).float().mean()) for r, c in zip(ref, cand)]
    kl = [float((r[0].exp() * (r[0] - c[0])).sum(-1).mean()) for r, c in zip(ref, cand)]
    all_d = float(np.mean(np.concatenate([c[1] for c in cand])) - np.mean(np.concatenate([r[1] for r in ref])))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(ref), len(ref))
        bs.append(math.exp(np.mean(np.concatenate([cand[i][1] for i in idx])) - np.mean(np.concatenate([ref[i][1] for i in idx]))))
    t1 = float(np.mean(top1))
    pr = math.exp(all_d)
    return {"ppl_ratio": pr, "ppl_ratio_ci95_book_bootstrap": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "top1_agree": t1, "kl_mean": float(np.mean(kl)), "per_book_delta_nll": d_nll, "per_book_top1": top1,
            "matched_quality": bool(pr <= 1.01 and t1 >= 0.98)}


@torch.no_grad()
def consistency(model, books_ids, T, steps, dev, dtype_name):
    rows = []
    for name, ids in books_ids:
        x = torch.tensor([ids[:T + steps]], device=dev)
        full = model(input_ids=x, use_cache=False, logits_to_keep=steps).logits[0].double()
        cache = make_static_cache(model, T + steps)
        prefill(model, cache, ids[:T], dev)
        dec = []
        for i in range(steps):
            o = model(input_ids=x[:, T + i:T + i + 1], past_key_values=cache, cache_position=torch.tensor([T + i], device=dev), use_cache=True)
            dec.append(o.logits[0, -1].double())
        dec = torch.stack(dec)
        lf, ld = torch.log_softmax(full, -1), torch.log_softmax(dec, -1)
        rows.append({"book": name, "max_kl": float((lf.exp() * (lf - ld)).sum(-1).max()),
                     "argmax_agree": float((dec.argmax(-1) == full.argmax(-1)).double().mean()),
                     "max_abs_logit_diff": float((dec - full).abs().max())})
        del cache
    lim = KL_LIMIT[dtype_name]
    return {"T": T, "steps": steps, "kl_limit": lim, "rows": rows,
            "pass": all(r["max_kl"] <= lim and r["argmax_agree"] == 1.0 for r in rows)}


def bandwidth(dev, reps=5):
    if dev.type != "cuda":
        return {"skipped": "not cuda"}
    n = 1 << 28  # 256M fp32 = 1 GiB per tensor
    a = torch.empty(n, device=dev); b = torch.rand(n, device=dev); c = torch.rand(n, device=dev)
    res = {}
    for name, fn, nbytes in (("copy", lambda: a.copy_(b), 2 * 4 * n), ("triad", lambda: torch.add(b, c, alpha=3.0, out=a), 3 * 4 * n)):
        fn(); torch.cuda.synchronize(dev)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(dev); ts.append(time.perf_counter() - t0)
        res[name] = {"gbps_median": nbytes / float(np.median(ts)) / 1e9, "gbps_best": nbytes / min(ts) / 1e9, "seconds": ts}
    ok = bool(torch.allclose(a[:1000], b[:1000] + 3.0 * c[:1000]))
    del a, b, c
    torch.cuda.empty_cache()
    res["check_ok"] = ok
    return res


class Decoder:
    """Teacher-forced decode steps at positions T..T+steps-1 on a prefilled static cache."""

    def __init__(self, model, cache, toks, T, dev):
        self.model, self.cache, self.dev = model, cache, dev
        self.toks = torch.tensor(toks, device=dev).view(-1, 1, 1)
        self.pos = torch.arange(T, T + len(toks), device=dev).view(-1, 1)
        self.ids_buf = self.toks[0].clone()
        self.pos_buf = self.pos[0].clone()
        self.graphs = {}

    def eager(self, i):
        return self.model(input_ids=self.toks[i], past_key_values=self.cache, cache_position=self.pos[i], use_cache=True).logits

    @torch.no_grad()
    def capture(self, key):
        # Defect C3-D4 (main run 1): capture used to run with autograd enabled, so each graph's
        # output kept its whole autograd graph (~3-4 GB of saved activations per capture) and
        # bitsandbytes took its training path. Capture must use the inference path, as eager does.
        assert not torch.is_grad_enabled()
        self.ids_buf.copy_(self.toks[0]); self.pos_buf.copy_(self.pos[0])
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(input_ids=self.ids_buf, past_key_values=self.cache, cache_position=self.pos_buf, use_cache=True)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(input_ids=self.ids_buf, past_key_values=self.cache, cache_position=self.pos_buf, use_cache=True).logits
        self.graphs[key] = (g, out)

    def replay(self, key, i):
        g, out = self.graphs[key]
        self.ids_buf.copy_(self.toks[i]); self.pos_buf.copy_(self.pos[i])
        g.replay()
        return out


def run_context(model, ab, ids, T, steps, reps, dev, modes, rng, deadline):
    """Times every ablation in every mode, interleaved per rep in seeded shuffled order."""
    toks = ids[T:T + steps]
    cache = make_static_cache(model, T + steps)
    t0 = time.perf_counter()
    prefill(model, cache, ids[:T], dev)
    sync(dev)
    res = {"prefill_s": time.perf_counter() - t0, "max_cache_len": T + steps, "modes": {}, "mem": {"after_prefill": mem_stats(dev)}}
    D = Decoder(model, cache, toks, T, dev)
    units = [("eager", a) for a in ABLATIONS]
    if "graph" in modes:
        if dev.type != "cuda":
            res["modes"]["graph"] = {"skipped": "not cuda"}
        else:
            gerr = {}
            for a in ABLATIONS:
                ab.set(removed_set(a))
                try:
                    D.capture(a)
                    units.append(("graph", a))
                except Exception as e:  # recorded, never hidden
                    gerr[a] = f"{type(e).__name__}: {str(e)[:300]}"
                finally:
                    ab.restore()
            res["graph_capture_errors"] = gerr
            res["mem"]["after_captures"] = mem_stats(dev)
    # eager vs graph logits check on the full configuration
    with torch.no_grad():
        ab.set(set())
        el = [D.eager(i)[0, -1].double().cpu() for i in range(steps)]
        if ("graph", "full") in units:
            gl = [D.replay("full", i)[0, -1].double().cpu() for i in range(steps)]
            le, lg_ = torch.log_softmax(torch.stack(el), -1), torch.log_softmax(torch.stack(gl), -1)
            kl = float((le.exp() * (le - lg_)).sum(-1).max())
            agree = bool((torch.stack(el).argmax(-1) == torch.stack(gl).argmax(-1)).all())
            res["graph_vs_eager"] = {"max_kl": kl, "argmax_equal": agree, "valid": bool(agree and kl <= GRAPH_KL_LIMIT)}
        ab.restore()
    times = {u: [] for u in units}
    with torch.no_grad():
        for rep in range(reps + 1):  # rep 0 = warm-up
            if time.time() > deadline:
                res["truncated_by_budget_at_rep"] = rep
                break
            for k in rng.permutation(len(units)):
                mode, a = units[k]
                if mode == "eager":
                    ab.set(removed_set(a))
                for i in range(steps):
                    sync(dev); ts = time.perf_counter()
                    if mode == "eager":
                        D.eager(i)
                    else:
                        D.replay(a, i)
                    sync(dev)
                    if rep > 0:
                        times[(mode, a)].append(time.perf_counter() - ts)
                ab.restore()
    res["mem"]["after_timing"] = mem_stats(dev)
    for (mode, a), ts in times.items():
        if not ts:
            continue
        mres = res["modes"].setdefault(mode, {})
        mres[a] = {"p50_s": float(np.median(ts)), "p95_s": float(np.quantile(ts, .95)), "ci95_p50": list(boot_ci(ts)), "n": len(ts), "raw": ts}
    for mode, mres in res["modes"].items():
        if "full" not in mres:
            continue
        f = mres["full"]["p50_s"]
        mres["share_removed"] = {c: 1.0 - mres[f"no_{c}"]["p50_s"] / f for c in REMOVABLE if f"no_{c}" in mres}
        if "no_all" in mres:
            mres["share_other_floor"] = mres["no_all"]["p50_s"] / f
            mres["additivity_sum_of_shares_plus_floor"] = sum(mres["share_removed"].values()) + mres["share_other_floor"]
    D.graphs.clear()
    del D, cache
    release(dev)
    return res


def patch_granite_capture(model):
    """Defect C3-D2 (found by the GPU smoke run): GraniteMoeHybridModel._update_mamba_mask runs
    `if cache_position[0] > 0`, a host sync that invalidates CUDA-graph capture. Its result is
    used only by Mamba layers (layer_mask = mamba_mask if layer_type == "mamba" else causal_mask).
    Granite-4.0-micro has none, so returning None is exact. Asserted, never assumed."""
    if getattr(model.config, "model_type", "") != "granitemoehybrid":
        return False
    types = list(getattr(model.config, "layer_types", []) or [])
    assert types and all(t == "attention" for t in types), "Mamba layers present: patch not exact"
    model.model._update_mamba_mask = lambda attention_mask, cache_position: None
    return True


def mem_stats(dev):
    if dev.type != "cuda":
        return {}
    return {"allocated": torch.cuda.memory_allocated(dev), "reserved": torch.cuda.memory_reserved(dev),
            "peak_allocated": torch.cuda.max_memory_allocated(dev)}


def release(dev):
    """Defect C3-D3 (smoke run): models were not freed between formats (Granite int8 OOM with
    14.48 GiB held). Collect reference cycles before emptying the cache; report what remains."""
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        return {f"cuda:{i}": torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())}
    return {}


def load(model_dir, fmt, dev, device_map=None):
    from transformers import AutoModelForCausalLM
    kw = {"attn_implementation": "c3_sdpa"}
    if fmt in ("fp32", "fp16", "bf16"):
        dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[fmt]
        if device_map:
            m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, device_map=device_map, **kw).eval()
        else:
            m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt, **kw).to(dev).eval()
        patch_granite_capture(m)
        return m
    from transformers import BitsAndBytesConfig
    if fmt == "nf4":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    elif fmt == "int8":
        q = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(fmt)
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, quantization_config=q, device_map={"": dev.index or 0}, **kw).eval()
    patch_granite_capture(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
    ap.add_argument("--books", nargs="+", type=Path, required=True)
    ap.add_argument("--contexts", nargs="+", type=int, required=True)
    ap.add_argument("--formats", nargs="+", default=["fp16", "nf4", "int8"])
    ap.add_argument("--modes", nargs="+", default=["eager", "graph"])
    ap.add_argument("--steps", type=int, default=16); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quality-T", type=int, default=4096); ap.add_argument("--quality-steps", type=int, default=32)
    ap.add_argument("--consistency-T", type=int, default=4096); ap.add_argument("--consistency-steps", type=int, default=3)
    ap.add_argument("--ref-device-map", default=None, help="'auto' to split the fp32 reference over GPUs")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--budget-s", type=float, default=1e9)
    ap.add_argument("--layers", type=int, default=None, help="TEST ONLY: truncate to the first N layers (recorded)")
    ap.add_argument("--label", default="result"); ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    deadline = time.time() + a.budget_s
    import transformers
    from transformers import AutoTokenizer
    register_attention()
    dev = torch.device(a.device)
    torch.manual_seed(20261110)
    rng = np.random.default_rng(20261110)
    out = {"meta": {"version": "1.3", "date": "2026-11-10", "label": a.label, "charter": "specs/C3_G0_gpu_decode_bottleneck_charter.md",
                    "model": a.model_name, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "books"},
                    "books": [b.name for b in a.books], "torch": torch.__version__, "cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "python": platform.python_version(),
                    "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor(),
                    "energy": "NOT MEASURED", "test_truncated_layers": a.layers},
           "nvsmi": {"start": nvsmi()}, "formats": {}}

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=1) + "\n")

    if a.layers:  # test-only truncation, applied through the config
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model_dir)
        cfg.num_hidden_layers = a.layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:a.layers]
        cfg.save_pretrained(a.output.parent / f"_trunc_cfg_{a.model_name}")
        out["meta"]["warning"] = "TRUNCATED MODEL: test only, not a result"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    need = max(max(a.contexts) + a.steps, a.quality_T + a.quality_steps, a.consistency_T + a.consistency_steps)
    books = []
    for b in a.books:
        ids = tok(b.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"]
        books.append((b.name, ids[:need]))
        out.setdefault("book_tokens", {})[b.name] = len(ids)
    timing_ids = books[0][1]
    out["bandwidth"] = bandwidth(dev)
    save()

    def model_load(fmt, device_map=None):
        from transformers import AutoModelForCausalLM  # noqa: F401
        if a.layers:
            from transformers import AutoConfig, AutoModelForCausalLM as M
            cfg = AutoConfig.from_pretrained(a.output.parent / f"_trunc_cfg_{a.model_name}")
            dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[fmt]
            mm = M.from_pretrained(a.model_dir, config=cfg, dtype=dt, attn_implementation="c3_sdpa").to(dev).eval()
            patch_granite_capture(mm)
            return mm.requires_grad_(False)
        return load(a.model_dir, fmt, dev, device_map).requires_grad_(False)  # C3-D4: inference only

    # 1. fp32 reference
    t0 = time.time()
    try:
        ref_dm = a.ref_device_map if dev.type == "cuda" else None
        ref = model_load("fp32", ref_dm)
        rdev = next(ref.parameters()).device
        out["reference"] = {"format": "fp32", "device_map": ref_dm, "load_s": time.time() - t0}
        refq = [teacher_forced(ref, ids, a.quality_T, a.quality_steps, rdev) for _, ids in books]
        out["reference"]["nll_mean"] = float(np.mean(np.concatenate([r[1] for r in refq])))
        if not ref_dm:  # fp32 static-path consistency (single device only)
            out["reference"]["consistency_fp32"] = consistency(ref, books, a.consistency_T, a.consistency_steps, rdev, "fp32")
        del ref
    except Exception as e:
        refq = None
        out["reference"] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
    out["mem_after_reference"] = release(dev)
    out["nvsmi"]["after_reference"] = nvsmi()
    save()

    # 2-5. formats
    for fmt in a.formats:
        fres = out["formats"].setdefault(fmt, {})
        if time.time() > deadline:
            fres["skipped"] = "budget"
            save()
            continue
        try:
            t0 = time.time()
            m = model_load(fmt)
            fres["load_s"] = time.time() - t0
        except Exception as e:
            fres["error"] = f"load: {type(e).__name__}: {str(e)[:500]}"
            fres["traceback"] = traceback.format_exc()[-3000:]
            save()
            continue
        ab = Ablator(m)
        fres["params_requiring_grad"] = sum(1 for q in m.parameters() if q.requires_grad)  # must be 0 (C3-D4)
        fres["weight_bytes"] = weight_bytes(m)
        fres["removable_counts"] = {k: len(v) for k, v in ab.mods.items()}
        if fmt in KL_LIMIT:
            try:
                fres["consistency"] = consistency(m, books, a.consistency_T, a.consistency_steps, dev, fmt)
            except Exception as e:
                fres["consistency"] = {"error": f"{type(e).__name__}: {str(e)[:500]}", "pass": False}
        if refq is not None:
            try:
                cq = [teacher_forced(m, ids, a.quality_T, a.quality_steps, dev) for _, ids in books]
                fres["quality_vs_fp32"] = quality_vs_ref(refq, cq)
            except Exception as e:
                fres["quality_vs_fp32"] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
        save()
        elem = 2 if fmt != "fp32" else 4
        for T in a.contexts:
            if time.time() > deadline:
                fres.setdefault("contexts", {})[str(T)] = {"skipped": "budget"}
                save()
                continue
            try:
                release(dev)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                r = run_context(m, ab, timing_ids, T, a.steps, a.reps, dev, a.modes, rng, deadline)
                wb = fres["weight_bytes"]
                kv = kv_read_bytes(m, T + a.steps, elem)
                tot = sum(wb.values()) + kv
                r["bytes_derived"] = {"weights_by_class": wb, "kv_read": kv, "total": tot,
                                      "shares": {"attn_core": kv / tot, "attn_proj": wb.get("attn_proj", 0) / tot,
                                                 "mlp": wb.get("mlp", 0) / tot, "lm_head": wb.get("lm_head", 0) / tot}}
                for mode, mres in r["modes"].items():
                    if "full" in mres:
                        mres["effective_gbps_derived"] = tot / mres["full"]["p50_s"] / 1e9
                r["nvsmi_after"] = nvsmi()
            except Exception as e:
                ab.restore()
                r = {"error": f"{type(e).__name__}: {str(e)[:800]}", "traceback": traceback.format_exc()[-3000:], "mem_at_error": mem_stats(dev)}
                release(dev)
            fres.setdefault("contexts", {})[str(T)] = r
            save()
            print(json.dumps({"format": fmt, "T": T, "full_ms": {md: round(mr["full"]["p50_s"] * 1e3, 3) for md, mr in r.get("modes", {}).items() if "full" in mr},
                              "shares": {md: {k: round(v, 3) for k, v in mr.get("share_removed", {}).items()} for md, mr in r.get("modes", {}).items()},
                              "graph_vs_eager": r.get("graph_vs_eager"), "error": r.get("error")}), flush=True)
        del m, ab
        fres["mem_after_release"] = release(dev)
        save()
    out["nvsmi"]["end"] = nvsmi()
    out["meta"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()


if __name__ == "__main__":
    main()
