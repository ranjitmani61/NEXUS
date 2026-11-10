"""Cycle-2 shared instrumentation (C2-G0 charter, specs/C2_G0_batch1_decode_bottleneck_charter.md).

Version 1.0, 2026-11-10.
- classify(): maps a Linear module name to a cost class.
- LinearTimer: forward pre/post hooks on leaf Linear modules (dense or dynamically
  quantized). CPU execution is synchronous, so each interval is the module's wall-clock time.
- linear_weight_bytes(): bytes of all Linear weights (+bias). These are read once per decode
  step at batch 1. [Derived]
- triad_gbps(): STREAM-like triad a = b + s*c on torch tensors. [Measured] host bandwidth.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj", "input_linear", "output_linear")
CLASSES = ("attn_core", "attn_proj", "mlp", "lm_head", "linear_other", "other")


def classify(name: str) -> str:
    leaf = name.split(".")[-1]
    if name == "lm_head" or leaf == "lm_head":
        return "lm_head"
    if leaf in ATTN_PROJ:
        return "attn_proj"
    if leaf in MLP:
        return "mlp"
    return "linear_other"


def is_linear(mod) -> bool:
    if isinstance(mod, torch.nn.Linear):
        return True
    t = type(mod)
    return t.__name__ == "Linear" and "quantized" in t.__module__


def linear_modules(model):
    return [(n, m) for n, m in model.named_modules() if is_linear(m)]


def linear_weight_bytes(model) -> dict:
    """Bytes per class of Linear weights + biases. Tied weights (lm_head = embedding) are
    counted once, under lm_head, because the decode step reads them through lm_head."""
    out = defaultdict(int)
    seen = set()
    for n, m in linear_modules(model):
        if isinstance(m, torch.nn.Linear):
            w = m.weight
            names = getattr(w, "tensor_data_names", None)
            if names:  # torchao tensor subclass (e.g. Int8Tensor): count the inner storage.
                # Its element_size() reports the outer dtype and data_ptr() is 0, so neither is usable.
                inner = [getattr(w, k) for k in names]
                key = inner[0].data_ptr()
                nb = sum(t.numel() * t.element_size() for t in inner)
            else:
                key = w.data_ptr()
                nb = w.numel() * w.element_size()
            if key in seen:
                continue
            seen.add(key)
            if m.bias is not None:
                nb += m.bias.numel() * m.bias.element_size()
        else:  # torch.ao dynamic quantized Linear: int8 weight + fp32 bias/scales
            w = m.weight()
            nb = w.numel() * 1
            b = m.bias()
            if b is not None:
                nb += b.numel() * b.element_size()
            if w.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                nb += w.q_per_channel_scales().numel() * 8
        out[classify(n)] += nb
    return dict(out)


class LinearTimer:
    """Accumulates wall-clock time per class over Linear modules while attached."""

    def __init__(self, model):
        self.acc = defaultdict(float)
        self.calls = defaultdict(int)
        self._t0 = {}
        self.handles = []
        for n, m in linear_modules(model):
            c = classify(n)
            self.handles.append(m.register_forward_pre_hook(self._pre(n)))
            self.handles.append(m.register_forward_hook(self._post(n, c)))

    def _pre(self, name):
        def f(mod, inp):
            self._t0[name] = time.perf_counter()
        return f

    def _post(self, name, cls):
        def f(mod, inp, out):
            self.acc[cls] += time.perf_counter() - self._t0.pop(name)
            self.calls[cls] += 1
        return f

    def reset(self):
        self.acc.clear()
        self.calls.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def triad_gbps(n_bytes: int = 1 << 30, reps: int = 5, threads: int = 4) -> dict:
    """STREAM-like triad on fp32 tensors of n_bytes each. Counts 3 * n_bytes per pass
    (2 reads + 1 write; write-allocate traffic is not counted, as in STREAM)."""
    torch.set_num_threads(threads)
    n = n_bytes // 4
    b = torch.rand(n)
    c = torch.rand(n)
    a = torch.empty(n)
    torch.add(b, c, alpha=3.0, out=a)  # warm-up / first touch
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.add(b, c, alpha=3.0, out=a)
        ts.append(time.perf_counter() - t0)
    ok = bool(torch.allclose(a[:1000], b[:1000] + 3.0 * c[:1000]))
    del a, b, c
    med = float(np.median(ts))
    return {"bytes_per_pass": 3 * n_bytes, "seconds": ts, "gbps_median": 3 * n_bytes / med / 1e9,
            "gbps_best": 3 * n_bytes / min(ts) / 1e9, "check_ok": ok, "threads": threads}


def quantize_variant(base, variant: str):
    """B1 family (charter §4 + Amendment A1). Returns a quantized deep copy of `base`."""
    import copy
    from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic
    from torch.ao.nn.quantized.dynamic import Linear as QLinear
    m = copy.deepcopy(base)
    # Explicit Linear-only mapping (v1.1 fix): with a module-name qconfig spec, the default
    # mapping also tried to convert nn.Embedding (model.embed_tokens) and raised.
    only_linear = {torch.nn.Linear: QLinear}
    if variant == "int8dyn":          # registered: per-tensor qint8, all Linear
        return quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8, mapping=only_linear).eval()
    if variant == "int8dyn_pc":       # A1: per-channel weight scales, all Linear
        return quantize_dynamic(m, {torch.nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8, mapping=only_linear).eval()
    if variant == "int8dyn_pc_nolm":  # A1: per-channel, lm_head left in its original dtype
        body = [n for n, _ in m.named_children() if n != "lm_head"]
        return quantize_dynamic(m, {n: per_channel_dynamic_qconfig for n in body}, dtype=torch.qint8, mapping=only_linear).eval()
    raise ValueError(variant)


def quantize_int8wo_(model):
    """Amendment A2: torchao Int8WeightOnlyConfig (weight-only int8, per-output-channel scales)
    on every nn.Linear, in place. Activations stay in the model dtype."""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    quantize_(model, Int8WeightOnlyConfig())
    return model.eval()


def boot_ratio(a, b, n=2000, seed=0):
    """95% bootstrap CI of median(a)/median(b), resampling each independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    r = [np.median(rng.choice(a, a.size)) / np.median(rng.choice(b, b.size)) for _ in range(n)]
    return float(np.quantile(r, .025)), float(np.quantile(r, .975))
