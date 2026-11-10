#!/usr/bin/env python3
"""C3 feasibility probe for a Kaggle GPU session (environment inventory; NOT a result).

Version 1.0, 2026-11-10. Runs as a Kaggle script kernel with GPU. It inventories the
environment so that the C3 charter can be written and pre-registered on facts, before any
measurement:
- GPU, driver, CUDA and PyTorch build; bf16/fp16 support;
- which candidate baseline libraries import, and their versions;
- whether pip can reach PyPI (internet);
- the attached model and dataset files with SHA-256, to compare against
  results/g5r/provenance_*.json (same weights as G5-R/C2);
- a load-and-forward sanity check of both models in fp16 and bf16 (finite logits).

The small matmul timings are smoke checks for dtype feasibility only, labelled as such; they
are not results and will not enter any decision. Only an allowlist of environment variables is
printed (no secrets). Output: /kaggle/working/c3_probe.json.
"""
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

OUT = {"version": "1.0", "label": "feasibility probe, NOT a result", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "out": r.stdout[-4000:], "err": r.stderr[-1500:]}
    except Exception as e:  # recorded, not hidden
        return {"rc": None, "error": repr(e)}


def save():
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working/c3_probe.json").write_text(json.dumps(OUT, indent=1) + "\n")


OUT["python"] = platform.python_version()
OUT["env_allowlist"] = {k: os.environ.get(k) for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE", "KAGGLE_CONTAINER_NAME",
                                                        "CUDA_VERSION", "NVIDIA_DRIVER_CAPABILITIES") if os.environ.get(k)}
OUT["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv")
OUT["nvidia_smi_full"] = sh("nvidia-smi")
OUT["cpu"] = sh("nproc; grep -m1 'model name' /proc/cpuinfo; free -g | head -2")
save()

import torch  # noqa: E402
OUT["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "available": torch.cuda.is_available(), "count": torch.cuda.device_count()}
if torch.cuda.is_available():
    OUT["torch"]["devices"] = [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)),
                                "mem_gb": torch.cuda.get_device_properties(i).total_memory / 2**30} for i in range(torch.cuda.device_count())]
    OUT["torch"]["bf16_supported"] = torch.cuda.is_bf16_supported()
    OUT["torch"]["arch_list"] = torch.cuda.get_arch_list()
save()

libs = {}
for name in ("transformers", "accelerate", "bitsandbytes", "torchao", "flash_attn", "xformers", "triton", "vllm", "llama_cpp",
             "auto_gptq", "gptqmodel", "awq", "optimum", "exllamav2", "hqq", "safetensors", "numpy"):
    try:
        m = importlib.import_module(name)
        libs[name] = getattr(m, "__version__", "imported (no __version__)")
    except Exception as e:
        libs[name] = f"NOT AVAILABLE: {type(e).__name__}: {str(e)[:160]}"
OUT["libs"] = libs
OUT["pip_freeze_head"] = sh(f"{sys.executable} -m pip freeze | grep -i -E 'torch|transformers|bitsandbytes|accelerate|triton|nvidia-cublas|flash|xformers|optimum|gptq|awq|llama' ")
t0 = time.time()
OUT["internet_pip"] = sh(f"{sys.executable} -m pip download --no-deps --dest /tmp/pipprobe transformers==4.57.1", timeout=180)
OUT["internet_pip"]["seconds"] = time.time() - t0
save()

inputs = []
for root, _, files in os.walk("/kaggle/input"):
    for f in files:
        p = Path(root) / f
        inputs.append(str(p))
OUT["inputs_count"] = len(inputs)
hashes = {}
for p in inputs:
    if "/kaggle/input/the-pg19" in p and not any(b in p for b in ("/860.txt", "/2544.txt", "/3247.txt", "/3340.txt")):
        continue  # only the four C2 books
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    hashes[p] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
OUT["input_hashes"] = hashes
OUT["pg19_dirs"] = sorted({str(Path(p).parent) for p in inputs if "pg19" in p})[:20]
save()

# dtype smoke (NOT a result): 4096x4096 GEMM and a GEMV, fp16 vs bf16 vs fp32
if torch.cuda.is_available():
    sm = {}
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        try:
            a = torch.randn(4096, 4096, device="cuda", dtype=dt); b = torch.randn(4096, 4096, device="cuda", dtype=dt)
            v = torch.randn(4096, 1, device="cuda", dtype=dt)
            for _ in range(3):
                a @ b; a @ v
            torch.cuda.synchronize()
            t = time.perf_counter(); [a @ b for _ in range(10)]; torch.cuda.synchronize(); gemm = (time.perf_counter() - t) / 10
            t = time.perf_counter(); [a @ v for _ in range(100)]; torch.cuda.synchronize(); gemv = (time.perf_counter() - t) / 100
            sm[str(dt)] = {"gemm_4096_ms": gemm * 1e3, "gemm_tflops": 2 * 4096**3 / gemm / 1e12, "gemv_4096_us": gemv * 1e6,
                           "gemv_gbps": 4096 * 4096 * a.element_size() / gemv / 1e9}
        except Exception as e:
            sm[str(dt)] = f"ERROR {type(e).__name__}: {str(e)[:200]}"
    OUT["dtype_smoke_not_a_result"] = sm
    save()

# model load + finite-logit sanity, both models, fp16 and bf16
def find_model(tag):
    for p in inputs:
        if p.endswith("config.json") and tag in p:
            return str(Path(p).parent)
    return None

if torch.cuda.is_available():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    book = next((p for p in inputs if p.endswith("/860.txt")), None)
    san = {}
    for tag in ("qwen2.5", "granite"):
        d = find_model(tag)
        san[tag] = {"dir": d}
        if d is None:
            continue
        tok = AutoTokenizer.from_pretrained(d)
        text = Path(book).read_text()[:20000] if book else "The quick brown fox " * 200
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :1024].cuda()
        for dt in (torch.float16, torch.bfloat16):
            try:
                t = time.time()
                m = AutoModelForCausalLM.from_pretrained(d, dtype=dt, attn_implementation="sdpa").cuda().eval()
                with torch.no_grad():
                    lg = m(input_ids=ids).logits.float()
                nll = torch.nn.functional.cross_entropy(lg[0, :-1], ids[0, 1:]).item()
                san[tag][str(dt)] = {"finite": bool(torch.isfinite(lg).all()), "nll_1024": nll, "load_and_fwd_s": time.time() - t,
                                     "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                del m, lg
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                san[tag][str(dt)] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
            save()
    OUT["model_sanity"] = san
OUT["done_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print(json.dumps({k: OUT[k] for k in ("torch", "libs", "done_utc") if k in OUT}, indent=1))
