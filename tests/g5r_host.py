"""G5-R host fingerprint (v1.0, 2026-11-05).

Added after the session's container was restarted on a DIFFERENT CPU (AMX-capable
Xeon family 6 model 207 -> Xeon family 6 model 85 without AMX or AVX512-BF16).
g5r_timing.py v1.0 hard-coded the original CPU string; every driver now records the
detected host instead.
"""
import platform


def host_info() -> dict:
    info = {"kernel": platform.release()}
    try:
        txt = open("/proc/cpuinfo").read()
        first = txt.split("\n\n")[0]
        kv = {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in first.splitlines() if ":" in l)}
        flags = set(kv.get("flags", "").split())
        info.update({
            "cpu_model_name": kv.get("model name", ""),
            "cpu_family": kv.get("cpu family", ""),
            "cpu_model": kv.get("model", ""),
            "logical_cpus": txt.count("processor\t:"),
            "amx_bf16": "amx_bf16" in flags, "avx512_bf16": "avx512_bf16" in flags, "avx512f": "avx512f" in flags,
        })
    except OSError:
        pass
    return info
