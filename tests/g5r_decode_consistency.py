#!/usr/bin/env python3
"""G5-R standing check: cached decode == full forward (v1.0, 2026-11-04).

Added after two Granite defects (no cache returned; empty-cache length = batch size),
which corrupted decode silently. Run for every model before any decode-based result is
used. Pass criterion: max KL(full || cached) <= 1e-3 and argmax agreement 1.0.
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_host  # noqa: E402
import g5r_hf as H  # noqa: E402
import g5r_cache as KC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True); ap.add_argument("--model-name", required=True)
ap.add_argument("--dtype", default="fp32"); ap.add_argument("--books", nargs="+", type=Path, required=True)
ap.add_argument("--T", type=int, default=8192); ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads)
from transformers import AutoModelForCausalLM, AutoTokenizer
H.register()
tok = AutoTokenizer.from_pretrained(a.model_dir)
m = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype], attn_implementation="g5r").eval()
rows = []
for b in a.books:
    ids = torch.tensor(tok(b.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"][: a.T + a.steps])
    r = KC.decode_consistency(m, ids, a.T, a.steps); r["book"] = b.name
    r["pass"] = bool(r["max_kl"] <= 1e-3 and r["argmax_agree"] == 1.0)
    rows.append(r); print(json.dumps(r), flush=True)
out = {"model": a.model_name, "dtype": a.dtype, "host": g5r_host.host_info(), "criterion": "max KL <= 1e-3 and argmax agreement 1.0", "rows": rows,
       "all_pass": all(r["pass"] for r in rows)}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=1) + "\n")
print("ALL PASS" if out["all_pass"] else "FAIL")
