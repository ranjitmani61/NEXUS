#!/usr/bin/env python3
"""C2 diagnostic: Granite cached-decode consistency in fp32 on a layer-truncated model.

Version 1.0, 2026-11-10. EXPLORATORY and POST-HOC: written after the charter §3 precondition
(the standing check, KL <= 1e-3 and argmax 1.0) failed for full Granite bf16 on host C, on
3 of 4 books, with argmax agreement 1.0. It does not replace that check and does not make
Granite admissible under the charter as written.

Question: is the residual bf16 rounding, or a cache-logic error? Full fp32 Granite (about
13.6 GB) does not fit in this VM. So the genuine weights of the first N decoder layers are
loaded, with the genuine embedding/tied lm_head and final norm, in fp32 and in bf16, and the
unchanged g5r_cache.decode_consistency is run on each.
Prediction if the cache logic is correct: fp32 KL at float rounding (<< 1e-6), with bf16
KL orders of magnitude larger on the same truncated network. A cache-logic error (as in
defect D4b) would show up in fp32 too.
Limitation: the truncated network is not the full model; it exercises the same cache code
and layer type (all 40 Granite-4.0-micro layers are attention layers) but not the 28 later layers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5r_cache as KC  # noqa: E402
import g5r_hf as H  # noqa: E402
import g5r_host  # noqa: E402


def weight_check(model_dir: Path, m, n_layers: int) -> dict:
    """Compare loaded tensors with the checkpoint's safetensors (cast to the model dtype)."""
    from safetensors import safe_open
    want = ["model.embed_tokens.weight", "model.norm.weight",
            f"model.layers.{n_layers - 1}.self_attn.q_proj.weight", f"model.layers.{n_layers - 1}.shared_mlp.output_linear.weight",
            "model.layers.0.self_attn.k_proj.weight"]
    sd = dict(m.named_parameters())
    res = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in want:
                if k in h.keys() and k in sd:
                    ref = h.get_tensor(k).to(sd[k].dtype)
                    res[k] = bool(torch.equal(ref, sd[k].detach()))
    return {"checked": res, "all_equal": len(res) == len(want) and all(res.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--books", nargs="+", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    H.register()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    cfg = AutoConfig.from_pretrained(a.model_dir)
    full_layers = cfg.num_hidden_layers
    cfg.num_hidden_layers = a.layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(cfg.layer_types[: a.layers])
    out = {"meta": {"version": "1.0", "date": "2026-11-10", "label": "EXPLORATORY, POST-HOC diagnostic (charter §3 precondition failed)",
                    "layers_kept": a.layers, "layers_full": full_layers, "T": a.T, "steps": a.steps,
                    "criterion_reported_not_applied": "max KL <= 1e-3 and argmax 1.0",
                    "torch": torch.__version__, "transformers": transformers.__version__, "host": g5r_host.host_info()},
           "dtypes": {}}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        m, info = AutoModelForCausalLM.from_pretrained(a.model_dir, config=cfg, dtype=dt, attn_implementation="g5r",
                                                       output_loading_info=True)
        m.eval()
        assert len(m.model.layers) == a.layers, len(m.model.layers)
        # Genuine weights only: nothing may be freshly initialised.
        assert not info["missing_keys"], info["missing_keys"][:5]
        chk = weight_check(Path(a.model_dir), m, a.layers)
        out["dtypes"].setdefault(name, {})["weight_check"] = chk
        assert chk["all_equal"], chk
        rows = []
        for b in a.books:
            ids = torch.tensor(tok(b.read_text().replace("\r\n", "\n"), add_special_tokens=False)["input_ids"][: a.T + a.steps])
            with torch.no_grad():
                r = KC.decode_consistency(m, ids, a.T, a.steps)
            r["book"] = b.name
            rows.append(r)
            print(json.dumps({"dtype": name, **r}), flush=True)
        out["dtypes"][name].update({"rows": rows, "max_kl": max(r["max_kl"] for r in rows),
                                    "argmax_agree_min": min(r["argmax_agree"] for r in rows),
                                    "unexpected_keys_count": len(info["unexpected_keys"])})
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=1) + "\n")
        del m


if __name__ == "__main__":
    main()
