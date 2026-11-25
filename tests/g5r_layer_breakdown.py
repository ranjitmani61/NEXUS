#!/usr/bin/env python3
"""G5-R per-layer breakdown of page fractions from raw shadow records (v1.0, 2026-11-04).

Usage: python tests/g5r_layer_breakdown.py RAW_DIR MODEL_NAME T --output results/g5r/layers_<m>_T<T>.json
Fractions are per query head (k / full pages), averaged over steps, heads and books.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path); ap.add_argument("model"); ap.add_argument("T", type=int)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(str(a.raw_dir / f"{a.model}_T{a.T}_*_records.npz")))
    acc = {}
    for f in files:
        z = np.load(f)
        layers = z["layer"]
        full = np.array([int(t) // 32 for t in z["T"]])[:, None]
        for key in ("box_k", "sph_k", "val_k", "orscore_k", "ormass_k"):
            frac = z[key] / np.maximum(full, 1)
            for L in np.unique(layers):
                acc.setdefault(int(L), {}).setdefault(key, []).append(frac[layers == L].ravel())
        for L in np.unique(layers):
            acc[int(L)].setdefault("box_union", []).append((z["box_union"][layers == L] / np.maximum(full[layers == L], 1)).ravel())
            acc[int(L)].setdefault("page0_mass", []).append(z["page0_mass"][layers == L].ravel())
    out = {"model": a.model, "T": a.T, "files": [Path(f).name for f in files], "layers": {}}
    for L, d in sorted(acc.items()):
        out["layers"][L] = {k: float(np.mean(np.concatenate(v))) for k, v in d.items()}
        out["layers"][L]["box_share_full"] = float(np.mean(np.concatenate(d["box_k"]) >= 1 - 1e-12))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'layer':>5} {'NX-BOX':>7} {'BOX∪KV':>7} {'NX-SPH':>7} {'NX-VAL':>7} {'OR-SCORE':>8} {'OR-MASS':>7} {'page0 mass':>10}")
    for L, d in out["layers"].items():
        print(f"{L:>5} {d['box_k']:7.3f} {d['box_union']:7.3f} {d['sph_k']:7.3f} {d['val_k']:7.3f} {d['orscore_k']:8.3f} {d['ormass_k']:7.3f} {d['page0_mass']:10.3f}")


if __name__ == "__main__":
    main()
