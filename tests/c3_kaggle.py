#!/usr/bin/env python3
"""Minimal Kaggle kernel client for C3 (push / status / output / quota). Version 1.0, 2026-11-10.

Uses the Kaggle public RPC API (https://api.kaggle.com/v1/kernels.KernelsApiService/...),
as tests/g5r_kaggle_fetch.py does. Authentication is injected by this session's network proxy;
the 'Bearer dummy' header is a placeholder that the proxy replaces. No credential is stored here.
The `kaggle` CLI cannot be used: it introspects the token from the request body, which the
proxy does not rewrite.

  quota                               GPU/TPU weekly quota
  push SCRIPT --title T [--slug user/slug] [--gpu] [--internet] [--model M ...] [--dataset D ...] [--timeout S]
  status REF                          session status of user/slug
  output REF --dest DIR               list and download the session's output files
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

API = "https://api.kaggle.com/v1/kernels.KernelsApiService"


def rpc(method: str, body: dict, timeout: int = 120) -> dict:
    r = subprocess.run(["curl", "-sS", "--max-time", str(timeout), "-X", "POST", "-H", "Content-Type: application/json",
                        "-H", "Authorization: Bearer dummy", "-d", json.dumps(body), f"{API}/{method}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{method}: curl rc={r.returncode} {r.stderr[:300]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"{method}: non-JSON reply {r.stdout[:400]}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("quota")
    p = sp.add_parser("push")
    p.add_argument("script", type=Path); p.add_argument("--title", required=True); p.add_argument("--slug")
    p.add_argument("--gpu", action="store_true"); p.add_argument("--internet", action="store_true")
    p.add_argument("--model", nargs="*", default=[]); p.add_argument("--dataset", nargs="*", default=[])
    p.add_argument("--timeout", type=int, default=900); p.add_argument("--machine-shape", default="NvidiaTeslaT4")
    s = sp.add_parser("status"); s.add_argument("ref")
    o = sp.add_parser("output"); o.add_argument("ref"); o.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "quota":
        print(json.dumps(rpc("GetAcceleratorQuotaStatistics", {}), indent=1))
    elif a.cmd == "push":
        body = {"newTitle": a.title, "text": a.script.read_text(), "language": "python", "kernelType": "script",
                "isPrivate": True, "enableGpu": a.gpu, "enableTpu": False, "enableInternet": a.internet,
                "datasetDataSources": a.dataset, "modelDataSources": a.model, "kernelDataSources": [],
                "competitionDataSources": [], "categoryIds": [], "sessionTimeoutSeconds": a.timeout}
        if a.gpu:
            body["machineShape"] = a.machine_shape
        if a.slug:
            body["slug"] = a.slug
        print(json.dumps(rpc("SaveKernel", body), indent=1))
    elif a.cmd == "status":
        u, k = a.ref.split("/", 1)
        print(json.dumps(rpc("GetKernelSessionStatus", {"userName": u, "kernelSlug": k}), indent=1))
    elif a.cmd == "output":
        u, k = a.ref.split("/", 1)
        res = rpc("ListKernelSessionOutput", {"userName": u, "kernelSlug": k})
        a.dest.mkdir(parents=True, exist_ok=True)
        (a.dest / "_listing.json").write_text(json.dumps({kk: vv for kk, vv in res.items() if kk != "log"}, indent=1))
        if res.get("log") is not None:
            (a.dest / "_session_log.txt").write_text(res["log"] if isinstance(res["log"], str) else json.dumps(res["log"]))
        for f in res.get("files", []):
            name, url = f.get("fileName") or f.get("name"), f.get("url")
            if not name or not url:
                continue
            dest = a.dest / Path(name).name
            r = subprocess.run(["curl", "-sS", "--fail", "--max-time", "600", "-o", str(dest), url], capture_output=True, text=True)
            print(name, "ok" if r.returncode == 0 else f"FAILED {r.stderr[:200]}")
        print(json.dumps({"files": [f.get("fileName") or f.get("name") for f in res.get("files", [])],
                          "next": res.get("nextPageToken")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
