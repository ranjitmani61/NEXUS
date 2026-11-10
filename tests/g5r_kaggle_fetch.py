#!/usr/bin/env python3
"""G5-R artifact fetcher: official Kaggle model files and Kaggle dataset files.

Version 1.0, 2026-11-04. Part of the G5 real-model study (see
specs/G5R_real_model_benchmark_contract.md).

Why this exists: in the 2026-11-04 cloud session huggingface.co was refused by the
environment network policy (HTTP 403 on CONNECT), while the Kaggle API was
reachable with proxy-injected credentials. The Kaggle API answers a download RPC
with a 302 redirect to a signed storage.googleapis.com URL that must be fetched
with GET. This script performs that two-step download with curl (so the
environment's HTTPS proxy and CA bundle apply), then records byte count and
SHA-256 for every file in a provenance manifest.

Provenance limits (stated, not hidden):
- Checksums are computed locally after download. They prove *which bytes* were
  used, not that the bytes equal the Hugging Face upstream revision; that
  cross-check could not be made because huggingface.co was unreachable.
- Kaggle "officialness" rests on the publishing organisation (qwen-lm = QwenLM,
  ibm-research = IBM Research) as displayed by the Kaggle API.

Usage:
  python tests/g5r_kaggle_fetch.py model qwen-lm qwen2.5 0.5b 1 --out DIR
  python tests/g5r_kaggle_fetch.py dataset tunguz the-pg19-language-modeling-benchmark-dataset \
      --files test/test/10146.txt ... --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

API = "https://api.kaggle.com/v1"


def _rpc(service: str, method: str, body: dict, timeout: int = 120) -> dict:
    out = subprocess.run(
        ["curl", "-sS", "--fail-with-body", "--max-time", str(timeout), "-X", "POST",
         "-H", "Content-Type: application/json", "-H", "Authorization: Bearer dummy",
         "-d", json.dumps(body), f"{API}/{service}/{method}"],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"RPC {method} failed rc={out.returncode}: {out.stdout[:400]} {out.stderr[:400]}")
    return json.loads(out.stdout)


def _redirect_location(service: str, method: str, body: dict) -> str:
    out = subprocess.run(
        ["curl", "-sS", "--max-time", "120", "-o", "/dev/null", "-D", "-", "-X", "POST",
         "-H", "Content-Type: application/json", "-H", "Authorization: Bearer dummy",
         "-d", json.dumps(body), f"{API}/{service}/{method}"],
        capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"No redirect for {body}: {out.stdout[:300]} {out.stderr[:300]}")


def _get(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sS", "--fail", "--max-time", "3600", "-o", str(tmp), url],
                           capture_output=True, text=True)
        if r.returncode == 0:
            tmp.replace(dest)
            return
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"GET failed for {dest.name}: {r.stderr[:300]}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def list_model_files(owner, model, instance, version, framework="MODEL_FRAMEWORK_TRANSFORMERS"):
    files, tok = [], None
    while True:
        body = {"ownerSlug": owner, "modelSlug": model, "framework": framework,
                "instanceSlug": instance, "versionNumber": version, "pageSize": 100}
        if tok:
            body["pageToken"] = tok
        d = _rpc("models.ModelApiService", "ListModelInstanceVersionFiles", body)
        files += d.get("files", [])
        tok = d.get("nextPageToken")
        if not tok:
            return files


def fetch_model(owner, model, instance, version, out: Path, skip_git=True) -> dict:
    meta = _rpc("models.ModelApiService", "GetModel", {"ownerSlug": owner, "modelSlug": model})
    inst = [i for i in meta.get("instances", []) if i.get("slug") == instance
            and i.get("framework") == "MODEL_FRAMEWORK_TRANSFORMERS"]
    listed = list_model_files(owner, model, instance, version)
    records = []
    for f in listed:
        name = f["name"]
        if skip_git and name.startswith(".git/"):
            continue
        dest = out / name
        if not (dest.exists() and dest.stat().st_size == int(f["size"])):
            loc = _redirect_location("models.ModelApiService", "DownloadModelInstanceVersion",
                                     {"ownerSlug": owner, "modelSlug": model,
                                      "framework": "MODEL_FRAMEWORK_TRANSFORMERS",
                                      "instanceSlug": instance, "versionNumber": version,
                                      "path": name})
            _get(loc, dest)
        size = dest.stat().st_size
        if size != int(f["size"]):
            raise RuntimeError(f"size mismatch {name}: got {size}, listed {f['size']}")
        records.append({"path": name, "bytes": size, "listed_bytes": int(f["size"]),
                        "sha256": sha256_file(dest), "kaggle_creation_date": f.get("creationDate")})
        print(f"  ok {name} {size}", flush=True)
    return {"source": "kaggle_models", "owner": owner, "model": model, "instance": instance,
            "framework": "transformers", "version": version,
            "publisher_author": meta.get("author"), "title": meta.get("title"),
            "license": inst[0].get("licenseName") if inst else None,
            "instance_url": inst[0].get("url") if inst else None,
            "skipped_git_directory_entries": sum(1 for f in listed if f["name"].startswith(".git/")),
            "files": records}


def _listed_dataset_sizes(owner, dataset) -> dict:
    sizes, tok = {}, None
    while True:
        body = {"ownerSlug": owner, "datasetSlug": dataset, "pageSize": 200}
        if tok:
            body["pageToken"] = tok
        d = _rpc("datasets.DatasetApiService", "ListDatasetFiles", body)
        for f in d.get("datasetFiles", []):
            sizes[f["name"]] = int(f["totalBytes"])
        tok = d.get("nextPageToken")
        if not tok:
            return sizes


def fetch_dataset_files(owner, dataset, names, out: Path, listed_sizes: dict | None = None) -> dict:
    """Kaggle serves larger single files as a zip container (observed 2026-11-04 for a
    1.43 MB PG-19 file). Such containers are detected by magic bytes and extracted;
    the extracted size must equal the size in Kaggle's file listing."""
    import zipfile
    listed = listed_sizes if listed_sizes is not None else _listed_dataset_sizes(owner, dataset)
    records = []
    for name in names:
        dest = out / name
        rec = {"path": name, "listed_bytes": listed.get(name)}
        if not dest.exists() or (listed.get(name) and dest.stat().st_size != listed[name]):
            loc = _redirect_location("datasets.DatasetApiService", "DownloadDataset",
                                     {"ownerSlug": owner, "datasetSlug": dataset, "fileName": name})
            raw = dest.with_suffix(dest.suffix + ".download")
            _get(loc, raw)
            with open(raw, "rb") as fh:
                magic = fh.read(4)
            if magic == b"PK\x03\x04":
                rec["served_as_zip"] = True
                rec["zip_sha256"] = sha256_file(raw)
                with zipfile.ZipFile(raw) as zf:
                    members = zf.namelist()
                    if len(members) != 1 or ".." in members[0] or members[0].startswith("/"):
                        raise RuntimeError(f"unexpected zip members for {name}: {members}")
                    dest.write_bytes(zf.read(members[0]))
                raw.unlink()
            else:
                raw.replace(dest)
        size = dest.stat().st_size
        if rec["listed_bytes"] is not None and size != rec["listed_bytes"]:
            raise RuntimeError(f"size mismatch {name}: {size} vs listed {rec['listed_bytes']}")
        rec.update({"bytes": size, "sha256": sha256_file(dest)})
        records.append(rec)
        print(f"  ok {name} {size}", flush=True)
    return {"source": "kaggle_datasets", "owner": owner, "dataset": dataset, "files": records}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("model")
    m.add_argument("owner"); m.add_argument("model"); m.add_argument("instance"); m.add_argument("version", type=int)
    m.add_argument("--out", type=Path, required=True); m.add_argument("--manifest", type=Path)
    d = sub.add_parser("dataset")
    d.add_argument("owner"); d.add_argument("dataset"); d.add_argument("--files", nargs="+", required=True)
    d.add_argument("--out", type=Path, required=True); d.add_argument("--manifest", type=Path)
    d.add_argument("--listed-sizes-json", type=Path, help="JSON with key listed_bytes: {name: bytes}; avoids re-listing (rate limits)")
    a = p.parse_args()
    if a.kind == "model":
        res = fetch_model(a.owner, a.model, a.instance, a.version, a.out)
    else:
        sizes = json.loads(a.listed_sizes_json.read_text())["listed_bytes"] if a.listed_sizes_json else None
        res = fetch_dataset_files(a.owner, a.dataset, a.files, a.out, sizes)
    res["fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(res, indent=2) + "\n"
    if a.manifest:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(text)
    print(text if len(text) < 4000 else text[:4000])


if __name__ == "__main__":
    sys.exit(main())
