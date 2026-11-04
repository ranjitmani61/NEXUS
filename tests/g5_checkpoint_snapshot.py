#!/usr/bin/env python3
"""G5 Pythia-14M: single-snapshot, hash-recorded model/tokenizer staging.

The Git revision and file hashes describe provenance; they are not a signature of
publisher authenticity and do not prove model outputs are correct. No pretrained
weights are bundled in this project.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

REPO = 'EleutherAI/pythia-14m'
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
REQUIRED = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json')
OPTIONAL = ('generation_config.json', 'special_tokens_map.json')
ALLOWED = REQUIRED + OPTIONAL
MAX_FILE_BYTES = 256 * 1024 * 1024


def _digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    count = 0
    with path.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > MAX_FILE_BYTES:
                raise ValueError(f'oversized checkpoint component: {path.name}')
            sha.update(chunk)
    return sha.hexdigest(), count


def _manifest_digest(files: dict) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode('ascii')).hexdigest()


def audit_snapshot(snapshot: Path, revision: str = REVISION) -> dict:
    """Audit the requested local snapshot, not the origin/authenticity of its files.

    HF cached snapshots normally symlink into a sibling `blobs/` directory.
    Reject links resolving outside that single repository cache directory.
    Only selected JSON and safetensors are staged; pickled weights never load.
    """
    if revision != REVISION:
        raise ValueError('Pythia checkpoint revision must be pinned to audited SHA')
    snapshot = Path(snapshot).absolute()
    if not snapshot.is_dir() or snapshot.name != revision or snapshot.parent.name != 'snapshots':
        raise ValueError('expected .../snapshots/<pinned-40-hex-revision> directory')
    cache_repo_root = snapshot.parent.parent.resolve(strict=True)
    entries = {}
    for name in ALLOWED:
        file = snapshot / name
        if file.is_symlink() and not file.exists():
            raise ValueError(f'broken symlink in checkpoint component: {name}')
        if name in REQUIRED and not file.exists():
            raise FileNotFoundError(f'missing required pinned file: {name}')
        if not file.exists():
            continue
        if not file.is_file():
            raise ValueError(f'non-file checkpoint component: {name}')
        resolved = file.resolve(strict=True)
        if not resolved.is_relative_to(cache_repo_root):
            raise ValueError(f'checkpoint component escapes its cached repository: {name}')
        digest, size = _digest(file)
        entries[name] = {'sha256': digest, 'size_bytes': size}
    if set(REQUIRED).difference(entries):
        raise FileNotFoundError('incomplete pinned snapshot')
    with (snapshot / 'config.json').open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or cfg.get('model_type') != 'gpt_neox':
        raise ValueError('expected GPT-NeoX model config')
    for name in ('tokenizer_config.json', 'tokenizer.json'):
        with (snapshot / name).open('r', encoding='utf-8') as f:
            if not isinstance(json.load(f), dict):
                raise ValueError(f'{name} must be JSON object')
    return {'repo_id': REPO, 'revision': REVISION, 'files': entries,
            'fingerprint_sha256': _manifest_digest(entries),
            'origin_warning': ('Local file hashes and revision-path identity are not publisher '
                               'authenticity proofs; compare with independently trusted hashes.')}


def verify_staged_files(staged_dir: Path, manifest: dict) -> None:
    path = Path(staged_dir)
    for name, expected in manifest['files'].items():
        got_sha, got_size = _digest(path / name)
        if got_sha != expected['sha256'] or got_size != expected['size_bytes']:
            raise RuntimeError(f'staged checkpoint component changed after audit: {name}')
    if _manifest_digest(manifest['files']) != manifest['fingerprint_sha256']:
        raise RuntimeError('checkpoint manifest fingerprint mismatch')


@contextmanager
def stage_pinned_snapshot(snapshot: Path) -> Iterator[tuple[Path, dict]]:
    """Stage model and tokenizer from exactly one pinned HF snapshot.

    This prevents accidental mixing of files from separate revisions and records
    content hashes. Temporary files are removed after the caller finishes loading
    and evaluating the model. No external downloads occur in this function.
    """
    snapshot = Path(snapshot)
    original = audit_snapshot(snapshot)
    with tempfile.TemporaryDirectory(prefix='nexus-g5-pythia-') as tmp:
        dest = Path(tmp)
        for name in original['files']:
            shutil.copyfile(snapshot / name, dest / name)
        verify_staged_files(dest, original)
        yield dest, original
        verify_staged_files(dest, original)
        # Original snapshot can mutate independently; if so fail closed, too.
        current = audit_snapshot(snapshot)
        if current['fingerprint_sha256'] != original['fingerprint_sha256']:
            raise RuntimeError('source checkpoint changed during G5 evaluation')
