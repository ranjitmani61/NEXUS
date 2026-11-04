"""Adversarial, OFFLINE tests for pinned snapshot provenance; NOT real-model G5."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest

from g5_checkpoint_snapshot import (REPO, REVISION, REQUIRED, audit_snapshot,
                                    stage_pinned_snapshot, verify_staged_files)
from g5_result_validator import InvalidTrace, validate
from test_g5_result_validator import fixture as trace_fixture


@pytest.fixture
def fake_cache(tmp_path:Path):
    """Bytes are deliberately FAKE; these tests never load pretrained weights."""
    snap=tmp_path/'models--EleutherAI--pythia-14m'/'snapshots'/REVISION
    snap.mkdir(parents=True)
    (snap/'config.json').write_text(json.dumps({'model_type':'gpt_neox'}))
    (snap/'model.safetensors').write_bytes(b'FAKE SAFETENSORS NOT A MODEL')
    (snap/'tokenizer.json').write_text('{}')
    (snap/'tokenizer_config.json').write_text('{}')
    (snap/'generation_config.json').write_text('{}')
    return snap


def test_stages_single_snapshot_and_records_file_hashes(fake_cache):
    with stage_pinned_snapshot(fake_cache) as (staged, manifest):
        assert manifest['repo_id']==REPO and manifest['revision']==REVISION
        assert set(REQUIRED).issubset(manifest['files'])
        assert all(len(v['sha256'])==64 for v in manifest['files'].values())
        assert (staged/'model.safetensors').read_bytes()==(fake_cache/'model.safetensors').read_bytes()
        verify_staged_files(staged,manifest)
        assert not (staged/'pytorch_model.bin').exists()
    assert not staged.exists()


def test_deterministic_manifest_from_same_bytes(fake_cache):
    m1=audit_snapshot(fake_cache);m2=audit_snapshot(fake_cache)
    assert m1==m2


def test_changes_after_snapshot_creation_fail(fake_cache):
    before=audit_snapshot(fake_cache)
    (fake_cache/'model.safetensors').write_bytes(b'CHANGED')
    assert audit_snapshot(fake_cache)['fingerprint_sha256'] != before['fingerprint_sha256']


def test_staged_file_changed_while_capture_runs_fails(fake_cache):
    with pytest.raises(RuntimeError,match='staged checkpoint component changed'):
        with stage_pinned_snapshot(fake_cache) as (staged,manifest):
            (staged/'tokenizer.json').write_text('{"modified":true}')


def test_source_file_changed_while_capture_runs_fails(fake_cache):
    with pytest.raises(RuntimeError,match='source checkpoint changed'):
        with stage_pinned_snapshot(fake_cache):
            (fake_cache/'tokenizer_config.json').write_text('{"tampered":true}')


def test_missing_model_file_rejected(fake_cache):
    (fake_cache/'model.safetensors').unlink()
    with pytest.raises(FileNotFoundError,match='model.safetensors'):
        audit_snapshot(fake_cache)


def test_wrong_revision_rejected(fake_cache):
    with pytest.raises(ValueError,match='pinned'):
        audit_snapshot(fake_cache, revision='0'*40)
    other=fake_cache.parent/('f'*40);other.mkdir()
    with pytest.raises(ValueError,match='snapshots'):
        audit_snapshot(other)


def test_source_symlink_escaping_repo_rejected(fake_cache,tmp_path):
    external=tmp_path/'outside.json';external.write_text('{}')
    (fake_cache/'tokenizer_config.json').unlink()
    (fake_cache/'tokenizer_config.json').symlink_to(external)
    with pytest.raises(ValueError,match='escapes'):
        audit_snapshot(fake_cache)


def test_cache_blob_symlink_is_allowed(fake_cache):
    repo=fake_cache.parent.parent
    blobs=repo/'blobs';blobs.mkdir()
    data=blobs/'blob1';data.write_bytes((fake_cache/'model.safetensors').read_bytes())
    (fake_cache/'model.safetensors').unlink()
    (fake_cache/'model.safetensors').symlink_to(data)
    audit_snapshot(fake_cache)


def test_corrupt_json_rejected(fake_cache):
    (fake_cache/'tokenizer.json').write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        audit_snapshot(fake_cache)


def test_wrong_model_architecture_rejected(fake_cache):
    (fake_cache/'config.json').write_text('{"model_type":"llama"}')
    with pytest.raises(ValueError,match='GPT-NeoX'):
        audit_snapshot(fake_cache)


def test_large_file_rejected(fake_cache):
    with (fake_cache/'model.safetensors').open('wb') as f:
        f.seek(256 * 1024 * 1024)
        f.write(b'0')
    with pytest.raises(ValueError,match='oversized'):
        audit_snapshot(fake_cache)


def test_manifest_integrity_checked_by_independent_validator(fake_cache,trace_fixture):
    payload=copy.deepcopy(trace_fixture)
    payload['metadata']['version']='1.2'
    payload['metadata']['checkpoint_manifest']=audit_snapshot(fake_cache)
    assert validate(payload)['models_executed_by_validator']==0
    payload['metadata']['checkpoint_manifest']['files']['model.safetensors']['sha256']='a'*64
    with pytest.raises(InvalidTrace,match='fingerprint'):
        validate(payload)


@pytest.mark.parametrize('mutation',['missing','wrong_revision','bad_sha','wrong_inventory',
                                    'wrong_size','wrong_origin'])
def test_bad_v12_checkpoint_metadata_rejected(fake_cache,trace_fixture,mutation):
    payload=copy.deepcopy(trace_fixture)
    payload['metadata']['version']='1.2'
    manifest=audit_snapshot(fake_cache)
    payload['metadata']['checkpoint_manifest']=manifest
    if mutation=='missing': del payload['metadata']['checkpoint_manifest']
    elif mutation=='wrong_revision': manifest['revision']='e'*40
    elif mutation=='bad_sha': manifest['files']['config.json']['sha256']='BAD'
    elif mutation=='wrong_inventory': del manifest['files']['tokenizer.json']
    elif mutation=='wrong_size': manifest['files']['model.safetensors']['size_bytes']=-1
    elif mutation=='wrong_origin': manifest['origin_warning']='Verified publisher signature'
    with pytest.raises(InvalidTrace):
        validate(payload)


def test_broken_optional_symlink_fails_closed(fake_cache,tmp_path):
    (fake_cache/'generation_config.json').unlink()
    (fake_cache/'generation_config.json').symlink_to(tmp_path/'missing.json')
    with pytest.raises(ValueError,match='broken symlink'):
        audit_snapshot(fake_cache)
