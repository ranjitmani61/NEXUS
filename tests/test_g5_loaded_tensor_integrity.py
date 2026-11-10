"""G5 v1.3: adversarial in-memory checkpoint integrity, synthetic tensor fixtures only."""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from g5_checkpoint_snapshot import verify_loaded_model_weights
from g5_result_validator import InvalidTrace, validate
from test_g5_result_validator import fixture as trace_fixture


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)
        self.extra = torch.nn.Parameter(torch.tensor([3.0]))


@pytest.fixture
def weights(tmp_path: Path):
    m = FixtureModel().eval()
    with torch.no_grad():
        m.fc.weight.copy_(torch.tensor([[0.5, -1.25, 2], [2, 0.2, -0.1]]))
        m.fc.bias.copy_(torch.tensor([-0.2, 0.4]))
    save_file({k: v.detach().clone() for k, v in m.state_dict().items()},
              str(tmp_path / 'model.safetensors'))
    return m, tmp_path


def test_valid_model_exact_match(weights):
    m, d = weights
    result = verify_loaded_model_weights(m, d)
    assert result['checkpoint_tensors_verified'] == 3
    assert result['unverified_model_parameters'] == 0
    assert result['checkpoint_elements_verified'] == 9


def test_mutated_in_memory_param_fails(weights):
    m, d = weights
    assert verify_loaded_model_weights(m, d)
    with torch.no_grad():
        m.fc.weight[0, 0] += 0.25
    with pytest.raises(RuntimeError, match='value mismatch'):
        verify_loaded_model_weights(m, d)


def test_corrupted_disk_file_fails(weights):
    m, d = weights
    changed={k: v.detach().clone() for k, v in m.state_dict().items()}
    changed['fc.bias'][0] += 1
    save_file(changed,str(d / 'model.safetensors'))
    with pytest.raises(RuntimeError, match='value mismatch'):
        verify_loaded_model_weights(m, d)


def test_missing_file_fails(weights):
    m, d = weights
    (d / 'model.safetensors').unlink()
    with pytest.raises(ValueError, match='ordinary file'):
        verify_loaded_model_weights(m, d)


def test_missing_tensor_fails(weights):
    m, d = weights
    save_file({'fc.weight': m.fc.weight.detach().clone()}, str(d / 'model.safetensors'))
    with pytest.raises(RuntimeError, match='unverified model parameters'):
        verify_loaded_model_weights(m, d)


def test_extra_checkpoint_tensor_fails(weights):
    m, d = weights
    expected = {k: v.detach().clone() for k, v in m.state_dict().items()}
    expected['unrecognized.weight'] = torch.ones(4)
    save_file(expected, str(d / 'model.safetensors'))
    with pytest.raises(RuntimeError, match='missing from loaded model'):
        verify_loaded_model_weights(m, d)


def test_shape_mismatch_fails(weights):
    m, d = weights
    expected = {k: v.detach().clone() for k, v in m.state_dict().items()}
    expected['fc.weight'] = expected['fc.weight'].reshape(6)
    save_file(expected, str(d / 'model.safetensors'))
    with pytest.raises(RuntimeError, match='shape mismatch'):
        verify_loaded_model_weights(m, d)


def test_nonfinite_fails(weights):
    m, d = weights
    with torch.no_grad():
        m.fc.weight[0,0] = float('nan')
    with pytest.raises(RuntimeError, match='non-finite'):
        verify_loaded_model_weights(m, d)


def test_fp16_weights_to_fp32_allowed(weights):
    m, d = weights
    expected = {k: v.detach().half().contiguous() for k, v in m.state_dict().items()}
    save_file(expected, str(d / 'model.safetensors'))
    with torch.no_grad():
        for name, param in m.named_parameters():
            param.copy_(expected[name].float())
    assert verify_loaded_model_weights(m, d)['checkpoint_tensors_verified']==3


def test_fp32_to_fp16_lossy_rejected(weights):
    m, d = weights
    m.half()
    with pytest.raises(RuntimeError, match='unsafe dtype'):
        verify_loaded_model_weights(m, d)


def test_tied_parameters_allow_single_checkpoint_copy(tmp_path):
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(2, 3)
            self.output = torch.nn.Linear(3, 2, bias=False)
            self.output.weight = self.embed.weight
    model = Tied()
    save_file({'embed.weight': model.embed.weight.detach().clone()},str(tmp_path / 'model.safetensors'))
    r=verify_loaded_model_weights(model,tmp_path)
    assert r['checkpoint_tensors_verified']==1


def test_v13_manifest_requires_memory_check(trace_fixture):
    p=copy.deepcopy(trace_fixture)
    m=p['metadata']
    m['version']='1.3'
    with pytest.raises(InvalidTrace, match='checkpoint_manifest'):
        validate(p)
    files={name:{'sha256':'b'*64,'size_bytes':4} for name in
         ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json')}
    import json,hashlib
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',', ':'),ensure_ascii=True).encode('ascii')).hexdigest()
    m['checkpoint_manifest']={'repo_id':m['model_id'], 'revision':m['resolved_revision'],
        'files':files, 'fingerprint_sha256':digest,
        'origin_warning':'Not publisher authenticity.'}
    with pytest.raises(InvalidTrace, match='model_weight_integrity'):
        validate(p)
    m['model_weight_integrity']={
        'status':'PASS: in-memory tensors match staged checkpoint',
        'method':'exact tensor equality after lossless dtype promotion',
        'checkpoint_tensors_verified':2,'checkpoint_elements_verified':5,
        'unverified_model_parameters':0,'checks':2,
        'authenticity_warning':'Not established independent publisher authenticity',
    }
    assert validate(p)['models_executed_by_validator']==0
    m['model_weight_integrity']['checks']=1
    with pytest.raises(InvalidTrace,match='before and after'):
        validate(p)


def test_alias_same_pointer_wrong_shape_must_not_count(tmp_path):
    class MalformedTied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Parameter(torch.ones(6))
            self.wrong = torch.nn.Parameter(self.embed.data.view(2, 3))
    m=MalformedTied()
    save_file({'embed':m.embed.detach().clone()},str(tmp_path/'model.safetensors'))
    with pytest.raises(RuntimeError,match='unverified model parameters'):
        verify_loaded_model_weights(m,tmp_path)
