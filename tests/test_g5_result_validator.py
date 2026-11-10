"""Adversarial trace schema test using generated synthetic oracle fixture, not pretrained."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from g5_result_validator import InvalidTrace, PILOT_MODEL, PILOT_REVISION, load_strict,validate
from g5_trace_oracle import evaluate_one, summarize


@pytest.fixture(scope='module')
def fixture():
    rng=np.random.default_rng(193)
    t=65; q_count=3; heads=2; layers=[0,2]
    rows=[]
    for layer in layers:
        for h in range(heads):
            for q in range(t-q_count,t):
                w=rng.dirichlet(np.ones(q+1))
                v=rng.standard_normal((q+1,4))
                r=evaluate_one(w,v,page_size=8,tolerance=.02)
                r.update(layer=layer,head=h,query_token_index=q)
                rows.append(r)
    return {'metadata':{
        'model_id':PILOT_MODEL,'requested_revision':PILOT_REVISION,
        'resolved_revision':PILOT_REVISION, 'status':'PILOT ONLY — NOT G5 PASS',
        'evaluated_length':t,'query_positions_per_head':q_count,
        'heads_per_layer':heads,'page_size':8,
        'output_error_norm_tolerance':.02,'input_text_sha256':'a'*64,
        'layers_sampled':layers,
    },'rows':rows,'reconstruction_checks':[
        {'layer':layer,'max_preprojection_absolute_error':.0001,'positions_checked':5}
        for layer in layers], 'summary':summarize(rows)}


def check_reject(payload):
    with pytest.raises(InvalidTrace):
        validate(payload)


def test_valid_fixture_but_only_structural(fixture):
    result=validate(fixture)
    assert result['rows']==12
    assert result['models_executed_by_validator']==0
    assert 'BLOCKED' in result['gate']


@pytest.mark.parametrize('mutation',[ 'remove_row', 'duplicate_row', 'wrong_layer',
 'wrong_head', 'wrong_query', 'wrong_tokens','wrong_pages', 'wrong_fraction',
 'true_error_above_certificate','mass_certificate_above_tolerance',
 'negative_error','nan_error','infinite_error','bool_page_count','summary_count',
 'summary_median','summary_fraction','missing_reconstruction','wrong_reconstruction_positions',
 'bad_revision','fake_pass_status','bad_text_sha','wrong_dimension','bad_oracle_note',
 'bad_guarantee','negative_cert','oversize_cert','hindsight_error','nan_summary',
 'bool_summary_flag'])
def test_tampering_fails_closed(fixture,mutation):
    p=copy.deepcopy(fixture); r=p['rows'][0]; m=p['metadata']; s=p['summary']
    if mutation=='remove_row': p['rows'].pop()
    elif mutation=='duplicate_row': p['rows'][-1]=copy.deepcopy(r)
    elif mutation=='wrong_layer': r['layer']=999
    elif mutation=='wrong_head': r['head']=2
    elif mutation=='wrong_query': r['query_token_index']=-1
    elif mutation=='wrong_tokens': r['tokens']+=1
    elif mutation=='wrong_pages': r['pages']+=1
    elif mutation=='wrong_fraction': r['oracle_mass_fraction']+=.01
    elif mutation=='true_error_above_certificate': r['oracle_exact_score_value_true_error']=r['oracle_exact_score_value_certificate']+.2
    elif mutation=='mass_certificate_above_tolerance': r['oracle_mass_certificate']=.5
    elif mutation=='negative_error': r['oracle_exact_score_value_true_error']=-.0001
    elif mutation=='nan_error': r['oracle_exact_score_value_true_error']=float('nan')
    elif mutation=='infinite_error': r['oracle_exact_score_value_true_error']=float('inf')
    elif mutation=='bool_page_count': r['pages']=True
    elif mutation=='summary_count': s['queries']+=1
    elif mutation=='summary_median': s['median_oracle_mass_fraction']+=.01
    elif mutation=='summary_fraction': s['fraction_value_oracle_fewer_pages_than_mass_oracle']=.1234
    elif mutation=='missing_reconstruction': p['reconstruction_checks'].pop()
    elif mutation=='wrong_reconstruction_positions': p['reconstruction_checks'][0]['positions_checked']=3
    elif mutation=='bad_revision': m['resolved_revision']='f'*40
    elif mutation=='fake_pass_status': m['status']='G5 PASS'
    elif mutation=='bad_text_sha': m['input_text_sha256']='xyz'
    elif mutation=='wrong_dimension': r['dim']+=1
    elif mutation=='bad_oracle_note': r['oracle_note']='real hardware accelerator'
    elif mutation=='bad_guarantee': r['guarantee_type']='formal proof for FP16 silicon'
    elif mutation=='negative_cert': r['oracle_exact_score_value_certificate']=-.01
    elif mutation=='oversize_cert': r['oracle_exact_score_value_certificate']=.3
    elif mutation=='hindsight_error': r['greedy_hindsight_true_error']=.04
    elif mutation=='nan_summary': s['min_oracle_mass_fraction']=float('nan')
    elif mutation=='bool_summary_flag': s['all_value_certificates_verified_numerically']=1
    check_reject(p)


def test_load_strict_rejects_duplicate_keys(tmp_path:Path):
    f=tmp_path/'dupe.json';f.write_text('{"rows":1,"rows":2}')
    with pytest.raises(InvalidTrace,match='duplicate'):
        load_strict(f)


@pytest.mark.parametrize('invalid',['NaN','Infinity','-Infinity','{broken','"\\ud800"'])
def test_load_strict_rejects_malformed_or_nonfinite(tmp_path:Path,invalid):
    f=tmp_path/'invalid.json'; f.write_text('{"x":'+invalid+'}')
    with pytest.raises((InvalidTrace,UnicodeError)):
        load_strict(f)


def test_roundtrip_strict_loader(tmp_path:Path,fixture):
    f=tmp_path/'valid.json';f.write_text(json.dumps(fixture,allow_nan=False))
    assert validate(load_strict(f))['rows']==12


def test_prior_summary_flag_can_hide_corrupted_row(fixture):
    """Demonstrates why the old `summarize` flag is not itself an audit."""
    p=copy.deepcopy(fixture)
    p['rows'][0]['oracle_exact_score_value_true_error']=0.8
    p['summary']=summarize(p['rows'])
    assert p['summary']['all_value_certificates_verified_numerically'] is True
    check_reject(p)
