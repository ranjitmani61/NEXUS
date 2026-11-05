#!/usr/bin/env python3
"""Fail-closed *structural/numerical* audit of NEXUS Pythia-14M G5 oracle JSON.

This validates internal consistency and explicit local certificate inequalities.
It cannot verify the underlying weights, original attention, or computed model output,
and it cannot authorize G5 PASS. No third-party libraries required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PILOT_MODEL = 'EleutherAI/pythia-14m'
PILOT_REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
METRICS = (
    'oracle_mass_fraction', 'oracle_exact_score_value_fraction',
    'greedy_hindsight_fraction', 'oracle_exact_score_value_true_error',
)


class InvalidTrace(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise InvalidTrace(reason)


def integer(x, name, low: int = 0) -> int:
    require(type(x) is int and x >= low, f'{name}: expected integer >= {low}')
    return x


def finite(x, name, low=None, high=None) -> float:
    require(type(x) in (int, float) and math.isfinite(x), f'{name}: expected finite number')
    value = float(x)
    require((low is None or value >= low) and (high is None or value <= high),
            f'{name}: outside [{low}, {high}]')
    return value


def close(a, b, name, tol=1e-10):
    require(math.isclose(a,b,abs_tol=tol,rel_tol=tol), f'{name}: inconsistent {a} vs {b}')


def validate(payload: dict) -> dict:
    """Check Pythia pilot schema, complete query/head/layer coverage and arithmetic.

    Returns a verified-only subset of structural diagnostics. The trace's model
    origin cannot be independently authenticated from the JSON object itself.
    """
    require(type(payload) is dict, 'root: object required')
    m=payload.get('metadata')
    require(type(m) is dict, 'metadata: object required')
    require(m.get('model_id') == PILOT_MODEL, 'unsupported model_id')
    require(m.get('requested_revision') == PILOT_REVISION, 'unpinned revision')
    require(m.get('resolved_revision') == PILOT_REVISION, 'unverified model revision')
    require(m.get('status') == 'PILOT ONLY — NOT G5 PASS', 'invalid pilot status')
    # v1.2 runs must include model+tokenizer hashes, with an independently
    # recomputed manifest fingerprint. This checks recorded metadata only.
    version=m.get('version')
    require(version in (None,'1.1','1.2','1.3'), 'unrecognized G5 capture version')
    if version in ('1.2','1.3'):
        manifest=m.get('checkpoint_manifest')
        require(type(manifest) is dict, 'v1.2 requires checkpoint_manifest')
        require(manifest.get('repo_id')==PILOT_MODEL and
                manifest.get('revision')==PILOT_REVISION,
                'checkpoint manifest identity mismatch')
        files=manifest.get('files')
        required={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json'}
        allowed=required|{'generation_config.json','special_tokens_map.json'}
        require(type(files) is dict and required.issubset(files) and
                set(files).issubset(allowed),'checkpoint file inventory invalid')
        for name, component in files.items():
            require(type(component) is dict, f'{name}: invalid checkpoint hash entry')
            checksum=component.get('sha256')
            require(type(checksum) is str and len(checksum)==64 and
                    all(c in '0123456789abcdef' for c in checksum),
                    f'{name}: invalid SHA256')
            integer(component.get('size_bytes'),f'{name}.size_bytes',1)
        canonical=json.dumps(files,sort_keys=True,separators=(',', ':'),ensure_ascii=True)
        expected=hashlib.sha256(canonical.encode('ascii')).hexdigest()
        require(manifest.get('fingerprint_sha256')==expected,
                'checkpoint fingerprint mismatched')
        require('not publisher' in str(manifest.get('origin_warning','')).lower() or
                'not a signature' in str(manifest.get('origin_warning','')).lower(),
                'checkpoint provenance caveat missing')
    if version=='1.3':
        weight_check=m.get('model_weight_integrity')
        require(type(weight_check) is dict,'v1.3 requires model_weight_integrity')
        require(weight_check.get('status')=='PASS: in-memory tensors match staged checkpoint',
                'v1.3 memory verification status missing')
        require(weight_check.get('method')=='exact tensor equality after lossless dtype promotion',
                'v1.3 weight integrity method mismatch')
        integer(weight_check.get('checkpoint_tensors_verified'),'checkpoint_tensors_verified',1)
        integer(weight_check.get('checkpoint_elements_verified'),'checkpoint_elements_verified',1)
        require(integer(weight_check.get('unverified_model_parameters'),
                        'unverified_model_parameters')==0,
                'v1.3 unverified model parameters')
        require(weight_check.get('checks')==2, 'v1.3 must compare weights before and after capture')
        require('not established' in str(weight_check.get('authenticity_warning','')).lower(),
                'v1.3 publisher authenticity caveat missing')
    t=integer(m.get('evaluated_length'), 'evaluated_length', 64)
    q_count=integer(m.get('query_positions_per_head'),'query_positions_per_head',1)
    heads=integer(m.get('heads_per_layer'),'heads_per_layer',1)
    psize=integer(m.get('page_size'),'page_size',1)
    tolerance=finite(m.get('output_error_norm_tolerance'), 'output_error_norm_tolerance', 0, 1)
    require(0<tolerance<1 and q_count<=t,'query_count/tolerance inconsistent')
    require(type(m.get('input_text_sha256')) is str and len(m['input_text_sha256'])==64 and
            all(ch in '0123456789abcdef' for ch in m['input_text_sha256']),
            'missing or malformed input text checksum')
    sample=m.get('layers_sampled')
    require(type(sample) is list and len(sample)>0, 'layers_sampled: nonempty list required')
    layers=[integer(i,'layer',0) for i in sample]
    require(len(set(layers))==len(layers) and layers==sorted(layers), 'layers must be unique, sorted')
    rows=payload.get('rows'); checks=payload.get('reconstruction_checks')
    summary=payload.get('summary')
    require(type(rows) is list and type(checks) is list and type(summary) is dict,
            'rows/checks/summary missing or wrong type')
    expected_positions=range(t-q_count,t)
    expected={(layer,head,q) for layer in layers for head in range(heads) for q in expected_positions}
    require(len(rows)==len(expected),f'row count mismatch expected {len(expected)} got {len(rows)}')
    seen=set(); dimensions=set()
    for index,row in enumerate(rows):
        tag=f'rows[{index}]'
        require(type(row) is dict, f'{tag}: not an object')
        key=(integer(row.get('layer'),tag+'.layer'),integer(row.get('head'),tag+'.head'),
             integer(row.get('query_token_index'),tag+'.query_token_index'))
        require(key in expected,f'{tag}: unexpected layer/head/query {key}')
        require(key not in seen,f'{tag}: duplicate layer/head/query {key}')
        seen.add(key)
        n=integer(row.get('tokens'),tag+'.tokens',1)
        dim=integer(row.get('dim'),tag+'.dim',1)
        pages=integer(row.get('pages'),tag+'.pages',1)
        dimensions.add(dim)
        require(n==key[2]+1, f'{tag}: causal prefix length incorrect')
        require(pages==(n+psize-1)//psize, f'{tag}: page count incorrect')
        pairs=(('oracle_mass_pages','oracle_mass_fraction'),
               ('oracle_exact_score_value_pages','oracle_exact_score_value_fraction'),
               ('greedy_hindsight_pages','greedy_hindsight_fraction'))
        for page_field, frac_field in pairs:
            pages_read=integer(row.get(page_field),f'{tag}.{page_field}',1)
            require(pages_read<=pages,f'{tag}.{page_field} exceeds pages')
            frac=finite(row.get(frac_field),f'{tag}.{frac_field}',0,1)
            close(frac,pages_read/pages,f'{tag}.{frac_field}')
        cert=finite(row.get('oracle_exact_score_value_certificate'),f'{tag}.value_cert',0)
        err=finite(row.get('oracle_exact_score_value_true_error'),f'{tag}.true_error',0)
        mass_cert=finite(row.get('oracle_mass_certificate'),f'{tag}.mass_cert',0)
        hindsight=finite(row.get('greedy_hindsight_true_error'),f'{tag}.hindsight_error',0)
        finite(row.get('max_single_page_mass'),f'{tag}.max_page_mass',0,1+1e-8)
        finite(row.get('max_value_norm'),f'{tag}.max_value_norm',0)
        require(err<=cert+2e-10, f'{tag}: reported value cert violated')
        require(cert<=tolerance+2e-10 and mass_cert<=tolerance+2e-10,
                f'{tag}: certificate above tolerance')
        require(hindsight<=tolerance+2e-10, f'{tag}: hindsight exceeds tolerance')
        require(row.get('guarantee_type') ==
                'exact arithmetic sufficient error bound; numerical verification only',
                f'{tag}: misleading guarantee type')
        require('not executable' in str(row.get('oracle_note','')),
                f'{tag}: missing oracle caution')
    require(seen==expected and len(dimensions)==1, 'incomplete row grid or inconsistent dimensions')
    require(len(checks)==len(layers),'missing reconstruction checks')
    checked_positions=len(set([0,t//2,t-1]+list(expected_positions)))
    observed_checks=set()
    for index,c in enumerate(checks):
        require(type(c) is dict,f'checks[{index}]: invalid object')
        layer=integer(c.get('layer'),'check layer')
        require(layer in layers and layer not in observed_checks,'duplicate/missing reconstruction layer')
        observed_checks.add(layer)
        require(integer(c.get('positions_checked'),'positions_checked',1)==checked_positions,
                'reconstruction did not cover all measured positions')
        finite(c.get('max_preprojection_absolute_error'),'reconstruction error',0,0.005+1e-10)
    require(observed_checks==set(layers),'reconstruction layers incomplete')
    require(integer(summary.get('queries'),'summary.queries')==len(rows),
            'summary query count incorrect')
    for metric in METRICS:
        values=sorted(finite(row.get(metric),metric,0) for row in rows)
        n=len(values)
        values_by_key={'min':values[0], 'max':values[-1],
                       'median':values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2}
        for key,value in values_by_key.items():
            actual=finite(summary.get(f'{key}_{metric}'),f'summary.{key}_{metric}',0)
            close(actual,value,f'summary.{key}_{metric}')
    true_fraction=sum(row['oracle_exact_score_value_pages']<row['oracle_mass_pages'] for row in rows)/len(rows)
    fraction=finite(summary.get('fraction_value_oracle_fewer_pages_than_mass_oracle'),
                    'summary.fraction_value_oracle_fewer_pages_than_mass_oracle',0,1)
    close(fraction,true_fraction,'summary.fraction_value_oracle_fewer_pages_than_mass_oracle')
    require(summary.get('all_value_certificates_verified_numerically') is True,
            'summary must assert verified certificates')
    return {'validation':'PASS: trace structure and recorded certificate inequalities',
            'rows':len(rows),'sampled_layers':len(layers),'heads_per_layer':heads,
            'evaluated_length':t,'models_executed_by_validator':0,
            'gate':'G5 BLOCKED: this validation cannot authenticate checkpoint or original traces'}


def load_strict(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size<=100*1024*1024,
            'JSON file missing or above size limit')
    def pairs_unique(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,f'duplicate JSON key: {k}')
            result[k]=v
        return result
    def invalid_constant(s):
        raise InvalidTrace(f'non-finite JSON constant: {s}')
    try:
        result=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs_unique,
                          parse_constant=invalid_constant)
        def no_surrogates(obj):
            if type(obj) is str:
                require(not any(0xD800 <= ord(c) <= 0xDFFF for c in obj),
                        'unpaired Unicode surrogate')
            elif type(obj) is list:
                for el in obj: no_surrogates(el)
            elif type(obj) is dict:
                for key,value in obj.items():
                    no_surrogates(key); no_surrogates(value)
        no_surrogates(result)
        return result
    except (json.JSONDecodeError,UnicodeDecodeError) as e:
        raise InvalidTrace(f'malformed JSON: {e}') from e


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('trace',type=Path)
    args=cli.parse_args()
    try:
        print(json.dumps(validate(load_strict(args.trace)),sort_keys=True))
    except (InvalidTrace,OSError) as e:
        cli.exit(1,f'INVALID G5 TRACE: {e}\n')


if __name__=='__main__':
    main()
