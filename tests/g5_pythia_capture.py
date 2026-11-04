#!/usr/bin/env python3
"""G5 PRETRAINED-PILOT runner for EleutherAI/pythia-14m. NOT a performance benchmark.

For a genuine pretrained diagnostic, install `torch transformers safetensors`,
obtain the pinned public weights and a representative local text sample:

 python tests/g5_pythia_capture.py \
   --text-file /path/to/sample.txt --max-tokens 1024 \
   --output results/g5_pythia14m_traces.json

Produces attention probability and V-based certificate comparisons on selected
causal prefill positions. It validates that the exposed attention weights and V
reconstruct the actual pre-projection context. The method computes FULL
attention; page counts are infeasible oracle diagnostics. It does NOT establish
fast decode, accuracy, energy, memory-traffic improvements or G5 PASS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from g5_trace_oracle import evaluate_one, summarize
from g5_result_validator import validate

REPO = 'EleutherAI/pythia-14m'
REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
VERSION = '1.1'
DATE = '2026-11-04'


def verify_checkpoint_revision(requested: str, resolved: str | None) -> None:
    """Fail closed on unpinned or unverifiable model provenance.

    This checks Git revision identity, NOT a local-weight file checksum.
    """
    if requested != REVISION:
        raise ValueError('Pilot requires the audited 40-character Pythia commit SHA')
    if resolved != REVISION:
        raise RuntimeError('Loaded model commit is missing or differs from the pinned commit')


def extract_v(qkv, num_heads:int):
    """GPT-NeoX fused QKV: [B,T,H*(3*Dh)] => V: [B,T,H,Dh]."""
    if qkv.ndim != 3 or qkv.shape[-1] % (3*num_heads):
        raise ValueError('Fused GPTNeoX QKV shape unexpected')
    b,t,_ = qkv.shape
    head_dim = qkv.shape[-1]//(3*num_heads)
    return qkv.reshape(b,t,num_heads,3*head_dim)[...,2*head_dim:3*head_dim]


def context_reconstruction_error(attention_weights, v, pre_projection, query_positions):
    """Absolute error of reconstructed attention context vs actual dense input.

    All inputs are torch.Tensor. Reconstructing a selected query requires all
    key/value tokens and therefore is validation, not an accelerated path.
    """
    import torch
    if v.ndim != 4 or pre_projection.ndim!=3 or attention_weights.ndim!=4:
        raise ValueError('Invalid attention/V/output ranks')
    b,t,h,dh = v.shape
    if b!=1 or attention_weights.shape != (1,h,t,t):
        raise ValueError('Unexpected attention shapes')
    if pre_projection.shape != (1,t,h*dh):
        raise ValueError('Pre-projection output incompatible with V')
    if not query_positions:
        raise ValueError('At least one query position must be checked')
    errors=[]
    for q in query_positions:
        if not (0<=q<t):raise ValueError('invalid query index')
        row = attention_weights[0,:,q,:].to(torch.float64)
        if (not bool(torch.isfinite(row).all()) or bool((row < 0).any()) or
                bool((row[:,q+1:] > 1e-7).any()) or
                bool(((row.sum(-1)-1.0).abs() > 2e-4).any())):
            raise ValueError(f'Invalid causal, nonnegative, normalized attention row at q={q}')
        p=row[:,:q+1]
        values=v[0,:q+1,:,:].to(torch.float64)
        # [H,K] * [K,H,D] => [H,D]
        reconstruct=torch.einsum('hk,khd->hd',p,values).reshape(h*dh)
        expected=pre_projection[0,q,:].to(torch.float64)
        errors.append(float((reconstruct-expected).abs().max().item()))
    return max(errors) if errors else 0.0


def capture_for_text(model, tokenizer, text:str, *, max_tokens:int=512,
                     last_queries:int=16, page_size:int=32, tolerance:float=.02):
    import torch
    if not hasattr(model,'gpt_neox'):
        raise RuntimeError('This capture only supports GPT-NeoX Pythia architecture')
    config=model.config
    if getattr(config,'model_type',None)!='gpt_neox':
        raise RuntimeError('Expected GPT-NeoX config; architecture mismatch')
    ids=tokenizer(text,return_tensors='pt',add_special_tokens=False)['input_ids']
    if ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<64:
        raise ValueError('Text must tokenize to at least 64 tokens')
    ids=ids[:, :min(max_tokens,config.max_position_embeddings)]
    t=int(ids.shape[1]); heads=int(config.num_attention_heads)
    if t < 64: raise ValueError('Provide at least 64 tokens for diagnostic')
    if not (0<last_queries<=t): raise ValueError('Invalid query count')
    qkv_out={}; dense_inputs={}; handles=[]
    layers=model.gpt_neox.layers
    try:
        for layer_idx, layer in enumerate(layers):
            att=layer.attention
            def qkv_hook(mod,inputs,output,layer_idx=layer_idx):
                qkv_out[layer_idx]=output.detach().cpu()
            def pre_dense_hook(mod,inputs,layer_idx=layer_idx):
                dense_inputs[layer_idx]=inputs[0].detach().cpu()
            handles.append(att.query_key_value.register_forward_hook(qkv_hook))
            handles.append(att.dense.register_forward_pre_hook(pre_dense_hook))
        model.eval()
        with torch.no_grad():
            outputs=model(ids,output_attentions=True,use_cache=False,return_dict=True)
    finally:
        for handle in handles: handle.remove()
    if not outputs.attentions or len(outputs.attentions)!=len(layers):
        raise RuntimeError('Attention matrices unavailable. Enable eager attention implementation.')
    if len(qkv_out)!=len(layers) or len(dense_inputs)!=len(layers):
        raise RuntimeError('Missing hook outputs; cannot trust capture')
    positions=list(range(t-last_queries,t))
    checked_positions=sorted(set([0,t//2,t-1] + positions))
    # Select first, middle, and last transformer layer for pilot, all heads.
    layer_idxs=sorted(set([0,len(layers)//2,len(layers)-1]))
    rows=[]; checks=[]
    for i in layer_idxs:
        a=outputs.attentions[i].detach().cpu()
        v=extract_v(qkv_out[i],heads)
        err=context_reconstruction_error(a,v,dense_inputs[i],checked_positions)
        checks.append({'layer':i,'max_preprojection_absolute_error':err, 'positions_checked':len(checked_positions)})
        if not np.isfinite(err) or err > 0.005:
            raise AssertionError(f'Pythia QKV V reconstruction failed, layer={i}, abs error={err}')
        for h in range(heads):
            for q in positions:
                w=a[0,h,q,:q+1].numpy()
                vals=v[0,:q+1,h,:].numpy()
                row=evaluate_one(w,vals,page_size,tolerance)
                row.update({'layer':int(i),'head':int(h),'query_token_index':int(q)})
                rows.append(row)
    return rows,checks,t


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text-file',type=Path,required=True)
    p.add_argument('--model',default=REPO)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--last-queries',type=int,default=16)
    p.add_argument('--page-size',type=int,default=32)
    p.add_argument('--tolerance',type=float,default=.02)
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/g5_pythia14m_traces.json'))
    a=p.parse_args()
    if a.max_tokens < 64 or a.last_queries<1:
        p.error('max-tokens must >=64 and last-queries >=1')
    if a.model!=REPO:
        p.error('Only Pythia-14M GPT-NeoX capture is validated in this pilot')
    if a.revision != REVISION:
        p.error('Revision must match the pinned Pythia pilot commit')
    if a.page_size <= 0 or not np.isfinite(a.tolerance) or not (0 < a.tolerance < 1):
        p.error('Positive page size and finite (0,1) tolerance required')
    text=a.text_file.read_text(encoding='utf8')
    digest=hashlib.sha256(text.encode()).hexdigest()
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit('Cannot run pretrained pilot: install transformers, torch, safetensors. '+str(e))
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,
            attn_implementation='eager',torch_dtype=torch.float32,
            local_files_only=a.local_files_only,trust_remote_code=False)
    verify_checkpoint_revision(a.revision, getattr(model.config,'_commit_hash',None))
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision,
            local_files_only=a.local_files_only,trust_remote_code=False)
    rows,checks,t=capture_for_text(model,tokenizer,text,max_tokens=a.max_tokens,
         last_queries=a.last_queries,page_size=a.page_size,tolerance=a.tolerance)
    result={'metadata':{
        'date':DATE,'version':VERSION,'status':'PILOT ONLY — NOT G5 PASS',
        'model_id':a.model,'requested_revision':a.revision,
        'resolved_revision':getattr(model.config,'_commit_hash',None),
        'training':'Pretrained checkpoint requested; exact tensor hashes not independently audited',
        'max_context':int(model.config.max_position_embeddings),
        'evaluated_length':t,'query_positions_per_head':a.last_queries,
        'layers_sampled':[int(x['layer']) for x in checks],
        'heads_per_layer':int(model.config.num_attention_heads),
        'input_text_sha256':digest,'input_text_path':str(a.text_file),
        'input_text_bytes':len(text.encode()),
        'page_size':a.page_size,'output_error_norm_tolerance':a.tolerance,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'python':platform.python_version(),'device':'cpu','inference_precision':'float32',
        'oracle_warning':'FULL attention and V extracted. No actual inference acceleration or token-level task-quality testing.',
        'decode_warning':'Attention matrices from causal prefill, not runtime KV-cache decode latency.',
        'statistics_warning':'Sampled adjacent positions and same model/input are correlated; no confidence claims.',
        'sample_warning':'A single Pythia-14M context (<=2048) is not representative of 8K+ long-context workloads.'
    },'reconstruction_checks':checks,'summary':summarize(rows),'rows':rows}
    # Fail closed before persisting any result with missing/contradictory rows.
    result['artifact_validation']=validate(result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
    print('saved',a.output)

if __name__=='__main__':
    main()
