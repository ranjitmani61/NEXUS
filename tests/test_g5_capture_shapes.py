"""Shape and end-to-end context reconstruction tests using random mock QKV.
No pretrained weights used.
"""
import numpy as np
import torch
from g5_pythia_capture import extract_v, context_reconstruction_error


def test_qkv_split_known_values():
    qkv=torch.arange(1*7*(3*2*4),dtype=torch.float32).reshape(1,7,24)
    v=extract_v(qkv,2)
    assert v.shape == (1,7,2,4)
    assert torch.equal(v[0,0,0], qkv[0,0,8:12])
    assert torch.equal(v[0,0,1], qkv[0,0,20:24])


def test_context_reconstruction_causal():
    gen=torch.Generator().manual_seed(1337)
    T,H,D=13,3,4
    v=torch.randn((1,T,H,D),generator=gen)
    logits=torch.randn((1,H,T,T),generator=gen)
    causal_mask=torch.triu(torch.ones((T,T),dtype=torch.bool),diagonal=1)
    logits=logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0),-1e10)
    a=logits.softmax(-1)
    pre_proj=torch.einsum('bhqk,bkhd->bqhd',a,v).reshape(1,T,H*D)
    e=context_reconstruction_error(a,v,pre_proj,[0,2,7,12])
    assert e<1e-6
    bad=pre_proj.clone(); bad[0,7,2]+=0.2
    assert context_reconstruction_error(a,v,bad,[7])>0.1
