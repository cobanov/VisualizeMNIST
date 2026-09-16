"""Independent NumPy reconstruction of every exported ResNet/ViT output.
Run with numpy, onnx and onnxruntime. Does not require PyTorch or training data.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]

def activation(x, kind):
    if kind=='relu': return np.maximum(x,0)
    if kind=='gelu': return .5*x*(1+np.vectorize(math.erf)(x/np.sqrt(2)))
    return x

def forward(manifest,weights,x):
    values={'input':x}
    for s in manifest['layers'][1:]:
        a=values[s['source']]; op=s['op']
        w=weights.get(s.get('weight')); b=weights.get(s.get('bias'))
        if op=='conv':
            p=s['padding']; padded=np.pad(a,((0,0),(0,0),(p,p),(p,p)))
            patches=np.lib.stride_tricks.sliding_window_view(padded,(s['kernel'],)*2,axis=(2,3))[:,:,::s['stride'],::s['stride']]
            result=activation(np.einsum('bcyxij,ocij->boyx',patches,w)+b[None,:,None,None],s.get('activation','relu'))
        elif op=='add': result=activation(a+values[s['skip']],s['activation'])
        elif op=='pool': result=a.mean((2,3) if s['axis']=='spatial' else 1)
        elif op=='patches': result=a.reshape(1,1,4,7,4,7).transpose(0,2,4,1,3,5).reshape(1,16,49)
        elif op=='embed': result=a@w+b+weights[s['position']]
        elif op=='norm': result=(a-a.mean(-1,keepdims=True))/np.sqrt(a.var(-1,keepdims=True)+s['epsilon'])*w+b
        elif op=='attention':
            qkv=(a@w+b).reshape(1,16,3,3,16).transpose(2,0,3,1,4)
            q,k,v=qkv[0],qkv[1],qkv[2]; values.update(q=q,k=k,v=v)
            score=q@k.swapaxes(-2,-1)/np.sqrt(s['headDim'])
            result=np.exp(score-score.max(-1,keepdims=True)); result/=result.sum(-1,keepdims=True)
        elif op=='mix': result=(a@values[s['values']]).transpose(0,2,1,3).reshape(1,16,48)
        elif op in ('token_dense','dense'): result=activation(a@w+b,s['activation'])
        elif op=='softmax': result=np.exp(a-a.max(-1,keepdims=True)); result/=result.sum(-1,keepdims=True)
        else: raise ValueError(op)
        values[s['tensor']]=result.astype(np.float32)
    return values


def check(kind):
    folder=ROOT/'web/models'; manifest=json.loads((folder/f'mnist_{kind}.json').read_text())
    onnx_path=folder/manifest['file']; onnx.checker.check_model(onnx.load(onnx_path))
    assert hashlib.sha256(onnx_path.read_bytes()).hexdigest()==manifest['sha256']
    raw=np.fromfile(folder/manifest['weights']['file'],dtype='<f4')
    weights={name:raw[t['offset']:t['offset']+math.prod(t['shape'])].reshape(t['shape']) for name,t in manifest['weights']['tensors'].items()}
    session=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(102)
    inputs=[np.zeros((1,1,28,28),np.float32),np.ones((1,1,28,28),np.float32),rng.random((1,1,28,28),dtype=np.float32)]
    corner=np.zeros_like(inputs[0]);corner[0,0,27,0]=1;inputs.append(corner)
    max_error=0
    for x in inputs:
        expected=forward(manifest,weights,x)
        outputs=dict(zip([o.name for o in session.get_outputs()],session.run(None,{'input':x})))
        for name,got in outputs.items():
            assert np.isfinite(got).all() and np.isfinite(expected[name]).all(), f'Non-finite output: {kind}/{name}'
            np.testing.assert_allclose(got,expected[name],atol=1e-4,rtol=7e-4,err_msg=f'{kind}/{name}')
            max_error=max(max_error,float(np.max(np.abs(got-expected[name]))))
        np.testing.assert_allclose(outputs['probs'].sum(),1,atol=1e-6)
        if kind=='vit': np.testing.assert_allclose(outputs['attention'].sum(-1),1,atol=1e-6)
    print(f'PASS {kind}: independent NumPy parity, {len(inputs)} inputs x {len(outputs)} tensors, max error {max_error:.2g}')

if __name__=='__main__':
    for kind in ['resnet','vit']: check(kind)
