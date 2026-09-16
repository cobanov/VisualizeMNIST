"""Train the two compact, inspectable MNIST models. No original weights are changed.
uv run --python 3.12 --with torch==2.8.0 --with torchvision==0.23.0 --with onnx==1.19.0 --with onnxruntime==1.22.1 tools/train_models.py
Checkpoints/data are local artifacts; published ONNX, weights and metrics are in web/models.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web/models'
ART = ROOT / 'artifacts/training'
SEED = 71


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, 2, 1)
        self.conv1 = nn.Conv2d(8, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)
        self.down = nn.Conv2d(8, 16, 3, 2, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        stem = F.relu(self.stem(x))
        c1 = F.relu(self.conv1(stem))
        c2 = self.conv2(c1)
        add1 = F.relu(stem + c2)
        down = F.relu(self.down(add1))
        c3 = F.relu(self.conv3(down))
        c4 = self.conv4(c3)
        add2 = F.relu(down + c4)
        pooled = add2.mean((2, 3))
        logits = self.head(pooled)
        return stem, c1, c2, add1, down, c3, c4, add2, pooled, logits, logits.softmax(-1)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(49, 48)
        self.pos = nn.Parameter(torch.randn(1, 16, 48) * .02)
        self.norm1 = nn.LayerNorm(48)
        self.qkv = nn.Linear(48, 144)
        self.project = nn.Linear(48, 48)
        self.norm2 = nn.LayerNorm(48)
        self.ff1 = nn.Linear(48, 96)
        self.ff2 = nn.Linear(96, 48)
        self.norm3 = nn.LayerNorm(48)
        self.head = nn.Linear(48, 10)

    def forward(self, x):
        b = x.shape[0]
        patches = x.reshape(b, 1, 4, 7, 4, 7).permute(0, 2, 4, 1, 3, 5).reshape(b, 16, 49)
        tokens = self.embed(patches) + self.pos
        n1 = self.norm1(tokens)
        qkv = self.qkv(n1).reshape(b, 16, 3, 3, 16).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1) / 4).softmax(-1)
        context = (attention @ v).transpose(1, 2).reshape(b, 16, 48)
        projected = self.project(context)
        add1 = tokens + projected
        n2 = self.norm2(add1)
        ff1 = F.gelu(self.ff1(n2))
        ff2 = self.ff2(ff1)
        add2 = add1 + ff2
        n3 = self.norm3(add2)
        pooled = n3.mean(1)
        logits = self.head(pooled)
        return patches, tokens, n1, attention, context, projected, add1, n2, ff1, ff2, add2, n3, pooled, logits, logits.softmax(-1), q, k, v


def specs(kind):
    layers = [dict(tensor='input', label='Input', op='input', shape=[1, 28, 28])]
    def add(tensor, label, op, shape, source, **extra):
        layers.append(dict(tensor=tensor, label=label, op=op, shape=shape, source=source, **extra))
    if kind == 'resnet':
        prev = 'input'
        for name, label, c, size, stride, activation in [
            ('stem', 'Stem', 8, 14, 2, 'relu'), ('conv1', 'Block 1 / Conv', 8, 14, 1, 'relu'),
            ('conv2', 'Block 1 / Linear', 8, 14, 1, 'linear'), ('add1', 'Residual 1', 8, 14, 1, 'relu'),
            ('down', 'Downsample', 16, 7, 2, 'relu'), ('conv3', 'Block 2 / Conv', 16, 7, 1, 'relu'),
            ('conv4', 'Block 2 / Linear', 16, 7, 1, 'linear'), ('add2', 'Residual 2', 16, 7, 1, 'relu')]:
            if name.startswith('add'):
                add(name, label, 'add', [c, size, size], prev, skip='stem' if name == 'add1' else 'down', activation='relu')
            else:
                add(name, label, 'conv', [c, size, size], prev, weight=name+'.weight', bias=name+'.bias', kernel=3, stride=stride, padding=1, activation=activation)
            prev = name
        add('pooled', 'Global average', 'pool', [1, 1, 16], prev, axis='spatial')
    else:
        add('patches', '16 patches', 'patches', [16, 7, 7], 'input', patch=7)
        add('tokens', 'Patch embeddings', 'embed', [1, 16, 48], 'patches', weight='embed.weight', bias='embed.bias', position='pos')
        add('n1', 'Norm 1', 'norm', [1, 16, 48], 'tokens', weight='norm1.weight', bias='norm1.bias', epsilon=1e-5, overview=False)
        add('attention', 'Attention · 3 heads', 'attention', [3, 16, 16], 'n1', weight='qkv.weight', bias='qkv.bias', heads=3, headDim=16, q='q', k='k', v='v')
        add('context', 'Weighted values', 'mix', [1, 16, 48], 'attention', values='v', heads=3, headDim=16, overview=False)
        add('projected', 'Attention projection', 'token_dense', [1, 16, 48], 'context', weight='project.weight', bias='project.bias', activation='linear', overview=False)
        add('add1', 'Attention + input', 'add', [1, 16, 48], 'projected', skip='tokens', activation='linear')
        add('n2', 'Norm 2', 'norm', [1, 16, 48], 'add1', weight='norm2.weight', bias='norm2.bias', epsilon=1e-5, overview=False)
        add('ff1', 'Feed-forward / GELU', 'token_dense', [1, 16, 96], 'n2', weight='ff1.weight', bias='ff1.bias', activation='gelu', overview=False)
        add('ff2', 'Feed-forward / Linear', 'token_dense', [1, 16, 48], 'ff1', weight='ff2.weight', bias='ff2.bias', activation='linear', overview=False)
        add('add2', 'Feed-forward + input', 'add', [1, 16, 48], 'ff2', skip='add1', activation='linear')
        add('n3', 'Final norm', 'norm', [1, 16, 48], 'add2', weight='norm3.weight', bias='norm3.bias', epsilon=1e-5, overview=False)
        add('pooled', 'Token average', 'pool', [1, 1, 48], 'n3', axis='tokens')
    add('logits', 'Logits', 'dense', [1, 1, 10], 'pooled', weight='head.weight', bias='head.bias', activation='linear')
    add('probs', 'Prediction', 'softmax', [1, 1, 10], 'logits', output=True)
    return layers


def export(kind, model, metrics):
    import onnx
    import onnxruntime as ort
    OUT.mkdir(exist_ok=True, parents=True)
    model = model.cpu().eval()
    layers = specs(kind)
    names = [s['tensor'] for s in layers[1:]] + (['q', 'k', 'v'] if kind == 'vit' else [])
    path = OUT / f'mnist_{kind}.onnx'
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), str(path), input_names=['input'], output_names=names,
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    errors = {}
    rng = np.random.default_rng(SEED)
    inputs = [np.zeros((1,1,28,28), np.float32), np.ones((1,1,28,28), np.float32), rng.random((1,1,28,28), dtype=np.float32)]
    border = np.zeros_like(inputs[0]); border[0,0,0,0] = 1; inputs.append(border)
    for x in inputs:
        with torch.no_grad(): expected = model(torch.from_numpy(x))
        for name, ref, got in zip(names, expected, session.run(None, {'input': x})):
            np.testing.assert_allclose(got, ref.numpy(), atol=3e-5, rtol=4e-4, err_msg=name)
            errors[name] = max(errors.get(name, 0), float(np.max(np.abs(ref.numpy()-got))))
    manifest = dict(id=f'mnist_{kind}', name='Tiny ResNet' if kind == 'resnet' else 'Tiny ViT', architecture=kind,
        file=path.name, input=dict(name='input', shape=[1,1,28,28], image=[28,28]), layers=layers,
        params=sum(p.numel() for p in model.parameters()), training=metrics,
        weights=dict(file=f'mnist_{kind}.weights.bin', tensors={}),
        auxiliary=['q','k','v'] if kind == 'vit' else [], verification=dict(inputs=4, maxAbsoluteError=errors))
    offset = 0
    with (OUT / manifest['weights']['file']).open('wb') as f:
        for name, tensor in model.state_dict().items():
            value = tensor.numpy()
            if value.ndim == 2: value = value.T.copy()  # browser dense weights are [input, output]
            manifest['weights']['tensors'][name] = dict(offset=offset, shape=list(value.shape))
            f.write(value.astype('<f4').tobytes()); offset += value.size
    manifest['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / f'mnist_{kind}.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'EXPORTED {kind}: {manifest["params"]} parameters, all {len(names)} outputs verified', flush=True)


def augment(x):
    b = len(x)
    angle = (torch.rand(b, device=x.device)-.5) * math.radians(24)
    scale = .9 + torch.rand(b, device=x.device)*.2
    affine = torch.zeros(b, 2, 3, device=x.device)
    affine[:,0,0] = angle.cos()*scale; affine[:,1,1] = angle.cos()*scale
    affine[:,0,1] = -angle.sin()*scale; affine[:,1,0] = angle.sin()*scale
    affine[:,:,2] = (torch.rand(b,2,device=x.device)-.5)*.28
    return F.grid_sample(x, F.affine_grid(affine, x.shape, align_corners=False), align_corners=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=['resnet','vit','both'], default='both')
    parser.add_argument('--epochs', type=int, default=40); parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(6)
    ART.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'DEVICE {device}', flush=True)
    train = MNIST(str(ART/'data'), train=True, download=True)
    test = MNIST(str(ART/'data'), train=False, download=True)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(SEED))
    train_x = train.data[order[:55000]].float().unsqueeze(1).to(device)/255
    train_y = train.targets[order[:55000]].to(device)
    val_x = train.data[order[55000:]].float().unsqueeze(1).to(device)/255
    val_y = train.targets[order[55000:]].to(device)
    test_x = test.data.float().unsqueeze(1).to(device)/255; test_y = test.targets.to(device)
    def accuracy(model, x, y):
        correct = 0
        with torch.no_grad():
            for start in range(0,len(x),512):
                correct += int((model(x[start:start+512])[-(4 if isinstance(model,TinyViT) else 1)].argmax(-1)==y[start:start+512]).sum())
        return correct/len(x)
    for kind in (['resnet','vit'] if args.model == 'both' else [args.model]):
        torch.manual_seed(SEED)
        model = (TinyResNet() if kind == 'resnet' else TinyViT()).to(device)
        checkpoint = ART/f'{kind}.pt'; report = ART/f'{kind}.json'
        if not args.export_only:
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0001)
            best = 0; history = []; started = time.time()
            for epoch in range(args.epochs):
                model.train(); perm=torch.randperm(len(train_x), device=device); loss_sum=0
                for ids in perm.split(256):
                    optimizer.zero_grad(set_to_none=True)
                    outputs=model(augment(train_x[ids])); logits=outputs[-(5 if kind=='vit' else 2)]
                    loss=F.cross_entropy(logits,train_y[ids]); loss.backward(); optimizer.step(); loss_sum += float(loss.detach())*len(ids)
                scheduler.step(); model.eval(); val=accuracy(model,val_x,val_y)
                history.append(dict(epoch=epoch+1, loss=loss_sum/55000, validationAccuracy=val))
                if val > best:
                    best=val; torch.save(model.state_dict(), checkpoint)
                print(f'{kind} epoch={epoch+1} loss={loss_sum/55000:.4f} val={val:.4%} elapsed={time.time()-started:.0f}s',flush=True)
            model.load_state_dict(torch.load(checkpoint, weights_only=True)); model.eval()
            score=accuracy(model,test_x,test_y)
            metrics=dict(dataset='MNIST', trainExamples=55000, validationExamples=5000, testExamples=10000,
                seed=SEED, epochs=args.epochs, selectedBy='best validation accuracy', validationAccuracy=best,
                testAccuracy=score, augmentation='rotation ±12°, translation ±2px, scale 0.9–1.1',
                input='28x28 grayscale [0,1]; no centering or deskew at inference', history=history,
                torchVersion=torch.__version__, device=str(device))
            report.write_text(json.dumps(metrics,indent=2)+'\n')
            print(f'TEST {kind}: {score:.4%} on 10000 held-out images',flush=True)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        export(kind,model,json.loads(report.read_text()))

if __name__ == '__main__': main()
