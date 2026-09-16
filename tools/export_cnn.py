"""Export okdalto's four-convolution CNN, preserving its trained parameters.

uv run --with numpy --with onnx --with onnxruntime tools/export_cnn.py
The independent NumPy forward checks every output, including padded borders.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reference/cnn/conv/data'
OUT = ROOT / 'web/models'
OUT.mkdir(parents=True, exist_ok=True)


def read(name, conv=False):
    lines = (DATA / name).read_text().strip().splitlines()
    if conv:
        return np.array([[[[float(v) for v in row.split()] for row in ch.split(',')]
                          for ch in line.split('!')] for line in lines], np.float32)
    return np.array([[float(v) for v in line.split()] for line in lines], np.float32)


params, nodes, layers, outputs = {}, [], [], []
layers.append(dict(tensor='input', label='Input', op='input', shape=[1, 32, 32]))
previous = 'input'
for i, (cin, cout, size) in enumerate([(1, 16, 16), (16, 32, 8), (32, 32, 4), (32, 32, 2)], 1):
    w, b = read(f'conv{i}Weight.txt', True), read(f'conv{i}Bias.txt').ravel()
    assert w.shape == (cout, cin, 3, 3) and b.shape == (cout,)
    params[f'C{i}'] = w
    params[f'CB{i}'] = b
    tensor = f'conv{i}'
    nodes += [helper.make_node('Conv', [previous, f'C{i}', f'CB{i}'], [f'z{i}'],
                               strides=[2, 2], pads=[1, 1, 1, 1]),
              helper.make_node('Relu', [f'z{i}'], [tensor])]
    layers.append(dict(tensor=tensor, label=f'Conv {i}', op='conv', shape=[cout, size, size],
                       source=previous, weight=f'C{i}', bias=f'CB{i}', kernel=3, stride=2, padding=1))
    outputs.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, [1, cout, size, size]))
    previous = tensor
nodes.append(helper.make_node('Flatten', [previous], ['flat'], axis=1))
layers.append(dict(tensor='flat', label='Flatten', op='flatten', shape=[1, 1, 128], source=previous))
outputs.append(helper.make_tensor_value_info('flat', TensorProto.FLOAT, [1, 128]))
previous = 'flat'
for i, cout in [(1, 128), (2, 10)]:
    params[f'W{i}'] = read(f'mlp{i}Weight.txt').T.copy()
    params[f'B{i}'] = read(f'mlp{i}Bias.txt').ravel()
    tensor = 'dense' if i == 1 else 'logits'
    nodes.append(helper.make_node('Gemm', [previous, f'W{i}', f'B{i}'], [tensor if i == 2 else 'dense_z']))
    if i == 1:
        nodes.append(helper.make_node('Relu', ['dense_z'], ['dense']))
    layers.append(dict(tensor=tensor, label='Dense' if i == 1 else 'Logits', op='dense',
                       shape=[1, 8, 16] if i == 1 else [1, 1, 10], source=previous,
                       weight=f'W{i}', bias=f'B{i}', activation='relu' if i == 1 else 'linear'))
    outputs.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, [1, cout]))
    previous = tensor
nodes.append(helper.make_node('Softmax', ['logits'], ['probs'], axis=1))
layers.append(dict(tensor='probs', label='Prediction', op='softmax', shape=[1, 1, 10], source='logits', output=True))
outputs.append(helper.make_tensor_value_info('probs', TensorProto.FLOAT, [1, 10]))
graph = helper.make_graph(nodes, 'okdalto_cnn',
    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 32, 32])], outputs,
    [numpy_helper.from_array(v, k) for k, v in params.items()])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT / 'mnist_cnn.onnx')
manifest = dict(id='mnist_cnn', name='Convolutional', subtitle='Four convolutions. One decision.',
    file='mnist_cnn.onnx', input=dict(name='input', shape=[1, 1, 32, 32], image=[32, 32]),
    layers=layers, params=sum(v.size for v in params.values()),
    source='https://github.com/okdalto/CNN-visualization/tree/99010a7681264e72a6e556215d89548c138ef63f',
    weights=dict(file='mnist_cnn.weights.bin', tensors={}))
offset = 0
with (OUT / manifest['weights']['file']).open('wb') as f:
    for name, value in params.items():
        manifest['weights']['tensors'][name] = dict(offset=offset, shape=list(value.shape))
        f.write(value.astype('<f4').tobytes())
        offset += value.size
(OUT / 'mnist_cnn.json').write_text(json.dumps(manifest, indent=2))


def forward(x):
    result = []
    for i in range(1, 5):
        w, b = params[f'C{i}'], params[f'CB{i}']
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))[:, :, ::2, ::2]
        x = np.maximum(np.einsum('bcyxij,ocij->boyx', patches, w) + b[None, :, None, None], 0)
        result.append(x)
    x = x.reshape(1, -1)
    result.append(x)
    x = np.maximum(x @ params['W1'] + params['B1'], 0)
    result.append(x)
    x = x @ params['W2'] + params['B2']
    result.append(x)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    result.append(x / x.sum(axis=1, keepdims=True))
    return result


session = ort.InferenceSession(str(OUT / 'mnist_cnn.onnx'))
rng = np.random.default_rng(7)
inputs = [np.zeros((1, 1, 32, 32), np.float32), np.ones((1, 1, 32, 32), np.float32),
          rng.random((1, 1, 32, 32), dtype=np.float32)]
border = np.zeros_like(inputs[0]); border[0, 0, 0, 0] = 1
inputs.append(border)
for x in inputs:
    for spec, ref, got in zip(layers[1:], forward(x), session.run(None, {'input': x})):
        assert ref.shape == got.shape
        np.testing.assert_allclose(got, ref, atol=2e-5, rtol=2e-4, err_msg=spec['tensor'])
print('PASS: CNN, 4 inputs × 8 intermediate/final tensors; original trained weights')
