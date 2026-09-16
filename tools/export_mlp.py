"""Convert the Processing sketch's txt weights into an ONNX MLP.

Every layer's activation is exposed as a graph output so the web
visualizer gets all intermediate tensors from a single run().
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "VisualizeMnist" / "data"
OUT = ROOT / "web" / "models"


def load(name):
    # rows end with a trailing comma
    rows = [l.strip().rstrip(",") for l in (DATA / name).read_text().splitlines() if l.strip()]
    return np.array([[float(v) for v in r.split(",")] for r in rows], dtype=np.float32)


w1, w2, w3 = load("weight1.txt"), load("weight2.txt"), load("weight3.txt")
b1, b2, b3 = load("biases1.txt")[0], load("biases2.txt")[0], load("biases3.txt")[0]
assert w1.shape == (784, 128) and w2.shape == (128, 16) and w3.shape == (16, 10)

inits = [
    numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(b1, "B1"),
    numpy_helper.from_array(w2, "W2"), numpy_helper.from_array(b2, "B2"),
    numpy_helper.from_array(w3, "W3"), numpy_helper.from_array(b3, "B3"),
]
nodes = [
    helper.make_node("Gemm", ["input", "W1", "B1"], ["z1"]),
    helper.make_node("Relu", ["z1"], ["h1"]),
    helper.make_node("Gemm", ["h1", "W2", "B2"], ["z2"]),
    helper.make_node("Relu", ["z2"], ["h2"]),
    helper.make_node("Gemm", ["h2", "W3", "B3"], ["logits"]),
    helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
]
t = lambda name, n: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, n])
graph = helper.make_graph(nodes, "mnist_mlp", [t("input", 784)],
                          [t("h1", 128), t("h2", 16), t("logits", 10), t("probs", 10)], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT / "mnist_mlp.onnx")

# What the visualizer draws, in order. grid = how a flat vector is tiled.
manifest = {
    "id": "mnist_mlp",
    "name": "Fully connected",
    "subtitle": "Every pixel. Every connection.",
    "params": sum(v.size for v in (w1, w2, w3, b1, b2, b3)),
    "source": "https://github.com/okdalto/VisualizeMNIST",
    "file": "mnist_mlp.onnx",
    "input": {"name": "input", "shape": [1, 784], "image": [28, 28]},
    "layers": [
        {"tensor": "input", "label": "Input", "op": "input", "shape": [1, 28, 28]},
        {"tensor": "h1", "label": "Dense 128", "op": "dense", "shape": [1, 8, 16], "source": "input", "weight": "W1", "bias": "B1", "activation": "relu"},
        {"tensor": "h2", "label": "Dense 16", "op": "dense", "shape": [1, 4, 4], "source": "h1", "weight": "W2", "bias": "B2", "activation": "relu"},
        {"tensor": "logits", "label": "Logits", "op": "dense", "shape": [1, 1, 10], "source": "h2", "weight": "W3", "bias": "B3", "activation": "linear"},
        {"tensor": "probs", "label": "Prediction", "op": "softmax", "shape": [1, 1, 10], "source": "logits", "output": True},
    ],
    "weights": {"file": "mnist_mlp.weights.bin", "tensors": {}},
}
off = 0
with open(OUT / "mnist_mlp.weights.bin", "wb") as f:
    for name, w in (("W1", w1), ("W2", w2), ("W3", w3), ("B1", b1), ("B2", b2), ("B3", b3)):
        f.write(w.astype("<f4").tobytes())
        manifest["weights"]["tensors"][name] = {"offset": off, "shape": list(w.shape)}
        off += w.size
(OUT / "mnist_mlp.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

# Independent NumPy forward, including every intermediate graph output.
import onnxruntime as ort
session = ort.InferenceSession(str(OUT / "mnist_mlp.onnx"))
rng = np.random.default_rng(7)
for x in (np.zeros((1, 784), np.float32), np.ones((1, 784), np.float32), rng.random((1, 784), dtype=np.float32)):
    h1 = np.maximum(x @ w1 + b1, 0)
    h2 = np.maximum(h1 @ w2 + b2, 0)
    logits = h2 @ w3 + b3
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    for expected, got in zip((h1, h2, logits, probs), session.run(None, {"input": x})):
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
print("PASS: MLP, 3 inputs x 4 intermediate/final tensors; original trained weights")
