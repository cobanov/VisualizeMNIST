# Tiny MNIST models

Two compact educational architectures, trained from scratch for this viewer.
They are not standard ResNet-18 or ViT-Base checkpoints. The original CNN and
MLP retain their upstream weights and are not part of these accuracy results.

| Model | Parameters | Validation (5,000) | Test (10,000) |
| --- | ---: | ---: | ---: |
| Tiny ResNet | 7,226 | 98.24% | 98.45% |
| Tiny ViT | 22,714 | 98.14% | 98.45% |

Training: MNIST, fixed seed 71, 55,000 training images, 5,000 validation images
from the original training partition. Forty epochs, batch 256, AdamW (initial
learning rate 0.002, weight decay 0.0001), cosine schedule to 0.0001. Random
affine augmentation uses rotation ±12°, normalized translation ±0.14 (about
±2 pixels), affine scale 0.9–1.1. Checkpoints are selected by validation
accuracy, then evaluated on the official 10,000-image test split. An initial
15-epoch run preceded the final 40-epoch run; no test examples enter training
or checkpoint selection. Reported test performance is not a blind benchmark
against further development. Training ran on an RTX 5090 with PyTorch 2.8.0.

Input: 28×28 grayscale floats in [0,1]. No centering, deskew, or contrast
normalization at inference. Canvas drawings can differ substantially from
MNIST; these scores do not measure arbitrary handwriting on the drawing pad.

## Architecture

**ResNet:** 3×3 stride-two stem (1→8), two 8-channel convolutions with an
identity shortcut, stride-two convolution (8→16), two 16-channel convolutions
with an identity shortcut, spatial average, 16→10 head, softmax. Residual
branches preserve signed values before addition; ReLU follows the sum.
No batch normalization or projection shortcuts.

**ViT:** sixteen nonoverlapping 7×7 patches, linear 49→48 embedding and learned
positional embedding, one pre-norm Transformer block with three attention heads
(16 features/head), 48→96→48 GELU feed-forward branch, final LayerNorm, token
average, 48→10 head, softmax. Both branches use identity residual additions.
No class token. Every attention row is softmax(QKᵀ/√16).

## Visualization and verification

All displayed tensors come from the trained ONNX graph. The inspector
reconstructs selected convolution, residual, patch, linear, normalization,
attention, value-mixing and averaging operations from the same weights/values.
GELU inspection uses a small erf approximation within float32 tolerance.

ResNet arcs show sampled identity connections between matching cell indices.
Other overview lines are a fixed, explicitly incomplete subset of real terms.
ViT shows all 16 patches and all three attention matrices. Selecting a head and
query traces its strongest four attention relationships, including self in the
ranking (self does not draw an arc). Attention weights are relationships, not
causal attribution or learned static weights. Signed new-model activations use
magnitude brightness; inspect cells for their signs. Norm/projection/FFN
intermediates are omitted from the default ViT overview but remain selectable
in Layer and Focus. The overview is a reading layout, not a complete graph.

Export verifies every intermediate against PyTorch on four inputs. Independent
NumPy reconstruction checks all 11 ResNet and 18 ViT outputs against ONNX Runtime
on zeros, ones, random noise and a corner pixel. Manifests record ONNX hashes,
training history and metrics. JavaScript tests separately cover patch indexing,
linear orientation, normalization, scaled attention, GELU and residual shapes.

```sh
uv run --python 3.12 --with torch==2.8.0 --with torchvision==0.23.0 \
  --with onnx==1.19.0 --with onnxruntime==1.22.1 tools/train_models.py
uv run --python 3.12 --with numpy==2.2.6 --with onnx==1.19.0 \
  --with onnxruntime==1.22.1 tools/check_new_models.py
node tools/check_math.mjs
```

Training data/checkpoints stay in ignored `artifacts/training/`. Committed ONNX,
manifest and binary weights are sufficient to serve and validate the models.
CUDA kernels and platform differences can prevent bit-identical retraining.

Architecture references: [Deep Residual Learning](https://arxiv.org/abs/1512.03385),
[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929).
Dataset loader: [torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
