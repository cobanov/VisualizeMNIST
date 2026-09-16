# VisualizeMNIST / Neural observatory

Updated: 2026-09-16. This replaces the earlier prototype handoff.

## Product

A browser-based neural network observatory with two **real pretrained models**:

- Original MLP: 784 → 128 → 16 → 10 logits → softmax.
- Original CNN: 1×32×32 → 16×16×16 → 32×8×8 → 32×4×4 → 32×2×2
  → flatten 128 → dense 128 → 10 logits → softmax.

Both preserve okdalto's supplied weights. No new training or measured dataset
accuracy is claimed. CNN source code uses four convolutions despite its README
mentioning three. Attribution and original CNN sources are in `reference/cnn`.

## Run and verify

```
python3 -m http.server 8765 --directory web
node tools/check_math.mjs
uv run --with numpy --with onnx --with onnxruntime tools/export_mlp.py
uv run --with numpy --with onnx --with onnxruntime tools/export_cnn.py
```

No bundler or npm install. Three.js and ONNX Runtime Web are pinned in the
import map. Inference attempts WebGPU, then WASM; rendering is Three.js WebGL.
All graph outputs are actual model values. Animation is explanatory playback,
not a measurement of GPU operation timing.

## Source map

- `web/assets/app.js`: drawing, race-safe model/inference lifecycle, inspector,
  operation playback, guided tour, UI.
- `web/assets/scene.js`: instanced 3D cells, channel stacks/contact sheets,
  receptive fields, moving patches, flatten morph, dense connections, picking.
- `web/assets/math.mjs`: pure NCHW indexing, manifest checks, convolution
  reconstruction, dense contribution ranking, activation transfer function.
- `tools/check_math.mjs`: runnable numeric checks without a test framework.
- `tools/export_*.py`: reproducible export plus independent NumPy equivalence
  checks for all intermediate/final tensors.
- `.github/workflows/pages.yml`: check, export and publish `web/` to Pages.

## Visual semantics

- Overview shows up to six evenly spaced channels, explicitly labeled as a
  subset. The selected output channel is inserted when needed. Focus reveals
  every channel as a labeled contact sheet. Selecting Flatten exposes all 32
  channels of the final 2×2 convolution so all 128 values can move.
- Input and softmax brightness are absolute [0,1]. Hidden activations use
  `1-exp(-abs(value)*exposure)` with a fixed user-controlled exposure, never
  per-inference max normalization. Negative logits use dark fill; raw signed
  values are available in the inspector.
- Conv inspector computes the full all-channel dot product plus bias and ReLU;
  it shows the error against the actual ONNX value. The 3×3 mini-preview shows
  the input channel with greatest absolute contribution (labeled).
- Dense view shows top 12 |x*w| or |w| for one selected neuron; solid positive
  weight, dashed negative. The detail table shows the first six contributions.
- Play/pause, step, scrub and speed are operation-local. Follow signal cycles
  through stages. Reduced motion starts paused. Layer buttons and sliders
  provide keyboard alternatives to picking cells.

## Design and deployment

English-only RAS family: black, monochrome, Gridnik, fine rules. No language
switch. Font binaries are ignored by Git and restored into the Pages artifact
from the existing deployment-only `GRIDNIK_REGULAR`/`GRIDNIK_LIGHT` secrets.
Forks without those assets can use system monospace. Do not commit font files.

Own repository: https://github.com/cobanov/VisualizeMNIST
Upstream: https://github.com/okdalto/VisualizeMNIST
Pages: https://cobanov.github.io/VisualizeMNIST/

## Deliberate limits and next architecture

The manifest supports batch-one digit models and input/conv/dense/flatten/
softmax operations. It is not an arbitrary ONNX graph viewer. A new architecture
needs explicit operation semantics, trained weights, preprocessing and numeric
parity checks before it can be offered in the UI.

Next useful additions: pooling-based CNN, then a small residual network. Treat
pooling/skip connections as first-class operations. Attention needs its own
representation, not another voxel grid. Avoid adding decorative connections
that do not correspond to the model. No new architecture is implemented merely
because it appears on this list.
