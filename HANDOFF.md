# VisualizeMNIST / Neural observatory

Updated: 2026-09-16. This replaces the earlier prototype handoff.

## Product

A browser-based neural network observatory with four **real trained models**:

- Original MLP: 784 → 128 → 16 → 10 logits → softmax.
- Original CNN: 1×32×32 → 16×16×16 → 32×8×8 → 32×4×4 → 32×2×2
  → flatten 128 → dense 128 → 10 logits → softmax.

The two original models preserve okdalto's supplied weights; no new dataset
accuracy is claimed for them. Tiny ResNet (7,226 parameters, two residual blocks)
and Tiny ViT (22,714 parameters, 16 patches, three attention heads) were trained
on MNIST. Both score 98.45% on the 10,000-image test split. See `MODEL_CARD.md`
for architecture, split, training, verification and limitations. CNN source code uses four convolutions despite its README
mentioning three. Attribution and original CNN sources are in `reference/cnn`.

## Run and verify

```
python3 -m http.server 8765 --directory web
node tools/check_math.mjs
uv run --with numpy --with onnx --with onnxruntime tools/export_mlp.py
uv run --with numpy --with onnx --with onnxruntime tools/export_cnn.py
uv run --with numpy --with onnx --with onnxruntime tools/check_new_models.py
```

No bundler or npm install. Three.js and ONNX Runtime Web are pinned in the
import map. Inference attempts WebGPU, then WASM; rendering is Three.js WebGL.
All graph outputs are actual model values. Feature maps update together as each
inference completes. There is no explanatory playback or transfer animation.

## Source map

- `web/assets/app.js`: drawing, race-safe model/inference lifecycle, inspector,
  live activation updates, UI.
- `web/assets/scene.js`: instanced 3D cells, channel stacks/contact sheets,
  receptive fields, static connection lines, picking.
- `web/assets/motion.mjs`: camera easing and perspective-aware centering.
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
  channels of the final 2×2 convolution so all 128 values can be inspected.
- Input and softmax brightness are absolute [0,1]. Hidden activations use
  `1-exp(-abs(value)*exposure)` with a fixed user-controlled exposure, never
  per-inference max normalization. Negative logits use dark fill; raw signed
  values are available in the inspector.
- Conv inspector computes the full all-channel dot product plus bias and ReLU;
  it shows the error against the actual ONNX value. The 3×3 mini-preview shows
  the input channel with greatest absolute contribution (labeled).
- Dense inspector shows top 12 |x*w| or |w| for one selected neuron; solid positive
  weight, dashed negative. The detail table shows the first six contributions.
- Drawing updates all visible layers and predictions directly, with no scan,
  transfer particles, playback delay, activation fade, auto layer changes or replay
  controls. Completed inference snapshots can display during a continuous stroke;
  stale results after clear or pointer release are discarded. Weights stay fixed.
- Overview has static thin connection lines inspired by the original Processing
  project. Dense layers show four strongest learned weights per neuron; conv layers
  show real receptive fields at four fixed spatial samples per visible output
  channel, using visible input channels. Flatten uses exact matching indices.
  Softmax has no faux learned edges. Line brightness tracks absolute input contribution.
- Explore computation is open by default and exposes the layer,
  channel, focus, connection and exposure controls; picking a cell shows its exact
  calculation. Selected receptive fields/dense terms overlay the overview wiring.
- User-triggered camera framing still eases over 0.9s; pointer interaction cancels
  it and reduced motion makes it immediate. Drawing preserves camera and selection.
- Input is at the +Z end of the network. Overview looks from the input side;
  both overview and focus read columns left to right, matching the drawing pad.
  Framing centers projected layer bounds rather than the 3D bounding-box center.
- Zero-activation convolution cells retain their grid and picking geometry but
  discard their black fill, so empty front channels do not obscure deeper maps.

## Design and deployment

Compact header: project name and architecture selector only. No tagline or
visible inspiration footer; source attribution remains in `THIRD_PARTY.md`.

English-only RAS family: black, monochrome, Gridnik, fine rules. No language
switch. Font binaries are ignored by Git and restored into the Pages artifact
from the existing deployment-only `GRIDNIK_REGULAR`/`GRIDNIK_LIGHT` secrets.
Forks without those assets can use system monospace. Do not commit font files.

Own repository: https://github.com/cobanov/VisualizeMNIST
Upstream: https://github.com/okdalto/VisualizeMNIST
Pages: https://mnist.cobanov.dev/

GitHub Pages custom domain: `mnist.cobanov.dev`. Cloudflare has a DNS-only CNAME
from `mnist` to `cobanov.github.io`. Deployment uses Actions; the custom domain
is configured in repository Pages settings, with no `CNAME` file required.

## Architecture extensions and limits

`tools/train_models.py` trains/exports the compact ResNet and ViT.
`tools/check_new_models.py` independently reconstructs every new-model tensor.
`web/assets/operations.mjs` supplies scalar token/attention/normalization helpers.
The manifest includes add, spatial/token average, patches, embedding, norm,
attention, value mixing and token-linear operations. All are batch-one digit
models, not arbitrary ONNX graph support.

ViT uses a three-column reading layout with all patches and heads visible.
Overview omits norm/projection/FFN intermediates; selecting one inserts it and
Focus shows it alone. ResNet retains channel stacks, with sampled identity arcs.
No decorative connections cross hidden operations. New-model signed values use
magnitude brightness, with exact sign and selected-operation errors in the
inspector. Attention uses sqrt(probability) brightness; its query arcs represent
attention probabilities, not attribution or static learned edges.

Preserve immediate updates, original model weights, input-facing centered camera,
minimal header and the default-open inspector. Future architectures require
trained weights, explicit operation semantics and independent parity checks.
