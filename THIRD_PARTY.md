# Attribution

This project builds on **okdalto / VisualizeMNIST** (GPL-3.0, see `LICENSE`).
The original Processing sketch and trained MLP weights remain in `VisualizeMnist/`.

The trained CNN and its Processing reference implementation come from
**okdalto / CNN-visualization**, commit
`99010a7681264e72a6e556215d89548c138ef63f` (LGPL-3.0).
The original source, weights and license are preserved under `reference/cnn/`.
`tools/export_cnn.py` converts those weights to ONNX, without retraining.

- https://github.com/okdalto/VisualizeMNIST
- https://github.com/okdalto/CNN-visualization

Three.js (MIT) and ONNX Runtime Web (MIT) are loaded from pinned jsDelivr URLs.
Foundry Gridnik font files are private deployment assets and are excluded from
this repository. Without those assets the application uses system monospace.
