# Neural observatory / VisualizeMNIST

[Open the interactive observatory](https://cobanov.github.io/VisualizeMNIST/)

Draw a digit and explore two real pretrained networks in 3D. Follow a convolution,
inspect every feature-map channel, watch a tensor flatten, and trace individual
weighted contributions. Includes pause, step, scrub, a guided signal tour and a
responsive interface. ONNX Runtime runs inference with WebGPU or WASM; Three.js
renders the scene with WebGL.

Built on [okdalto/VisualizeMNIST](https://github.com/okdalto/VisualizeMNIST) and
[okdalto/CNN-visualization](https://github.com/okdalto/CNN-visualization), preserving
their trained weights. See [attribution](THIRD_PARTY.md) and the
[development handoff](HANDOFF.md) for tensor semantics, checks and limitations.

```sh
python3 -m http.server 8765 --directory web
node tools/check_math.mjs
```

Open http://localhost:8765. Private Gridnik font files are optional; absent files
fall back to system monospace. No npm installation or build step is required.

## Original Processing project

# VisualizeMnist
This project is real-time visualization of a network recognizing digits from user's input.

![텍스트](Images/visualizeMNIST.gif) 

[Youtube link](https://youtu.be/WQYCK1YpsjE?t=0s)

[Interactive web demo](https://okdalto.github.io/VisualizeMNIST_web)

I trained a network using MNIST dataset and parsed the weight data in python. With this data, I implemented my own custom functions that are needed to run the network in Processing including matrix multiplication function, activation functions. At first trial, because MNIST dataset is preprocessed for numbers to be in the center of the images, there was a precision problem when the user's input is placed little bit far away from the center. I used data augmentation technic in the training process to resolve this problem. 


# Installation
To run this code, you need [Processing](https://www.processing.org/download/) IDE and a library named [peasycam](http://mrfeinberg.com/peasycam/).

# Instagram
If you want to see more of my work, check my [Instagram](https://www.instagram.com/okdalto/)
