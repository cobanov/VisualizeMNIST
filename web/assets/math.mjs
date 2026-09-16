// Tensor indexing is NCHW with batch=1. Presentation never changes raw values.
export const sizeOf = shape => shape.reduce((a, b) => a * b, 1);
export const indexOf = (shape, c, y, x) => (c * shape[1] + y) * shape[2] + x;
export const channels = (count, limit = 6) => Array.from({length: Math.min(count, limit)}, (_, i) =>
  count <= limit ? i : Math.round(i * (count - 1) / (limit - 1)));
export const intensity = (value, op, exposure = 1) =>
  op === 'input' || op === 'softmax' ? Math.max(0, Math.min(1, value)) :
  1 - Math.exp(-Math.abs(value) * exposure);

export function validate(manifest) {
  if (!Array.isArray(manifest.layers) || !manifest.layers.length) throw Error('Missing layers');
  const known = new Map();
  for (const s of manifest.layers) {
    if (!['input', 'conv', 'dense', 'flatten', 'softmax'].includes(s.op)) throw Error(`Unsupported operation: ${s.op}`);
    if (known.has(s.tensor) || s.shape?.length !== 3 || !s.shape.every(n => Number.isInteger(n) && n > 0)) throw Error('Invalid tensor shape or name');
    if (s.op !== 'input' && !known.has(s.source)) throw Error(`Missing source: ${s.source}`);
    if (s.op === 'conv') {
      const src = known.get(s.source), w = manifest.weights.tensors[s.weight]?.shape;
      if (!w || w.join() !== [s.shape[0], src.shape[0], s.kernel, s.kernel].join() ||
          ![1, 2].every(d => Math.floor((src.shape[d] + 2 * s.padding - s.kernel) / s.stride) + 1 === s.shape[d])) throw Error('Invalid convolution shape');
    }
    if (s.op === 'dense') {
      const w = manifest.weights.tensors[s.weight]?.shape;
      if (!w || w.join() !== [sizeOf(known.get(s.source).shape), sizeOf(s.shape)].join()) throw Error('Invalid dense shape');
    }
    if (s.op === 'flatten' && sizeOf(s.shape) !== sizeOf(known.get(s.source).shape)) throw Error('Invalid flatten shape');
    known.set(s.tensor, s);
  }
  if (manifest.input.shape[0] !== 1 || sizeOf(manifest.input.shape) !== sizeOf(manifest.layers[0].shape)) throw Error('Only batch-one inputs are supported');
  if (manifest.layers.at(-1).op !== 'softmax' || sizeOf(manifest.layers.at(-1).shape) !== 10) throw Error('This digit viewer requires 10 output classes');
}

export function convolution(spec, sourceShape, input, weights, bias, channel, y, x) {
  const terms = [];
  let sum = bias[channel];
  for (let c = 0; c < sourceShape[0]; c++) for (let ky = 0; ky < spec.kernel; ky++) for (let kx = 0; kx < spec.kernel; kx++) {
    const sy = y * spec.stride + ky - spec.padding, sx = x * spec.stride + kx - spec.padding;
    const padding = sy < 0 || sx < 0 || sy >= sourceShape[1] || sx >= sourceShape[2];
    const value = padding ? 0 : input[indexOf(sourceShape, c, sy, sx)];
    const weight = weights[((channel * sourceShape[0] + c) * spec.kernel + ky) * spec.kernel + kx];
    const contribution = value * weight;
    sum += contribution;
    terms.push({c, y: sy, x: sx, padding, value, weight, contribution});
  }
  return {terms, sum, value: Math.max(0, sum), bias: bias[channel]};
}

export function denseTerms(input, weights, countOut, target, mode = 'contribution', limit = 12) {
  return Array.from(input, (value, i) => {
    const weight = weights[i * countOut + target];
    return {i, weight, value, contribution: weight * value};
  }).sort((a, b) => Math.abs(mode === 'weights' ? b.weight : b.contribution) -
    Math.abs(mode === 'weights' ? a.weight : a.contribution)).slice(0, limit);
}
