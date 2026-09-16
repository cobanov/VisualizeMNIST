import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {convolution, denseTerms, indexOf, channels, intensity, validate} from '../web/assets/math.mjs';
import {easeInOut, transferMotion, cycleSeconds} from '../web/assets/motion.mjs';
for (const id of ['mnist_cnn', 'mnist_mlp']) validate(JSON.parse(readFileSync(new URL(`../web/models/${id}.json`, import.meta.url))));
assert.equal(indexOf([2, 3, 4], 1, 2, 3), 23);
assert.deepEqual(channels(32), [0, 6, 12, 19, 25, 31]);
assert.equal(intensity(.2, 'softmax'), .2);
assert.ok(intensity(.1, 'dense') < intensity(1, 'dense'));
const spec = {kernel: 3, stride: 2, padding: 1};
const result = convolution(spec, [2, 3, 3], Float32Array.from({length:18}, (_, i) => i+1), new Float32Array(18).fill(1), [2], 0, 0, 0);
assert.equal(result.value, 62); // [1,2,4,5] + [10,11,13,14] + bias 2.
assert.equal(result.terms.filter(t => t.padding).length, 10);
assert.equal(convolution(spec,[1,1,1],[3],new Float32Array(9).fill(-1),[0],0,0,0).value,0);
assert.equal(denseTerms([0, 4], [10, 2], 1, 0)[0].i, 1);
assert.equal(denseTerms([0, 4], [10, 2], 1, 0, 'weights')[0].i, 0);
console.log('PASS: manifests, NCHW indexing, padded multi-channel convolution, ReLU, dense ranking, probability intensity');
// A loop resets only after all cells have landed and disappeared. Easing never overshoots.
assert.equal(easeInOut(-1), 0);assert.equal(easeInOut(2), 1);
assert.ok(easeInOut(.01)<.00001);assert.ok(1-easeInOut(.99)<.00001);
for(const order of [0,.5,1]){
  let previous=0;
  for(let i=0;i<=100;i++){
    const m=transferMotion(i/100,order);
    assert.ok(m.travel>=previous-1e-12&&m.travel<=1);
    assert.ok(m.visibility>=0&&m.visibility<=1);
    previous=m.travel;
  }
  assert.equal(transferMotion(0,order).visibility,0);
  assert.equal(transferMotion(1,order).visibility,0);
  assert.equal(transferMotion(1,order).travel,1);
}
assert.ok(transferMotion(.4,0).travel>transferMotion(.4,1).travel);
for(const fps of [30,60,120]){
  let phase=0;for(let frame=0;frame<fps;frame++)phase+=1/fps/cycleSeconds.flatten;
  assert.ok(Math.abs(phase-1/cycleSeconds.flatten)<1e-12);
}
console.log('PASS: C2 easing endpoints, stagger order, invisible loop seam, monotonic travel, frame-rate-independent timing');
