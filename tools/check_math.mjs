import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {convolution, denseTerms, indexOf, channels, intensity, validate} from '../web/assets/math.mjs';
import {easeInOut, perspectiveCenter} from '../web/assets/motion.mjs';
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
// Camera easing remains bounded and smooth; tensor updates have no playback clock.
assert.equal(easeInOut(-1),0);assert.equal(easeInOut(2),1);
assert.ok(easeInOut(.01)<.00001);assert.ok(1-easeInOut(.99)<.00001);
console.log('PASS: camera easing endpoints');
// A large near input and small distant outputs must balance in screen space.
const corners=[{x:-9,y:-7,z:12},{x:3,y:5,z:12},{x:15,y:2,z:-15},{x:10,y:-1,z:-15}];
const distance=40,center=perspectiveCenter(corners,distance);
for(const [i,axis] of ['x','y'].entries()){
  const projected=corners.map(p=>(p[axis]-center[i])/(distance-p.z));
  assert.ok(Math.abs(Math.min(...projected)+Math.max(...projected))<1e-8);
}
console.log('PASS: perspective camera centering');
// Overview wiring uses fixed learned-weight ranks even when the drawing changes.
const weights=[1,-5,3,-4,2];
const fixed=denseTerms([0,1,2,3,4],weights,1,0,'weights',4);
assert.deepEqual(fixed.map(t=>t.i),[1,3,2,4]);
assert.deepEqual(denseTerms([99,0,0,0,0],weights,1,0,'weights',4).map(t=>t.i),fixed.map(t=>t.i));
assert.equal(fixed[0].contribution,-5);
console.log('PASS: stable overview wiring, signed input contributions');
