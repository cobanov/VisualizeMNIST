import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {convolution, denseTerms, indexOf, channels, intensity, validate} from '../web/assets/math.mjs';
import {easeInOut, perspectiveCenter} from '../web/assets/motion.mjs';
for (const id of ['mnist_cnn', 'mnist_mlp', 'mnist_resnet', 'mnist_vit']) validate(JSON.parse(readFileSync(new URL(`../web/models/${id}.json`, import.meta.url))));
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

// Patch layout must be a bijection of the 784 input pixels.
const {patchIndex, tokenTerms, normValue, attentionRow, activate}=await import('../web/assets/operations.mjs');
const pixels=[];for(let p=0;p<16;p++)for(let y=0;y<7;y++)for(let x=0;x<7;x++)pixels.push(patchIndex(p,y,x));
assert.deepEqual([...pixels].sort((a,b)=>a-b),Array.from({length:784},(_,i)=>i));
assert.equal(patchIndex(5,0,0),203);
assert.equal(tokenTerms([1,2,3,4],[2,0,0,3],1,2,2,1).reduce((sum,t)=>sum+t.contribution,0),12);
const norm=normValue(new Float32Array([1,3,2,6]),1,2,1,[1,2],[0,1]);
assert.equal(norm.mean,4);assert.equal(norm.variance,4);assert.ok(Math.abs(norm.value-3)<1e-5);
const attention=attentionRow([1,0,0,1],[1,0,0,1],0,0,2,2);
assert.ok(Math.abs(attention.probabilities[0]-.669761549)<1e-8);
assert.equal(attention.probabilities.reduce((a,b)=>a+b),1);
assert.ok(Math.abs(activate(1,'gelu')-.841344746)<1e-7);
assert.equal(convolution({...spec,activation:'linear'},[1,1,1],[3],new Float32Array(9).fill(-1),[0],0,0,0).value,-3);
const residual=JSON.parse(readFileSync(new URL('../web/models/mnist_resnet.json',import.meta.url)));
residual.layers.find(s=>s.op==='add').skip='input';assert.throws(()=>validate(residual),/residual shape/);
console.log('PASS: patch mapping, token weights, LayerNorm, scaled attention, GELU, signed convolution, residual validation');
