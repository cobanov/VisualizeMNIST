import * as ort from 'onnxruntime-web';
import {NetworkScene} from './scene.js';
import {sizeOf, indexOf, validate, convolution, denseTerms, intensity} from './math.mjs';

ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads=1;
const $=id=>document.getElementById(id);
const motion=matchMedia('(prefers-reduced-motion: reduce)');
let touring=false, tourTime=0;
let playing=!motion.matches, phase=0, position=0, selected=1, channel=0, model=null, revision=0, generation=0;
let dirty=true, empty=false, drawing=false, lastPoint=null, operation=null, lastTime=0;
const scene=new NetworkScene($('view'),(li,index)=>{
  selectLayer(li);
  const s=model.manifest.layers[li];channel=Math.floor(index/(s.shape[1]*s.shape[2]));
  $('channel').value=String(channel);position=index%(s.shape[1]*s.shape[2]);phase=0;setPlaying(false);
  scene.channel=channel;scene.rebuild();updateOperation();
});
const pad=$('pad'),ctx=pad.getContext('2d'),small=document.createElement('canvas'),smallCtx=small.getContext('2d',{willReadFrequently:true});
const detail=$('detail'),g=detail.getContext('2d');
const explanations={
  input:'A grayscale image becomes the network’s input. Each cell is one pixel, from 0 to 1.',
  conv:'A 3 × 3 window moves across the input. All input channels contribute to each output cell, followed by ReLU.',
  flatten:'The same values, rearranged. Channel, row, column order is preserved as the tensor becomes a vector.',
  dense:'Every input connects to this neuron. Inspect the largest weighted contributions or the strongest learned weights.',
  softmax:'Logits become probabilities through a shared normalization. The ten probabilities sum to one.'
};
for(let i=0;i<10;i++){
  const bar=document.createElement('div');bar.className='bar';
  bar.innerHTML=`<span>${i}</span><div class="bar-track"><div class="bar-fill"></div></div><span>0%</span>`;$('bars').append(bar);
}
const bars=[...$('bars').children];
function changed(){revision++;dirty=true;}
function clear(){ctx.fillStyle='#000';ctx.fillRect(0,0,280,280);empty=true;changed();$('input-note').textContent='Draw a digit or choose an example.';}
function sample(which=7){
  clear();ctx.strokeStyle='#fff';ctx.lineWidth=18;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();
  if(which===7){ctx.moveTo(68,64);ctx.lineTo(216,64);ctx.lineTo(118,220);}else{ctx.moveTo(67,88);ctx.bezierCurveTo(96,25,221,35,202,102);ctx.bezierCurveTo(196,128,105,166,71,217);ctx.lineTo(210,217);}
  ctx.stroke();empty=false;changed();$('input-note').textContent=`Hand-drawn example ${which}. Try changing a stroke.`;
}
function point(e){const r=pad.getBoundingClientRect();return{x:(e.clientX-r.left)*280/r.width,y:(e.clientY-r.top)*280/r.height};}
pad.addEventListener('pointerdown',e=>{drawing=true;lastPoint=point(e);pad.setPointerCapture(e.pointerId);ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(lastPoint.x,lastPoint.y,9,0,Math.PI*2);ctx.fill();empty=false;changed();$('input-note').textContent='Your input. Predictions update as you draw.';});
pad.addEventListener('pointermove',e=>{if(!drawing)return;const p=point(e);ctx.strokeStyle='#fff';ctx.lineWidth=18;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(lastPoint.x,lastPoint.y);ctx.lineTo(p.x,p.y);ctx.stroke();lastPoint=p;changed();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])pad.addEventListener(event,()=>drawing=false);
$('clear').onclick=clear;$('sample').onclick=()=>sample(7);$('sample2').onclick=()=>sample(2);
function setPlaying(value){playing=value;if(!value)setTour(false);$('play').textContent=value?'Pause':'Play';$('play').setAttribute('aria-pressed',String(value));}
setPlaying(playing);motion.addEventListener('change',()=>{if(motion.matches)setPlaying(false);});
function setTour(value){touring=value;tourTime=0;$('tour').setAttribute('aria-pressed',String(value));$('tour').textContent=value?'Stop tour':'Follow signal';}
$('tour').onclick=()=>{setTour(!touring);if(touring){if(scene.focused){scene.focus(false);refreshFocus();}setPlaying(true);selectLayer(1);}};
$('play').onclick=()=>setPlaying(!playing);
$('step').onclick=()=>{setPlaying(false);position=(position+1)%countPositions();phase=0;updateOperation();};
$('progress').oninput=()=>{setPlaying(false);position=+$('progress').value;phase=0;updateOperation();};
$('channel').onchange=()=>{channel=+$('channel').value;scene.channel=channel;scene.rebuild();updateOperation();};
$('edge-mode').onchange=()=>updateOperation();
$('exposure').oninput=()=>{scene.exposure=+$('exposure').value;scene.layers.forEach(l=>l.dirty=true);drawDetail();};
$('focus').onclick=()=>{setTour(false);scene.focus(!scene.focused);refreshFocus();updateOperation();};
$('reset-view').onclick=()=>scene.frame();
function refreshFocus(){
  $('focus').setAttribute('aria-pressed',String(scene.focused));$('focus').textContent=scene.focused?'Overview':'Focus layer';
  $('scene-mode').textContent=scene.focused?'LAYER EXPLORER':'NETWORK OVERVIEW';
  $('scene-caption').textContent=scene.focused?'All channels. Click a cell to inspect its exact value.':'Real activations. Follow the highlighted operation.';
}
function countPositions(){const s=model?.manifest.layers[selected];return s?s.shape[1]*s.shape[2]:1;}
function selectLayer(li){
  if(!model)return;selected=li;position=0;phase=0;channel=0;scene.channel=0;scene.setActive(li);
  // Rebuild restores the overview sample if a previous selection inserted a channel.
  scene.rebuild();
  const spec=model.manifest.layers[li];
  [...$('layers').children].forEach((b,i)=>b.setAttribute('aria-pressed',String(i===li)));
  $('selected-name').textContent=spec.label;$('explanation').textContent=explanations[spec.op];
  $('channel-control').hidden=spec.op!=='conv';$('edge-control').hidden=spec.op!=='dense';
  $('channel').replaceChildren(...Array.from({length:spec.shape[0]},(_,i)=>new Option(`${String(i).padStart(2,'0')} / ${spec.shape[0]} channels`,String(i))));
  $('play').disabled=spec.op==='input';$('step').disabled=false;$('progress').disabled=false;
  if(spec.op==='conv'){position=Math.floor(spec.shape[1]/2)*spec.shape[2]+Math.floor(spec.shape[2]/2);}
  refreshFocus();updateOperation();
}
async function fetchOK(url){const r=await fetch(url);if(!r.ok)throw Error(`${r.status}: ${url}`);return r;}
async function load(file){
  setTour(false);const token=++generation,previous=model;model=null;$('model-select').disabled=true;
  $('loading').hidden=false;$('loading').textContent='Loading trained network…';
  if(previous){await previous.pending?.catch(()=>{});await previous.session.release();}
  let session;
  try{
    const manifest=await(await fetchOK(`models/${file}`)).json();validate(manifest);
    let backend;
    try{session=await ort.InferenceSession.create(`models/${manifest.file}`,{executionProviders:['webgpu']});backend='WEBGPU INFERENCE';}
    catch{session=await ort.InferenceSession.create(`models/${manifest.file}`,{executionProviders:['wasm']});backend='WASM INFERENCE';}
    const buffer=await(await fetchOK(`models/${manifest.weights.file}`)).arrayBuffer(),weights={};
    for(const [name,t]of Object.entries(manifest.weights.tensors)){
      const count=sizeOf(t.shape);if(t.offset<0||(t.offset+count)*4>buffer.byteLength)throw Error('Invalid weight buffer');
      weights[name]=new Float32Array(buffer,t.offset*4,count);
    }
    if(token!==generation){await session.release();return;}
    model={manifest,session,weights,values:null,pending:null};scene.load(manifest);scene.active=1;scene.channel=0;
    $('layers').replaceChildren(...manifest.layers.map((s,i)=>{
      const button=document.createElement('button');button.setAttribute('aria-label',`Inspect ${s.label}`);
      const number=document.createElement('span');number.textContent=String(i).padStart(2,'0');
      const title=document.createElement('strong');title.textContent=s.label;
      const shape=document.createElement('small');shape.textContent=s.shape[0]>1?s.shape.join(' × '):`${sizeOf(s.shape)} values`;
      button.append(number,title,shape);button.onclick=()=>{setTour(false);selectLayer(i);};return button;
    }));
    $('input-size').textContent=manifest.input.image.join(' × ');$('backend').textContent=backend;
    $('params').textContent=`${manifest.params.toLocaleString('en-US')} PARAMETERS`;
    $('tensor-summary').textContent=`${manifest.layers.length} stages · ${manifest.layers.filter(s=>s.op==='conv').length?'3 × 3 kernels / stride 2 / padding 1':'Dense layers / ReLU / softmax'}`;
    selectLayer(1);changed();$('loading').hidden=true;
  }catch(error){if(session)await session.release();showError(error);}finally{if(token===generation)$('model-select').disabled=false;}
}
function showError(error){console.error(error);$('loading').hidden=false;$('loading').textContent=`Unable to run this model. ${error.message} Choose another architecture or reload.`;}
$('model-select').onchange=()=>load($('model-select').value);
async function infer(current){
  const version=revision;dirty=false;
  const[h,w]=current.manifest.input.image;small.width=w;small.height=h;smallCtx.drawImage(pad,0,0,w,h);
  const px=smallCtx.getImageData(0,0,w,h).data,input=Float32Array.from({length:w*h},(_,i)=>px[i*4]/255);
  const tensor=new ort.Tensor('float32',input,current.manifest.input.shape),start=performance.now();
  let outputs;
  try{
    outputs=await current.session.run({[current.manifest.input.name]:tensor});
    if(model!==current||version!==revision)return;
    current.values={[current.manifest.input.name]:input};for(const[k,t]of Object.entries(outputs))current.values[k]=new Float32Array(t.data);
    scene.values(current.values);$('latency').textContent=`${(performance.now()-start).toFixed(1)} MS INFERENCE`;
    const probs=current.values.probs;let top=0;for(let i=1;i<10;i++)if(probs[i]>probs[top])top=i;
    $('digit').textContent=empty?'·':String(top);$('confidence').textContent=empty?'Blank input':`${(probs[top]*100).toFixed(1)}% probability`;
    bars.forEach((bar,i)=>{bar.classList.toggle('top',!empty&&i===top);bar.querySelector('.bar-fill').style.width=empty?'0%':`${probs[i]*100}%`;bar.lastChild.textContent=empty?'·':`${Math.round(probs[i]*100)}%`;});
    updateOperation();
  }catch(error){if(model===current)showError(error);}finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(t=>t.dispose());}
}
function updateOperation(){
  operation=null;if(!model)return;
  const spec=model.manifest.layers[selected];position=Math.max(0,Math.min(position,countPositions()-1));
  const index=channel*countPositions()+position,y=Math.floor(position/spec.shape[2]),x=position%spec.shape[2];
  $('progress').max=String(countPositions()-1);$('progress').value=String(position);
  $('position').textContent=`${position+1} / ${countPositions()}`;
  $('operation').textContent={conv:'Convolution scan',dense:'Weighted sum',flatten:'Tensor → vector',softmax:'Probability',input:'Input pixels'}[spec.op];
  if(!model.values)return;
  const values=model.values,raw=values[spec.tensor][index];
  operation={index,terms:[],raw};
  if(spec.op==='conv'){
    const source=model.manifest.layers.find(s=>s.tensor===spec.source);
    Object.assign(operation,convolution(spec,source.shape,values[spec.source],model.weights[spec.weight],model.weights[spec.bias],channel,y,x));
    $('equation').textContent=`ReLU(Σ x·w + ${operation.bias.toFixed(3)}) = ${operation.value.toFixed(4)}`;
    $('selection').textContent=`Channel ${channel} · cell [${y}, ${x}] · ${source.shape[0]} input channels · error vs model ${Math.abs(raw-operation.value).toExponential(1)}`;
  }else if(spec.op==='dense'){
    operation.showEdges=$('edge-mode').value!=='off';
    operation.terms=denseTerms(values[spec.source],model.weights[spec.weight],sizeOf(spec.shape),index,$('edge-mode').value);
    const bias=model.weights[spec.bias][index];
    $('equation').textContent=`${spec.activation==='relu'?'ReLU(Σ x·w + b)':'Σ x·w + b'} = ${raw.toFixed(4)}  ·  b = ${bias.toFixed(3)}`;
    $('selection').textContent=`Neuron ${index} · ${operation.showEdges?'12 largest '+($('edge-mode').value==='weights'?'|weights|':'|contributions|'):'connections hidden'} / ${values[spec.source].length} inputs`;
  }else if(spec.op==='flatten'){
    const source=model.manifest.layers.find(s=>s.tensor===spec.source),[c,h,w]=source.shape;
    $('equation').textContent=`[${Math.floor(index/(h*w))}, ${Math.floor(index/w)%h}, ${index%w}] → vector[${index}] = ${raw.toFixed(4)}`;
    $('selection').textContent=`${c} channels × ${h} rows × ${w} columns. No weights, no values lost.`;
  }else if(spec.op==='softmax'){
    $('equation').textContent=`p(${index}) = exp(z${index}) / Σ exp(z) = ${(raw*100).toFixed(2)}%`;
    $('selection').textContent=`All 10 probabilities sum to ${values.probs.reduce((a,b)=>a+b,0).toFixed(6)}. Brightness uses absolute probability.`;
  }else{
    $('equation').textContent=`pixel[${y}, ${x}] = ${raw.toFixed(4)}`;$('selection').textContent='Black = 0. White = 1. Canvas resampled to the model’s native resolution.';
  }
  scene.operationLinks(operation);drawDetail();
}
function drawDetail(){
  g.fillStyle='#050505';g.fillRect(0,0,504,188);g.font='16px monospace';g.textBaseline='middle';
  if(!operation||!model)return;
  const spec=model.manifest.layers[selected];
  if(spec.op==='conv'){
    const scores=new Map();for(const t of operation.terms)scores.set(t.c,(scores.get(t.c)||0)+Math.abs(t.contribution));
    const c=[...scores].sort((a,b)=>b[1]-a[1])[0][0],terms=operation.terms.filter(t=>t.c===c);
    g.fillStyle='#aaa';g.fillText(`INPUT CH ${c}`,14,18);g.fillText('KERNEL',197,18);g.fillText('OUTPUT',389,18);
    for(let i=0;i<9;i++){
      const t=terms[i],x=i%3,y=Math.floor(i/3);
      for(let k=0;k<2;k++){
        const px=14+k*183+x*50,py=37+y*46,val=k?t.weight:t.value;
        const light=k?Math.min(1,Math.abs(val)*3):intensity(val,'dense',scene.exposure);
        g.fillStyle=`rgb(${Math.round(light*200)} ${Math.round(light*200)} ${Math.round(light*200)})`;g.fillRect(px,py,46,42);
        g.strokeStyle=k&&val<0?'#fff':'#555';g.setLineDash(k&&val<0?[4,3]:[]);g.strokeRect(px,py,46,42);g.setLineDash([]);
        g.fillStyle=light>.5?'#000':'#ddd';g.font='11px monospace';g.fillText(t.padding&&!k?'pad':val.toFixed(2),px+3,py+21);
      }
    }
    g.fillStyle='#ddd';g.font='18px monospace';g.fillText('×',169,103);g.fillText('→',356,103);
    g.fillStyle=`rgb(${Math.round(intensity(operation.raw,'conv',scene.exposure)*230)} ${Math.round(intensity(operation.raw,'conv',scene.exposure)*230)} ${Math.round(intensity(operation.raw,'conv',scene.exposure)*230)})`;g.fillRect(390,58,84,60);
    g.fillStyle='#ddd';g.font='13px monospace';g.fillText(operation.raw.toFixed(4),390,140);
  }else if(spec.op==='dense'){
    const terms=operation.terms.slice(0,6),max=Math.max(...terms.map(t=>Math.abs(t.contribution)),.00001);
    terms.forEach((t,i)=>{const y=20+i*29;g.fillStyle='#aaa';g.font='14px monospace';g.fillText(String(t.i).padStart(3,'0'),14,y);g.fillText(`${t.value.toFixed(2)} × ${t.weight.toFixed(2)}`,66,y);g.fillStyle=t.contribution>=0?'#ddd':'#666';g.fillRect(280,y-5,Math.abs(t.contribution)/max*130,9);g.fillStyle='#ccc';g.fillText(t.contribution.toFixed(2),425,y);});
  }else{
    const l=scene.layers[selected],values=l.raw,cols=spec.op==='softmax'?10:Math.min(32,spec.shape[2]),rows=Math.ceil(values.length/cols),cell=Math.min(460/cols,135/rows);
    const ox=(504-cols*cell)/2,oy=(188-rows*cell)/2;
    values.forEach((val,i)=>{const light=Math.round(intensity(val,spec.op,scene.exposure)*225);g.fillStyle=`rgb(${light} ${light} ${light})`;g.fillRect(ox+i%cols*cell,oy+Math.floor(i/cols)*cell,Math.max(1,cell-1),Math.max(1,cell-1));});
    const i=operation.index;g.strokeStyle='#fff';g.strokeRect(ox+i%cols*cell,oy+Math.floor(i/cols)*cell,cell,cell);
  }
}
function loop(t){
  requestAnimationFrame(loop);const dt=Math.min((t-lastTime)/1000,.05);lastTime=t;
  if(model&&dirty&&!model.pending){const current=model;current.pending=infer(current).finally(()=>current.pending=null);}
  if(model?.values&&playing&&!drawing&&model.manifest.layers[selected].op!=='input'){
    phase+=dt*Number($('speed').value)*1.6;
    if(touring){tourTime+=dt*Number($('speed').value);if(tourTime>7){tourTime=0;selectLayer(selected>=model.manifest.layers.length-1?1:selected+1);}}
    if(phase>=1){phase%=1;position=(position+1)%countPositions();updateOperation();}
  }
  scene.showOperation(operation,phase);scene.render(dt,motion.matches);
}
addEventListener('keydown',e=>{
  if(['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(e.target.tagName))return;
  if(e.code==='Space'){e.preventDefault();setPlaying(!playing);}else if(e.key.toLowerCase()==='c')clear();
});
sample(7);load('mnist_cnn.json');requestAnimationFrame(loop);
