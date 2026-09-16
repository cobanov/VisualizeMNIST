import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {channels, indexOf, intensity, sizeOf, denseTerms, convolution} from './math.mjs?v=4';
import {tokenTerms, patchIndex} from './operations.mjs';
import {easeInOut, perspectiveCenter} from './motion.mjs?v=2';

const white = new THREE.Color('#e0e0e0'), black = new THREE.Color('#000');
const matrix = new THREE.Matrix4(), color = new THREE.Color();
const v = new THREE.Vector3();
export class NetworkScene {
  constructor(element, onPick) {
    this.element = element;
    this.renderer = new THREE.WebGLRenderer({antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor('#000');
    element.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, .1, 500);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .075;
    this.controls.addEventListener('start', () => { this.cameraMove = null; });
    this.controls.minDistance = 3;
    this.controls.maxDistance = 180;
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.overlays = new THREE.Group(); this.scene.add(this.overlays);
    this.layers = [];
    this.connections=new THREE.Group();this.scene.add(this.connections);
    this.focused = false; this.active = 1; this.exposure = 1;
    this.labels = document.getElementById('scene-labels');
    new ResizeObserver(() => {
      const w = element.clientWidth, h = element.clientHeight;
      this.renderer.setSize(w,h,false); this.camera.aspect = w/h; this.camera.updateProjectionMatrix();
      if (this.layers.length) this.frame(false);
    }).observe(element);
    const ray = new THREE.Raycaster();
    let down;
    this.renderer.domElement.addEventListener('pointerdown', e => {down = [e.clientX,e.clientY];});
    this.renderer.domElement.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX-down[0],e.clientY-down[1]) > 5) return;
      const r=element.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2),this.camera);
      const hit = ray.intersectObjects(this.layers.filter(l=>l.group.visible).map(l=>l.mesh))[0];
      if (hit) { const layer=this.layers.findIndex(l=>l.mesh===hit.object); onPick(layer,this.layers[layer].indices[hit.instanceId]); }
    });
    this.marker=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)), new THREE.LineBasicMaterial({color:'#fff',transparent:true,opacity:.65,depthTest:false}));
    this.marker.visible=false;this.overlays.add(this.marker);
    this.links=new THREE.Group();this.overlays.add(this.links);
  }
  disposeGroup(group) {
    const geometries=new Set(), materials=new Set();
    group.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);});
    for(const g of geometries)g.dispose();for(const m of materials){m.map?.dispose();m.dispose();}group.clear();
  }
  load(manifest,weights) {
    this.weights=weights;
    this.disposeGroup(this.root); this.clearLinks(); this.labels.replaceChildren();
    this.manifest=manifest; this.focused=false;this.active=manifest.architecture?4:1;
    let cursor=0;
    this.layers=manifest.layers.map((spec,li)=>{
      const [c,h,w]=spec.shape, cell=manifest.architecture==='vit'?(spec.op==='input'?.22:spec.op==='patches'?.22:spec.op==='attention'?.22:.16):spec.op==='input'?.55:spec.op==='conv'||spec.op==='add'?.55:.42;
      const width=spec.op==='flatten'?13.44:Math.max(w*cell,2.5);
      const x=0, z=cursor; cursor+=spec.op==='input'?10:spec.op==='conv'?7:4.5;
      const group=new THREE.Group();this.root.add(group);
      const label=document.createElement('div');label.className='layer-label';
      const title=document.createElement('b');title.textContent=spec.label;label.append(title,document.createElement('span'));this.labels.append(label);
      return {spec,li,group,label,x,z,width,cell,raw:new Float32Array(sizeOf(spec.shape)),indices:[],positions:new Map()};
    });
    this.totalWidth=cursor;
    // Input faces +Z, matching the drawing pad and the focused layer view.
    for(const l of this.layers)l.z=cursor/2-l.z;
    this.rebuild(true);
  }
  rebuild(reframe = false) {
    this.clearLinks();this.marker.visible=false;
    const vit=this.manifest.architecture==='vit';
    if(vit){const visible=this.layers.filter(l=>l.spec.overview!==false||l.li===this.active);const cols=3,rows=Math.ceil(visible.length/cols);for(const l of this.layers){const i=Math.max(0,visible.indexOf(l));l.x=(i%cols-1)*15;l.y=((rows-1)/2-Math.floor(i/cols))*12;l.z=0;}}

    for(const l of this.layers) {
      this.disposeGroup(l.group);
      l.channelLabels?.forEach(a=>a.element.remove());l.channelLabels=[];
      const [c,h,w]=l.spec.shape;
      l.group.visible=this.focused?l.li===this.active:(!vit||l.spec.overview!==false||l.li===this.active);
      l.label.hidden=!l.group.visible;
      const all=vit||this.focused||this.manifest.layers[this.active]?.op==='flatten'&&this.manifest.layers[this.active].source===l.spec.tensor;
      const shown=all?Array.from({length:c},(_,i)=>i):channels(c);
      if(!this.focused&&l.li===this.active&&c>1&&!shown.includes(this.channel||0)){shown[shown.length-1]=this.channel||0;shown.sort((a,b)=>a-b);}
      l.shown=shown;l.indices=[];l.positions=new Map();
      const cell=l.cell;
      const tileCols=l.spec.op==='attention'?3:Math.ceil(Math.sqrt(c)), tileRows=Math.ceil(c/tileCols);
      const gapX=w*cell+.7,gapY=h*cell+.9;
      for(let ci=0;ci<shown.length;ci++)for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
        const ch=shown[ci], index=indexOf(l.spec.shape,ch,y,x);
        let px,py,pz;
        if((this.focused||vit)&&c>1){px=(this.focused?0:l.x)+(ci%tileCols-(tileCols-1)/2)*gapX+(x-(w-1)/2)*cell;py=((tileRows-1)/2-Math.floor(ci/tileCols))*gapY+((h-1)/2-y)*cell;pz=0;}
        else if(l.spec.op==='flatten') {const cols=32;px=(index%cols-(cols-1)/2)*cell+(this.focused?0:l.x);py=(1.5-Math.floor(index/cols))*cell;pz=this.focused?0:l.z;}
        else {px=(this.focused?0:l.x)+(x-(w-1)/2)*cell;py=((h-1)/2-y)*cell;pz=this.focused?0:l.z+(ci-(shown.length-1)/2)*(shown.length>6?4.25/(shown.length-1):.85);}
        if(vit&&!this.focused)py+=l.y;
        l.indices.push(index);l.positions.set(index,new THREE.Vector3(px,py,pz));
      }
      const count=l.indices.length, geo=new THREE.BoxGeometry(cell*.82,cell*.82,cell*.2);
      const material=new THREE.MeshBasicMaterial();
      if(['conv','add'].includes(l.spec.op))material.onBeforeCompile=shader=>{
        // Empty feature-map cells must not occlude deeper channels. Geometry stays pickable.
        shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>',
          '#include <color_fragment>\nif (dot(diffuseColor.rgb, vec3(1.0)) == 0.0) discard;');
      };
      const mesh=new THREE.InstancedMesh(geo,material,count);mesh.frustumCulled=false;l.mesh=mesh;
      const outline=new THREE.EdgesGeometry(geo).attributes.position.array;
      const edges=new Float32Array(count*outline.length);
      l.indices.forEach((index,i)=>{
        const p=l.positions.get(index);mesh.setMatrixAt(i,matrix.makeTranslation(p.x,p.y,p.z));mesh.setColorAt(i,black);
        for(let k=0;k<outline.length;k+=3) edges.set([outline[k]+p.x,outline[k+1]+p.y,outline[k+2]+p.z],i*outline.length+k);
      });
      const edgeGeo=new THREE.BufferGeometry();edgeGeo.setAttribute('position',new THREE.BufferAttribute(edges,3));
      l.outline=new THREE.LineSegments(edgeGeo,new THREE.LineBasicMaterial({color:'#666',transparent:true,opacity:.45}));
      l.group.add(mesh,l.outline);
      l.dirty=true;
      const bounds=new THREE.Box3().setFromObject(l.group);
      l.bounds=bounds;l.anchor=new THREE.Vector3((bounds.min.x+bounds.max.x)/2,bounds.min.y-.55,(bounds.min.z+bounds.max.z)/2);
      l.label.querySelector('span').textContent=c>1?`${h}×${w} · ${shown.length}/${c} ${l.spec.op==='attention'?'heads':l.spec.op==='patches'?'patches':'channels'}`:`${sizeOf(l.spec.shape)} ${l.spec.op==='input'?'pixels':'values'}`;
    }
    for(const l of this.layers)if(this.focused&&l.group.visible&&l.spec.shape[0]>1){
      for(const c of l.shown){const p=l.positions.get(indexOf(l.spec.shape,c,0,0)).clone();p.y+=.55;
        const element=document.createElement('span');element.className='channel-label';element.textContent=`${l.spec.op==='attention'?'HEAD':l.spec.op==='patches'?'PATCH':'CH'} ${String(c).padStart(2,'0')}`;this.labels.append(element);l.channelLabels.push({element,position:p});
      }
    }
    this.buildConnections();this.setActive(this.active);if(reframe)this.frame();
  }
  buildConnections() {
    this.disposeGroup(this.connections);
    if(this.focused)return;
    for(const l of this.layers){
      const source=this.layers.find(a=>a.spec.tensor===l.spec.source),spec=l.spec;
      if(!source||!l.group.visible||!source.group.visible||!['conv','dense','flatten','add','pool','embed','token_dense','patches'].includes(spec.op))continue;
      const edges=[],positions=[];
      // Fixed spatial samples keep convolution wiring readable. Every segment is a real input term.
      const targets=['conv','add','patches'].includes(spec.op)?l.shown.flatMap(c=>channels(spec.shape[1],2).flatMap(y=>channels(spec.shape[2],2).map(x=>indexOf(spec.shape,c,y,x)))):['embed','token_dense'].includes(spec.op)?l.indices.filter(i=>i%spec.shape[2]===0):l.indices;
      for(const target of targets){
        let terms;
        if(spec.op==='dense')terms=denseTerms(source.raw,this.weights[spec.weight],sizeOf(spec.shape),target,'weights',4);
        else if(spec.op==='conv'){
          const [,h,w]=spec.shape;
          terms=convolution(spec,source.spec.shape,source.raw,this.weights[spec.weight],this.weights[spec.bias],Math.floor(target/(h*w)),Math.floor(target/w)%h,target%w).terms
            .filter(t=>!t.padding).map(t=>({i:indexOf(source.spec.shape,t.c,t.y,t.x),weight:t.weight}));
        }else if(['embed','token_dense'].includes(spec.op)){
          const width=spec.op==='embed'?source.spec.shape[1]*source.spec.shape[2]:source.spec.shape[2];
          terms=tokenTerms(source.raw,this.weights[spec.weight],Math.floor(target/spec.shape[2]),width,spec.shape[2],target%spec.shape[2]).sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight)).slice(0,3);
        }else if(spec.op==='patches'){
          const [,h,w]=spec.shape;terms=[{i:patchIndex(Math.floor(target/(h*w)),Math.floor(target/w)%h,target%w,spec.patch),weight:1}];
        }else if(spec.op==='pool'){
          const count=spec.axis==='tokens'?source.spec.shape[1]:source.spec.shape[1]*source.spec.shape[2];
          terms=channels(count,4).map(i=>({i:spec.axis==='tokens'?i*source.spec.shape[2]+target:target*count+i,weight:1/count}));
        }else terms=[{i:target,weight:1}]; // Flatten and residual addition preserve tensor indices.
        for(const term of terms){
          const from=source.positions.get(term.i),to=l.positions.get(target);
          if(!from||!to)continue;
          positions.push(...from.toArray(),...to.toArray());edges.push(term);
        }
      }
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
      geometry.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(positions.length),3));
      const lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.3,depthWrite:false}));
      lines.userData={source,edges};this.connections.add(lines);
    }
    // Identity shortcuts bypass the learned branch, using matching tensor indices.
    for(const l of this.layers.filter(l=>l.spec.op==='add'&&l.group.visible)){
      const source=this.layers.find(a=>a.spec.tensor===l.spec.skip);if(!source?.group.visible)continue;
      const positions=[],edges=[],vit=this.manifest.architecture==='vit';
      for(const channel of l.shown){const index=indexOf(l.spec.shape,channel,Math.floor(l.spec.shape[1]/2),Math.floor(l.spec.shape[2]/2));
        const a=source.positions.get(index),b=l.positions.get(index);if(!a||!b)continue;
        const top=Math.max(source.bounds.max.y,l.bounds.max.y)+2+channel*.06;
        const curve=vit?new THREE.CubicBezierCurve3(a,a.clone().add(new THREE.Vector3(0,3,1)),b.clone().add(new THREE.Vector3(0,3,1)),b):new THREE.CubicBezierCurve3(a,new THREE.Vector3(a.x,top,a.z),new THREE.Vector3(b.x,top,b.z),b);
        const points=curve.getPoints(24);for(let i=1;i<points.length;i++){positions.push(...points[i-1].toArray(),...points[i].toArray());edges.push({i:index,weight:1});}
      }
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(new Float32Array(positions.length),3));
      const lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.8,depthWrite:false,depthTest:false}));lines.userData={source,edges};this.connections.add(lines);
    }
    this.updateConnections();
  }
  updateConnections() {
    for(const lines of this.connections.children){
      const {source,edges}=lines.userData,colors=lines.geometry.attributes.color;
      edges.forEach((term,i)=>{
        const light=.08+.75*intensity(source.raw[term.i]*term.weight,'dense',this.exposure);
        colors.setXYZ(i*2,light,light,light);colors.setXYZ(i*2+1,light,light,light);
      });
      colors.needsUpdate=true;
    }
  }
  setActive(index) {
    this.active=index;
    for(const l of this.layers){l.label.classList.toggle('active',l.li===index);l.label.hidden=!l.group.visible||(!this.focused&&this.manifest.architecture!=='vit'&&l.li!==index&&!(this.manifest.architecture==='resnet'&&l.spec.op==='add'));l.outline.material.opacity=l.li===index?.9:.55;}
  }
  focus(value) {this.focused=value;this.rebuild(true);}
  frame(animate = true) {
    if(!this.layers.length)return;
    const bounds=new THREE.Box3();
    for(const l of this.layers)if(l.group.visible)bounds.union(l.bounds);
    const center=bounds.getCenter(new THREE.Vector3());
    const direction=this.focused?new THREE.Vector3(0,0,1):this.manifest.architecture==='vit'?new THREE.Vector3(0,.12,1).normalize():new THREE.Vector3(1,.55,1.3).normalize();
    const destination=center.clone().addScaledVector(direction,100);
    const rotation=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(destination,center,this.camera.up));
    const inverse=rotation.clone().invert(),tan=Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2)),corners=[];
    let distance=0;
    for(const l of this.layers.filter(l=>l.group.visible))for(const x of [l.bounds.min.x,l.bounds.max.x])for(const y of [l.bounds.min.y-1,l.bounds.max.y])for(const z of [l.bounds.min.z,l.bounds.max.z]){
      const p=new THREE.Vector3(x,y,z).sub(center).applyQuaternion(inverse);
      corners.push(p);
      distance=Math.max(distance,Math.abs(p.x)/(tan*this.camera.aspect)+p.z,Math.abs(p.y)/tan+p.z);
    }
    distance=distance*(this.focused||this.manifest.architecture==='vit'?1.08:.86)+1;
    const [offsetX,offsetY]=perspectiveCenter(corners,distance);
    center.add(new THREE.Vector3(offsetX,offsetY,0).applyQuaternion(rotation));
    destination.copy(center).addScaledVector(direction,distance);
    if(animate&&this.framed&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
      this.cameraMove={from:this.camera.position.clone(),targetFrom:this.controls.target.clone(),to:destination,target:center,elapsed:0};
    }else{
      this.cameraMove=null;this.camera.position.copy(destination);this.controls.target.copy(center);this.controls.update();
    }
    this.framed=true;
  }
  values(values) {
    for(const l of this.layers) { const raw=values[l.spec.tensor];if(raw.length!==sizeOf(l.spec.shape))throw Error(`Tensor size mismatch: ${l.spec.tensor}`);l.raw=raw;l.dirty=true; }
    this.updateConnections();
  }
  clearLinks() {this.disposeGroup(this.links);}
  line(positions,negative=false) {
    if(!positions.length)return;
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    const material=negative?new THREE.LineDashedMaterial({color:'#aaa',dashSize:.1,gapSize:.08,transparent:true,opacity:.62,depthTest:false}):new THREE.LineBasicMaterial({color:'#ccc',transparent:true,opacity:.52,depthTest:false});
    const line=new THREE.LineSegments(geo,material);if(negative)line.computeLineDistances();this.links.add(line);
  }
  point(layer,index) {return layer.positions.get(index);}
  showOperation(info) {
    this.marker.visible=false;if(!info)return;
    const l=this.layers[this.active],target=l.positions.get(info.index);
    if(target){this.marker.visible=true;this.marker.position.copy(target);this.marker.scale.setScalar(l.cell*1.14);}
  }
  operationLinks(info) {
    this.clearLinks();if(!info||this.focused)return;
    const l=this.layers[this.active],source=this.layers.find(a=>a.spec.tensor===l.spec.source);
    if(!source)return;
    const p=l.positions.get(info.index);if(!p)return;
    const pos=[],neg=[];
    if(['dense','embed','token_dense'].includes(l.spec.op)&&info.showEdges&&source.group.visible)for(const t of info.terms){const a=source.positions.get(t.i);if(a)(t.weight>=0?pos:neg).push(...a.toArray(),...p.toArray());}
    if(l.spec.op==='conv') {
      // Show the receptive-field boundary on each represented input channel.
      const y=Math.floor(info.index/l.spec.shape[2])%l.spec.shape[1], x=info.index%l.spec.shape[2];
      for(const c of source.shown){const center=source.positions.get(indexOf(source.spec.shape,c,0,0));if(!center)continue;
        const sx=center.x+(x*l.spec.stride-l.spec.padding-.5)*source.cell;
        const sy=center.y-(y*l.spec.stride-l.spec.padding-.5)*source.cell;
        const d=l.spec.kernel*source.cell,z=center.z+.06;
        const corners=[[sx,sy,z],[sx+d,sy,z],[sx+d,sy-d,z],[sx,sy-d,z]];
        for(let i=0;i<4;i++)pos.push(...corners[i],...corners[(i+1)%4]);
      }
    }
    if(l.spec.op==='attention'){
      const patches=this.layers.find(a=>a.spec.op==='patches');
      if(patches?.group.visible){
        const center=patch=>patches.positions.get(indexOf(patches.spec.shape,patch,3,3));
        const a=center(info.query);
        const keys=info.attention.probabilities.map((weight,key)=>({weight,key})).sort((a,b)=>b.weight-a.weight).slice(0,4);
        for(const {weight,key}of keys){if(key===info.query)continue;const b=center(key);const curve=new THREE.QuadraticBezierCurve3(a,a.clone().lerp(b,.5).add(new THREE.Vector3(0,0,2)),b);const points=curve.getPoints(20),segments=[];for(let i=1;i<points.length;i++)segments.push(...points[i-1].toArray(),...points[i].toArray());this.line(segments);this.links.children.at(-1).material.opacity=.25+.75*Math.sqrt(weight);}
      }
    }
    this.line(pos);this.line(neg,true);
  }
  render(dt,reduced) {
    if(this.cameraMove){
      const move=this.cameraMove;move.elapsed+=dt;
      const t=reduced?1:easeInOut(move.elapsed/.9);
      this.camera.position.lerpVectors(move.from,move.to,t);this.controls.target.lerpVectors(move.targetFrom,move.target,t);
      // Clear residual orbit damping before applying the authored camera move.
      this.controls.enableDamping=false;this.controls.update();this.controls.enableDamping=true;
      if(t===1)this.cameraMove=null;
    }
    for(const l of this.layers){
      if(l.dirty){
        for(let i=0;i<l.indices.length;i++){
          const raw=l.raw[l.indices[i]],strength=raw<0&&!this.manifest.architecture?.015:intensity(raw,l.spec.op,this.exposure);
          color.copy(black).lerp(white,strength);l.mesh.setColorAt(i,color);
        }
        l.mesh.instanceColor.needsUpdate=true;l.dirty=false;
      }
      for(const a of l.channelLabels){v.copy(a.position).project(this.camera);a.element.style.left=`${(v.x*.5+.5)*this.element.clientWidth}px`;a.element.style.top=`${(-v.y*.5+.5)*this.element.clientHeight}px`;}
      if(!l.label.hidden){v.copy(l.anchor).project(this.camera);l.label.style.left=`${(v.x*.5+.5)*this.element.clientWidth}px`;l.label.style.top=`${(-v.y*.5+.5)*this.element.clientHeight}px`;l.label.style.visibility=Math.abs(v.z)>1?'hidden':'visible';}
    }
    this.controls.update();this.renderer.render(this.scene,this.camera);
  }
}
