import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {channels, indexOf, intensity, sizeOf} from './math.mjs';

const white = new THREE.Color('#e0e0e0'), black = new THREE.Color('#000');
const matrix = new THREE.Matrix4(), color = new THREE.Color();
const v = new THREE.Vector3(), scale = new THREE.Vector3(), q = new THREE.Quaternion();
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
    this.controls.minDistance = 3;
    this.controls.maxDistance = 180;
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.overlays = new THREE.Group(); this.scene.add(this.overlays);
    this.layers = [];
    this.focused = false; this.active = 1; this.exposure = 1;
    this.labels = document.getElementById('scene-labels');
    new ResizeObserver(() => {
      const w = element.clientWidth, h = element.clientHeight;
      this.renderer.setSize(w,h,false); this.camera.aspect = w/h; this.camera.updateProjectionMatrix();
      if (this.layers.length) this.frame();
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
    this.ghost = new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial({color:'#eee',transparent:true,opacity:.78,depthTest:false}),512);
    this.ghost.count=0; this.ghost.frustumCulled=false; this.overlays.add(this.ghost);
    this.marker=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)), new THREE.LineBasicMaterial({color:'#fff',depthTest:false}));
    this.marker.visible=false;this.overlays.add(this.marker);
    this.links=new THREE.Group();this.overlays.add(this.links);
  }
  disposeGroup(group) {
    const geometries=new Set(), materials=new Set();
    group.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);});
    for(const g of geometries)g.dispose();for(const m of materials){m.map?.dispose();m.dispose();}group.clear();
  }
  load(manifest) {
    this.disposeGroup(this.root); this.clearLinks(); this.labels.replaceChildren();
    this.manifest=manifest; this.focused=false;
    let cursor=0;
    this.layers=manifest.layers.map((spec,li)=>{
      const [c,h,w]=spec.shape, cell=spec.op==='input'?.55:spec.op==='conv'?.55:.42;
      const width=spec.op==='flatten'?13.44:Math.max(w*cell,2.5);
      const x=0, z=cursor; cursor+=spec.op==='input'?10:spec.op==='conv'?7:4.5;
      const group=new THREE.Group();this.root.add(group);
      const label=document.createElement('div');label.className='layer-label';
      const title=document.createElement('b');title.textContent=spec.label;label.append(title,document.createElement('span'));this.labels.append(label);
      return {spec,li,group,label,x,z,width,cell,raw:new Float32Array(sizeOf(spec.shape)),display:null,indices:[],positions:new Map()};
    });
    this.totalWidth=cursor;
    for(const l of this.layers)l.z-=cursor/2;
    this.rebuild();
  }
  rebuild() {
    this.clearLinks();this.ghost.count=0;this.marker.visible=false;
    for(const l of this.layers) {
      this.disposeGroup(l.group);
      l.channelLabels?.forEach(a=>a.element.remove());l.channelLabels=[];
      const [c,h,w]=l.spec.shape;
      l.group.visible=!this.focused||l.li===this.active;
      l.label.hidden=!l.group.visible;
      const all=this.focused||this.manifest.layers[this.active]?.op==='flatten'&&this.manifest.layers[this.active].source===l.spec.tensor;
      const shown=all?Array.from({length:c},(_,i)=>i):channels(c);
      if(!this.focused&&l.li===this.active&&c>1&&!shown.includes(this.channel||0)){shown[shown.length-1]=this.channel||0;shown.sort((a,b)=>a-b);}
      l.shown=shown;l.indices=[];l.positions=new Map();
      const cell=l.cell;
      const tileCols=Math.ceil(Math.sqrt(c)), tileRows=Math.ceil(c/tileCols);
      const gapX=w*cell+.7,gapY=h*cell+.9;
      for(let ci=0;ci<shown.length;ci++)for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
        const ch=shown[ci], index=indexOf(l.spec.shape,ch,y,x);
        let px,py,pz;
        if(this.focused&&c>1){px=(ci%tileCols-(tileCols-1)/2)*gapX+(x-(w-1)/2)*cell;py=((tileRows-1)/2-Math.floor(ci/tileCols))*gapY+((h-1)/2-y)*cell;pz=0;}
        else if(l.spec.op==='flatten') {const cols=32;px=(index%cols-(cols-1)/2)*cell+(this.focused?0:l.x);py=(1.5-Math.floor(index/cols))*cell;pz=this.focused?0:l.z;}
        else {px=(this.focused?0:l.x)+(x-(w-1)/2)*cell;py=((h-1)/2-y)*cell;pz=this.focused?0:l.z+(ci-(shown.length-1)/2)*(shown.length>6?4.25/(shown.length-1):.85);}
        l.indices.push(index);l.positions.set(index,new THREE.Vector3(px,py,pz));
      }
      const count=l.indices.length, geo=new THREE.BoxGeometry(cell*.82,cell*.82,cell*.2);
      const mesh=new THREE.InstancedMesh(geo,new THREE.MeshBasicMaterial(),count);mesh.frustumCulled=false;l.mesh=mesh;
      const outline=new THREE.EdgesGeometry(geo).attributes.position.array;
      const edges=new Float32Array(count*outline.length);
      l.indices.forEach((index,i)=>{
        const p=l.positions.get(index);mesh.setMatrixAt(i,matrix.makeTranslation(p.x,p.y,p.z));mesh.setColorAt(i,black);
        for(let k=0;k<outline.length;k+=3) edges.set([outline[k]+p.x,outline[k+1]+p.y,outline[k+2]+p.z],i*outline.length+k);
      });
      const edgeGeo=new THREE.BufferGeometry();edgeGeo.setAttribute('position',new THREE.BufferAttribute(edges,3));
      l.outline=new THREE.LineSegments(edgeGeo,new THREE.LineBasicMaterial({color:'#666',transparent:true,opacity:.45}));
      l.group.add(mesh,l.outline);
      l.display=new Float32Array(count).fill(-1);l.dirty=true;
      const bounds=new THREE.Box3().setFromObject(l.group);
      l.bounds=bounds;l.anchor=new THREE.Vector3((bounds.min.x+bounds.max.x)/2,bounds.min.y-.55,(bounds.min.z+bounds.max.z)/2);
      l.label.querySelector('span').textContent=c>1?`${h}×${w} · ${shown.length}/${c} channels`:`${sizeOf(l.spec.shape)} ${l.spec.op==='input'?'pixels':'values'}`;
    }
    for(const l of this.layers)if(this.focused&&l.group.visible&&l.spec.shape[0]>1){
      for(const c of l.shown){const p=l.positions.get(indexOf(l.spec.shape,c,0,0)).clone();p.y+=.55;
        const element=document.createElement('span');element.className='channel-label';element.textContent=`CH ${String(c).padStart(2,'0')}`;this.labels.append(element);l.channelLabels.push({element,position:p});
      }
    }
    this.setActive(this.active);this.frame();
  }
  setActive(index) {
    this.active=index;
    for(const l of this.layers){l.label.classList.toggle('active',l.li===index);l.label.hidden=!l.group.visible||(!this.focused&&l.li!==index);l.outline.material.opacity=l.li===index?.9:.55;}
  }
  focus(value) {this.focused=value;this.rebuild();}
  frame() {
    if(!this.layers.length)return;
    const bounds=new THREE.Box3();
    for(const l of this.layers)if(l.group.visible)bounds.union(l.bounds);
    const center=bounds.getCenter(new THREE.Vector3());
    const direction=this.focused?new THREE.Vector3(0,0,1):new THREE.Vector3(-1,.55,1.3).normalize();
    this.camera.position.copy(center).addScaledVector(direction,100);this.camera.lookAt(center);
    const inverse=this.camera.quaternion.clone().invert(),tan=Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2));
    let distance=0;
    for(const l of this.layers.filter(l=>l.group.visible))for(const x of [l.bounds.min.x,l.bounds.max.x])for(const y of [l.bounds.min.y-1,l.bounds.max.y])for(const z of [l.bounds.min.z,l.bounds.max.z]){
      const p=new THREE.Vector3(x,y,z).sub(center).applyQuaternion(inverse);
      distance=Math.max(distance,Math.abs(p.x)/(tan*this.camera.aspect)+p.z,Math.abs(p.y)/tan+p.z);
    }
    this.camera.position.copy(center).addScaledVector(direction,distance*1.08+1);
    this.controls.target.copy(center);this.controls.update();
  }
  values(values) {
    for(const l of this.layers) { const raw=values[l.spec.tensor];if(raw.length!==sizeOf(l.spec.shape))throw Error(`Tensor size mismatch: ${l.spec.tensor}`);l.raw=raw;l.dirty=true; }
  }
  clearLinks() {this.disposeGroup(this.links);}
  line(positions,negative=false) {
    if(!positions.length)return;
    const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
    const material=negative?new THREE.LineDashedMaterial({color:'#aaa',dashSize:.1,gapSize:.08,transparent:true,opacity:.62,depthTest:false}):new THREE.LineBasicMaterial({color:'#ccc',transparent:true,opacity:.52,depthTest:false});
    const line=new THREE.LineSegments(geo,material);if(negative)line.computeLineDistances();this.links.add(line);
  }
  point(layer,index) {return layer.positions.get(index);}
  showOperation(info,phase) {
    this.ghost.count=0;this.marker.visible=false;
    if(!info)return;
    const l=this.layers[this.active],source=this.layers.find(a=>a.spec.tensor===l.spec.source);
    const target=l.positions.get(info.index);
    if(target){this.marker.visible=true;this.marker.position.copy(target);this.marker.scale.setScalar(l.cell*(1.1+.2*Math.sin(Math.PI*phase)));}
    if(this.focused||!source)return;
    let pairs=[];
    if(l.spec.op==='conv'&&target) {
      for(const term of info.terms){if(term.padding)continue;const p=source.positions.get(indexOf(source.spec.shape,term.c,term.y,term.x));if(p)pairs.push([p,target,source.cell*.9]);}
    } else if(l.spec.op==='flatten') {
      for(const [index,p]of source.positions){const end=l.positions.get(index);if(end)pairs.push([p,end,source.cell*.8]);}
    } else if(l.spec.op==='dense'&&target&&info.showEdges) {
      for(const term of info.terms){const p=source.positions.get(term.i);if(p)pairs.push([p,target,.065]);}
    } else if(l.spec.op==='softmax'&&target) {
      const p=source.positions.get(info.index);if(p)pairs.push([p,target,.09]);
    }
    const t=Math.max(0,Math.min(1,(phase-.25)/.65)),ease=t*t*(3-2*t);
    this.ghost.count=Math.min(512,pairs.length);
    for(let i=0;i<this.ghost.count;i++){
      const[a,b,s]=pairs[i];v.copy(a).lerp(b,ease);scale.set(s,s,s*.4);matrix.compose(v,q,scale);this.ghost.setMatrixAt(i,matrix);
    }
    this.ghost.instanceMatrix.needsUpdate=true;
  }
  operationLinks(info) {
    this.clearLinks();if(!info||this.focused)return;
    const l=this.layers[this.active],source=this.layers.find(a=>a.spec.tensor===l.spec.source);
    if(!source)return;
    const p=l.positions.get(info.index);if(!p)return;
    const pos=[],neg=[];
    if(l.spec.op==='dense'&&info.showEdges)for(const t of info.terms){const a=source.positions.get(t.i);if(a)(t.weight>=0?pos:neg).push(...a.toArray(),...p.toArray());}
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
    this.line(pos);this.line(neg,true);
  }
  render(dt,reduced) {
    for(const l of this.layers){
      let changing=false;
      for(let i=0;i<l.indices.length;i++){
        const raw=l.raw[l.indices[i]], target=intensity(raw,l.spec.op,this.exposure);
        const last=l.display[i], next=reduced||last<0?target:last+(target-last)*(1-Math.exp(-dt*14));
        if(Math.abs(next-last)>.0005||l.dirty){l.display[i]=next;const strength=raw<0?.015:next;color.copy(black).lerp(white,strength);l.mesh.setColorAt(i,color);changing=true;}
      }
      if(changing)l.mesh.instanceColor.needsUpdate=true;l.dirty=false;
      for(const a of l.channelLabels){v.copy(a.position).project(this.camera);a.element.style.left=`${(v.x*.5+.5)*this.element.clientWidth}px`;a.element.style.top=`${(-v.y*.5+.5)*this.element.clientHeight}px`;}
      if(!l.label.hidden){v.copy(l.anchor).project(this.camera);l.label.style.left=`${(v.x*.5+.5)*this.element.clientWidth}px`;l.label.style.top=`${(-v.y*.5+.5)*this.element.clientHeight}px`;l.label.style.visibility=Math.abs(v.z)>1?'hidden':'visible';}
    }
    this.controls.update();this.renderer.render(this.scene,this.camera);
  }
}
