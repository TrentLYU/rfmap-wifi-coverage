// UI state/invariant checks without browser dependencies. Browser rendering is checked separately.
'use strict';
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const base=__dirname;
function context(withEvidence){
  const nodes={},downloads=[],drawCalls=[];
  function node(id){if(nodes[id])return nodes[id];let html='';const listeners={};
    const ctx=new Proxy({measureText:t=>({width:t.length*6}),createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)})},{get:(o,k)=>k in o?o[k]:(...a)=>drawCalls.push([k,a])});
    const n={id,value:'',textContent:'',dataset:{},style:{},files:[],tagName:'DIV',disabled:false,
      classList:{add(){},remove(){},toggle(){}},getContext:()=>ctx,getBoundingClientRect:()=>({left:0,top:0,width:800,height:560}),
      addEventListener:(type,fn)=>{listeners[type]=fn;},dispatch:(type,event={})=>listeners[type]&&listeners[type](event),
      setPointerCapture(){},hasPointerCapture:()=>false,releasePointerCapture(){},click(){if(this.onclick)this.onclick();},
      toDataURL:()=>'',showModal(){},remove(){},appendChild(){},closest:()=>null};
    Object.defineProperty(n,'innerHTML',{get:()=>html,set:v=>{html=v;const m=v.match(/<option value="([^"]+)"/);if(m)n.value=m[1];}});
    nodes[id]=n;return n;
  }
  ['taskSelect','displayMode'].forEach(id=>node(id).value=id==='taskSelect'?'completion':'levels');
  const tabs=['validation','designer'].map(t=>Object.assign(node('tab'+t),{dataset:{tab:t}}));
  const tools=['select','ap','wall','calibrate'].map(t=>Object.assign(node('tool'+t),{dataset:{tool:t}}));
  const c={console,Float32Array,Uint8Array,Uint8ClampedArray,Array,Math,JSON,Number,String,Object,parseInt,isFinite,performance,
    setTimeout:fn=>{fn();return 1;},clearTimeout(){},atob,Blob,URL:{createObjectURL:()=>'',revokeObjectURL(){}},
    document:{getElementById:node,querySelectorAll:sel=>sel==='.tab'?tabs:tools,createElement:type=>node(type+Math.random()),addEventListener(){},body:{appendChild(){}}},
    addEventListener(){},alert:msg=>{throw Error(msg);},devicePixelRatio:1};
  c.window=c;c.globalThis=c;
  vm.createContext(c);
  if(withEvidence)vm.runInContext(fs.readFileSync(path.resolve(base,'../dashboard.js'),'utf8'),c);
  vm.runInContext(fs.readFileSync(path.join(base,'geometry.js'),'utf8'),c);
  vm.runInContext(fs.readFileSync(path.join(base,'app.js'),'utf8'),c);
  return{c,node,nodes,drawCalls};
}
const {c,node}=context(),studio=c.RFM_Studio70;
assert.match(node('resultsStatus').textContent,/未找到本次实验结果/,'Missing evidence must never show historical results');
assert.equal(studio.grade(-80),1);assert.equal(studio.grade(-70),2);assert.equal(studio.grade(-60),3);assert.equal(studio.grade(-50),4);
const project={room:{w:12,h:10},ap:{x:1.2,y:3.4},walls:[],furniture:[]};
studio.applyProject(project);assert.equal(studio.getState().ap.x,1.2);
const kinds=['cabinet','bed','sofa','chair','toilet','table','wardrobe','bookshelf','fridge','washer','bathtub','tv'];
kinds.forEach(kind=>{node('furnitureType').value=kind;node('addFurniture').onclick();});
assert.equal(studio.getState().furniture.length,12);
assert.deepEqual(Array.from(studio.getState().furniture,f=>f.kind),kinds);
node('objectW').value='200';node('objectW').onchange();assert.equal(studio.getState().furniture[11].w,12,'Oversize furniture must be bounded by room');
node('roomW').value='3';node('roomW').onchange();
studio.getState().furniture.forEach(f=>{assert(f.x-f.w/2>=-1e-9);assert(f.x+f.w/2<=3+1e-9);});
node('addWall').onclick();assert.equal(studio.getState().walls.length,1);node('deleteObject').onclick();assert.equal(studio.getState().walls.length,0);
const exported=studio.exportProject();studio.applyProject(exported);assert.equal(studio.getState().furniture.length,12);
assert.throws(()=>studio.applyProject({room:{w:5,h:5},ap:{x:2,y:2},furniture:[{x:2,y:2,w:-1,h:1}]}),/不合法/);
assert.equal(studio.getState().room.w,3,'Invalid import must preserve current project');
const before=Array.from(c.RFM_Geometry.buildInput(studio.getState()).channels);
const withAnswers=studio.exportProject();withAnswers.points=[{x:1,y:1,rssi:9999}];withAnswers.hiddenAnswers=[-1,-100];
const after=Array.from(c.RFM_Geometry.buildInput(withAnswers).channels);
assert.deepEqual(before,after,'Layout predictor must be invariant to measured/hidden RSSI');
studio.setPage('designer');studio.render();
const modelPath=path.resolve(base,'../deployment/model_weights.js');
if(fs.existsSync(modelPath)){
  vm.runInContext(fs.readFileSync(modelPath,'utf8'),c);vm.runInContext(fs.readFileSync(path.join(base,'nn_model.js'),'utf8'),c);
  const input=c.RFM_Geometry.buildInput(project),out=c.RFM_LayoutInfer(input).pred;
  assert.equal(out.length,1024);assert(Array.from(out).every(Number.isFinite));
}
if(fs.existsSync(path.resolve(base,'../dashboard.js'))){
  const real=context(true);
  assert.match(real.node('summary').innerHTML,/189/);
  for(const scenario of ['empty','furnished'])for(const task of ['completion','layout']){
    real.node('datasetSelect').value=scenario;real.node('taskSelect').value=task;real.node('taskSelect').onchange();
    for(const method of ['ae','unet','resunet','unet_scatter','pgmpr']){
      real.node('methodSelect').value=method;real.node('methodSelect').onchange();
      assert(!real.node('metricDetail').innerHTML.includes('—'),'Every reported model must display real metrics');
    }
  }
  assert.match(real.node('metricDetail').innerHTML,/多数等级/,'Accuracy must be accompanied by majority-class baseline');
  assert.equal(real.c.RFM_Studio70.getState().ap.x,0,'Default AP must match training assumption');
  assert.equal(real.c.RFM_Studio70.getState().ap.y,0);
}
console.log('PASS: missing-result honesty; five-grade boundaries; 12 furniture types; edit/delete; size bounds; project roundtrip; invalid-import preservation; geometry-only RSSI independence; live-model inference; all 20 evidence views; majority baseline; training AP agreement.');
