const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
async function fixture(){
 const ids=[...fs.readFileSync('viewer/fvv.html','utf8').matchAll(/id="([^"]+)"/g)].map(x=>x[1]),elements={};
 for(const id of ids)elements[id]={value:'',dataset:{},files:[],children:[],hidden:false,disabled:false,
  replaceChildren(){this.children=[];this.value='';},append(o){this.children.push(o);if(!this.value)this.value=o.value;},removeAttribute(){this.value=undefined;}};
 let next,disconnected=0,connected=0;const calls=[];
 const response={models:[{id:'lego-18000-v1',bytes:118471316,active:true}],profiles:[{id:'quality',label:'Quality',iterations:18000}],job:null,loading:false,load_error:''};
 const context={document:{getElementById:id=>elements[id],createElement:()=>({})},location:{search:'?scene=lego'},
  window:{FVV:{disconnect(){disconnected++;},connect(){connected++;}},addEventListener(){}},
  setTimeout:f=>{next=f;return 1;},clearTimeout(){},XMLHttpRequest:class{},
  fetch:async(url,opt)=>{calls.push({url,opt});return {ok:true,json:async()=>opt.method?{loading:true}:response};}};
 vm.runInNewContext(fs.readFileSync('viewer/fvv-manager.js','utf8'),context);
 await new Promise(r=>setImmediate(r));
 return {elements,response,calls,get disconnected(){return disconnected;},get connected(){return connected;},async poll(){await next();}};
}
test('model loading disconnects browsing, keeps the scene, and reconnects only when ready',async()=>{
 const f=await fixture();assert.equal(f.elements.loadModel.disabled,false);
 await f.elements.loadModel.onclick();assert.equal(f.disconnected,1);assert.equal(f.connected,0);
 const call=f.calls.find(x=>x.opt.method);assert.match(call.url,/models\/load\?scene=lego$/);assert.equal(JSON.parse(call.opt.body).id,'lego-18000-v1');
 f.response.loading=true;await f.poll();assert.equal(f.connected,0);
 f.response.loading=false;await f.poll();assert.equal(f.connected,1);
});
test('training progress follows server iterations and does not switch the browsing model',async()=>{
 const f=await fixture();await f.elements.startTraining.onclick();assert.equal(f.disconnected,0);
 f.response.job={state:'running',stage:'训练',iteration:3000,iterations:6000,progress:50};await f.poll();
 assert.equal(f.elements.modelProgress.value,50);assert.match(f.elements.modelTask.textContent,/3000\/6000/);assert.equal(f.elements.startTraining.disabled,true);
 f.response.job={state:'complete',id:'trained',iterations:6000,progress:100};await f.poll();
 assert.equal(f.elements.loadTrained.hidden,false);assert.equal(f.connected,0);assert.match(f.elements.modelTask.textContent,/6000/);
});
test('oversized uploads are rejected before opening a connection',async()=>{
 const f=await fixture();f.elements.modelFile.files=[{size:513*1048576}];f.elements.importModel.onclick();assert.match(f.elements.modelTask.textContent,/512 MiB/);
});
