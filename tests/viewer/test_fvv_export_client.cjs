const {test}=require('node:test');const assert=require('node:assert/strict');const vm=require('node:vm');const fs=require('fs');
async function fixture(pose={position:[0,0,0],yaw:12,pitch:3,time:1,fov:40,temporal_samples:3}){
const elements={};for(const id of [...fs.readFileSync('viewer/fvv.html','utf8').matchAll(/id="([^"]+)"/g)].map(x=>x[1]))elements[id]={value:'',children:[],replaceChildren(){this.children=[];},append(x){this.children.push(x);}};
Object.assign(elements.orbitSeconds,{value:'12'});elements.orbitFps.value='30';elements.orbitWidth.value='960';elements.orbitRadius.value='1.5';elements.orbitElevation.value='15';
elements.orbitPath.value='showcase';
const calls=[],data={job:null,exports:[]};let timer;
vm.runInNewContext(fs.readFileSync('viewer/fvv-export.js','utf8'),{URLSearchParams,document:{getElementById:id=>elements[id],createElement:()=>({children:[],append(x){this.children.push(x);}})},location:{search:'?scene=lego'},window:{FVV:{exportPose:()=>pose},addEventListener(){},confirm:()=>true},setTimeout:f=>{timer=f;return 1;},clearTimeout(){},fetch:async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>data};}});
await new Promise(r=>setImmediate(r));return {elements,calls,data,poll:()=>timer()};
}
test('orbit export sends current camera and settings without disconnecting playback',async()=>{
const f=await fixture();await f.elements.saveOrbit.onclick();const call=f.calls.find(x=>x.options.method==='POST');const data=JSON.parse(call.options.body);assert.equal(data.path,'showcase');assert.equal(f.elements.orbitRadius.disabled,true);assert.equal(data.yaw,12);assert.equal(data.radius,1.5);assert.equal(data.width,960);assert.equal(data.temporal_samples,3);assert.match(call.url,/orbit-export\?scene=lego$/);
});
test('orbit export requires a loaded model and displays completion downloads',async()=>{
const f=await fixture(null);await f.elements.saveOrbit.onclick();assert.match(f.elements.orbitStatus.textContent,/先加载/);assert.equal(f.calls.filter(x=>x.options.method==='POST').length,0);
f.data.job={state:'complete',progress:100};f.data.exports=[{id:'a'.repeat(32),created_at:'2026-10-31T10:00:00Z'}];await f.poll();assert.equal(f.elements.orbitProgress.value,100);const links=f.elements.orbitDownloads.children[0].children.filter(x=>x.href);assert.equal(links.length,2);assert.match(links[1].href,/parameters\?scene=lego$/);
});
