/* FVV state-machine tests with a fake DOM; no GPU or browser claims. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
async function fixture(ready=true, maxInFlight=1, decode) {
  const elements={}; let tick, socket, now=0, closed=0, drawn=0;
  const docEvents={};
  for(const id of [...fs.readFileSync('viewer/fvv.html','utf8').matchAll(/id="([^"]+)"/g)].map(x=>x[1])) {
    elements[id]={value:'',events:{},clientWidth:960,clientHeight:640,textContent:'',
      addEventListener(k,fn){this.events[k]=fn;},focus(){},setPointerCapture(){},
      getContext(){return {drawImage(){drawn++;}};}};
  }
  elements.speed.value='1';elements.resolution.value='960';elements.fov.value='60';
  class Socket {static OPEN=1;constructor(){socket=this;this.readyState=1;this.sent=[];}send(x){this.sent.push(JSON.parse(x));}close(){this.readyState=3;}}
  const document={hidden:false,getElementById:id=>elements[id],addEventListener:(k,f)=>docEvents[k]=f};
  const context={document,window:{addEventListener(){}},location:{search:'?scene=lego',protocol:'http:',host:'localhost'},
    performance:{now:()=>now},requestAnimationFrame:f=>tick=f,WebSocket:Socket,
    fetch:async()=>({json:async()=>({ready,max_in_flight:maxInFlight,message:'未配置',position:[0,0,-3],yaw:0,pitch:0,time_duration:[0,10]})}),
    createImageBitmap:decode || (async()=>({width:960,height:640,close(){closed++;}})),setTimeout};
  vm.runInNewContext(fs.readFileSync('viewer/fvv.js','utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  return {elements,document,docEvents,get socket(){return socket;},step(t){now=t;tick(t);},get drawn(){return drawn;},get closed(){return closed;}};
}
test('FVV keeps one frame in flight and releases decoded bitmaps',async()=>{
  const f=await fixture();f.step(16);f.step(32);assert.equal(f.socket.sent.length,1);
  await f.socket.onmessage({data:{}});assert.equal(f.drawn,1);assert.equal(f.closed,1);
  f.step(48);assert.equal(f.socket.sent.length,2);
  assert.equal(f.socket.sent[0].width,960);
});
test('FVV camera changes position, rotates, and scrubs time while paused',async()=>{
  const f=await fixture();f.elements.fvvCanvas.events.keydown({key:'w',preventDefault(){}});
  f.elements.fvvCanvas.events.pointerdown({pointerId:1,clientX:0,clientY:0});
  f.elements.fvvCanvas.events.pointermove({clientX:100,clientY:25});
  f.elements.time.value='3';f.elements.time.oninput();f.step(50);
  assert.equal(f.socket.sent[0].time,3);assert.equal(f.socket.sent[0].yaw,20);
  assert.notEqual(f.socket.sent[0].position[2],-3);
});
test('FVV hidden pages disconnect and reject late decoded frames',async()=>{
  const f=await fixture();f.step(16);const receive=f.socket.onmessage;
  f.document.hidden=true;f.docEvents.visibilitychange();assert.equal(f.socket.readyState,3);
  await receive({data:{}});assert.equal(f.drawn,0);
});
test('FVV unconfigured service shows an explicit message without a socket',async()=>{
  const f=await fixture(false);assert.equal(f.socket,undefined);assert.match(f.elements.message.textContent,/未配置/);
});

test('FVV overlaps three requests but never queues a fourth before decode',async()=>{
  const f=await fixture(true,3);for(const t of [20,40,60,80,100])f.step(t);
  assert.equal(f.socket.sent.length,3);
  await f.socket.onmessage({data:{}});f.step(120);
  assert.equal(f.socket.sent.length,4);
});
test('FVV serializes decoding and holds credits until bitmap release',async()=>{
  const resolvers=[];let active=0,peak=0;
  const f=await fixture(true,3,()=>{active++;peak=Math.max(peak,active);return new Promise(r=>resolvers.push(()=>{active--;r({width:960,height:640,close(){}});}));});
  for(const t of [20,40,60])f.step(t);
  const first=f.socket.onmessage({data:{}}), second=f.socket.onmessage({data:{}});
  await new Promise(r=>setImmediate(r));f.step(80);assert.equal(f.socket.sent.length,3);assert.equal(resolvers.length,1);
  resolvers.shift()();await first;await new Promise(r=>setImmediate(r));assert.equal(resolvers.length,1);
  resolvers.shift()();await second;assert.equal(peak,1);assert.equal(f.drawn,2);
});
test('FVV clamps the advertised queue and limits sends to sixty per second',async()=>{
  const f=await fixture(true,999);for(let t=1;t<=100;t++)f.step(t);
  assert.equal(f.socket.sent.length,3);
});
test('FVV releases a bitmap decoded after the connection closes without displaying it',async()=>{
  let resolve, released=0;
  const f=await fixture(true,3,()=>new Promise(r=>resolve=r));
  f.step(20);const decoding=f.socket.onmessage({data:{}});
  await new Promise(r=>setImmediate(r));f.socket.onclose();
  resolve({width:960,height:640,close(){released++;}});await decoding;
  assert.equal(f.drawn,0);assert.equal(released,1);
});
