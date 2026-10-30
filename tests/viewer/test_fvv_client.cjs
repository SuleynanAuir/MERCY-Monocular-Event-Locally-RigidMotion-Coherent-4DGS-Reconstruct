/* FVV state-machine tests with a fake DOM; no GPU or browser claims. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
async function fixture(ready=true, maxInFlight=1, decode, duration=[0,10], protocol) {
  const elements={}; let tick, timer, socket, now=0, closed=0, drawn=0;
  const docEvents={};
  for(const id of [...fs.readFileSync('viewer/fvv.html','utf8').matchAll(/id="([^"]+)"/g)].map(x=>x[1])) {
    elements[id]={value:'',events:{},clientWidth:960,clientHeight:640,textContent:'',
      addEventListener(k,fn){this.events[k]=fn;},focus(){},setPointerCapture(){},
      getContext(){return {drawImage(){drawn++;}};}};
  }
  elements.speed.value='1';elements.resolution.value='960';elements.fov.value='60';elements.frameRate.value='60';
  class Socket {static OPEN=1;constructor(){socket=this;this.readyState=1;this.sent=[];}send(x){this.sent.push(JSON.parse(x));}close(){this.readyState=3;}}
  const document={hidden:false,getElementById:id=>elements[id],addEventListener:(k,f)=>docEvents[k]=f};
  const context={document,window:{addEventListener(){}},location:{search:'?scene=lego',protocol:'http:',host:'localhost'},
    performance:{now:()=>now},requestAnimationFrame:f=>tick=f,WebSocket:Socket,
    fetch:async()=>({json:async()=>({ready,frame_protocol:protocol,max_in_flight:maxInFlight,message:'未配置',position:[0,0,-3],yaw:0,pitch:0,time_duration:duration})}),
    createImageBitmap:decode || (async()=>({width:960,height:640,close(){closed++;}})),setTimeout:f=>{timer=f;return 1;},clearTimeout(){}};
  vm.runInNewContext(fs.readFileSync('viewer/fvv-loop-cache.js','utf8'),context);
  vm.runInNewContext(fs.readFileSync('viewer/fvv.js','utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  return {elements,document,docEvents,get socket(){return socket;},step(t){now=t;tick(t);},timer(t){now=t;timer?.();},get drawn(){return drawn;},get closed(){return closed;}};
}
test('FVV keeps one frame in flight and releases decoded bitmaps',async()=>{
  const f=await fixture();f.step(16);f.step(32);assert.equal(f.socket.sent.length,1);
  await f.socket.onmessage({data:{}});f.step(80);assert.equal(f.drawn,1);assert.equal(f.closed,1);
  f.step(90);assert.equal(f.socket.sent.length,2);
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
test('FVV serializes decoding and releases network credits after decode',async()=>{
  const resolvers=[];let active=0,peak=0;
  const f=await fixture(true,3,()=>{active++;peak=Math.max(peak,active);return new Promise(r=>resolvers.push(()=>{active--;r({width:960,height:640,close(){}});}));});
  for(const t of [20,40,60])f.step(t);
  const first=f.socket.onmessage({data:{}}), second=f.socket.onmessage({data:{}});
  await new Promise(r=>setImmediate(r));f.step(80);assert.equal(f.socket.sent.length,3);assert.equal(resolvers.length,1);
  resolvers.shift()();await first;await new Promise(r=>setImmediate(r));assert.equal(resolvers.length,1);
  resolvers.shift()();await second;f.step(100);f.step(120);assert.equal(peak,1);assert.equal(f.drawn,2);assert.equal(f.closed,0);
});
test('FVV clamps the advertised queue and limits sends to sixty per second',async()=>{
  const f=await fixture(true,999);for(let t=1;t<=100;t++)f.step(t);
  assert.equal(f.socket.sent.length,6);
});
test('FVV releases a bitmap decoded after the connection closes without displaying it',async()=>{
  let resolve, released=0;
  const f=await fixture(true,3,()=>new Promise(r=>resolve=r));
  f.step(20);const decoding=f.socket.onmessage({data:{}});
  await new Promise(r=>setImmediate(r));f.socket.onclose();
  resolve({width:960,height:640,close(){released++;}});await decoding;
  assert.equal(f.drawn,0);assert.equal(released,1);
});
test('FVV bounds decoded images and presents only on display ticks',async()=>{
  const f=await fixture(true,8);for(let t=20;t<=160;t+=20)f.step(t);
  for(let i=0;i<3;i++)await f.socket.onmessage({data:{}});
  assert.equal(f.drawn,0);assert.equal(f.closed,0);
  f.step(220);assert.equal(f.drawn,1);assert.equal(f.closed,1);
  f.document.hidden=true;f.docEvents.visibilitychange();assert.equal(f.closed,3);
});
test('FVV accepts high frame targets while bounding requests to eight',async()=>{
  const f=await fixture(true,999);f.elements.frameRate.value='120';
  for(let t=1;t<=300;t++)f.step(t);
  assert.equal(f.socket.sent.length,8);
});

test('FVV timer requests advance animation even between display ticks',async()=>{
  const f=await fixture(true,8);f.elements.frameRate.value='120';
  f.step(20);f.timer(38);f.timer(56);
  assert.equal(f.socket.sent.length,3);
  assert.ok(f.socket.sent[2].time>f.socket.sent[1].time);
  assert.ok(f.socket.sent[1].time>f.socket.sent[0].time);
});
test('FVV scrubbing rejects frames from the previous timeline',async()=>{
  const f=await fixture(true,3);f.step(20);
  f.elements.time.value='4';f.elements.time.oninput();
  await f.socket.onmessage({data:{}});f.step(100);
  assert.equal(f.drawn,0);assert.equal(f.closed,1);
  assert.equal(f.socket.sent.at(-1).time,4);
});
test('FVV actual display FPS falls to zero during a stall',async()=>{
  const f=await fixture();f.step(20);await f.socket.onmessage({data:{}});f.step(100);f.step(1200);
  assert.match(f.elements.metrics.textContent,/^0 FPS/);
});
test('FVV buffers a small arrival burst and presents it over separate display ticks',async()=>{
  const f=await fixture(true,8);f.step(20);f.step(40);f.step(60);
  for(let i=0;i<3;i++)await f.socket.onmessage({data:{}});
  assert.equal(f.closed,0);f.step(72);assert.equal(f.drawn,1);
  f.step(92);assert.equal(f.drawn,2);f.step(112);assert.equal(f.drawn,3);
});

test('FVV antialiasing only samples animation during playback',async()=>{
  const f=await fixture(true,8);f.step(20);
  assert.equal(f.socket.sent[0].temporal_samples,3);
  f.elements.play.onclick();f.step(40);
  assert.equal(f.socket.sent.at(-1).temporal_samples,1);
  f.elements.temporalSamples.value='1';f.elements.play.onclick();f.step(60);
  assert.equal(f.socket.sent.at(-1).temporal_samples,1);
});
test('FVV decoded jitter buffer never retains more than twelve images',async()=>{
  const f=await fixture(true,8);f.step(20);
  for(let i=0;i<16;i++) {
    if(i)f.timer(20+i*20);
    await f.socket.onmessage({data:{}});
  }
  assert.equal(f.drawn,0);assert.equal(f.closed,4);
  f.document.hidden=true;f.docEvents.visibilitychange();assert.equal(f.closed,16);
});

test('FVV complete loop survives a closed socket and invalidates on camera rotation',async()=>{
  const f=await fixture(true,8,undefined,[0,.1]);f.elements.smoothLoop.checked=true;
  f.step(20);await f.socket.onmessage({data:{size:10}});let received=1;f.step(40);f.step(1300);
  let now=1320;
  for(let i=0;i<40 && !f.elements.cacheStatus.textContent.includes('已预载');i++) {
    f.timer(now);if(f.socket.sent.length>received){await f.socket.onmessage({data:{size:10}});received++;}
    f.step(now);now+=20;await new Promise(r=>setImmediate(r));
  }
  assert.match(f.elements.cacheStatus.textContent,/已预载/);
  const socket=f.socket, count=socket.sent.length, before=f.drawn;
  socket.onclose({code:1006});
  for(let i=0;i<10;i++){f.step(now);now+=20;await new Promise(r=>setImmediate(r));}
  assert.ok(f.drawn>before);assert.equal(socket.sent.length,count);
  f.elements.fvvCanvas.events.pointerdown({pointerId:1,clientX:0,clientY:0});
  f.elements.fvvCanvas.events.pointermove({clientX:20,clientY:0});
  assert.doesNotMatch(f.elements.cacheStatus.textContent,/已预载/);
});

test('FVV identifies a fresh frame after server-side camera coalescing',async()=>{
  const f=await fixture(true,3,undefined,[0,10],'fvv-jpeg-v1');for(const t of [20,40,60])f.step(t);
  assert.equal(f.socket.sent[2].request_id,3);assert.equal(f.socket.sent[2].coalesce,true);
  const header=new ArrayBuffer(8),v=new DataView(header);v.setUint32(0,0x46565631);v.setUint32(4,3);
  await f.socket.onmessage({data:new Blob([header,'jpeg'])});f.step(80);
  assert.equal(f.drawn,1);assert.ok(f.socket.sent.at(-1).request_id>3);
});

test('FVV does not reconnect a slow pipeline while responses still arrive',async()=>{
  const f=await fixture(true,8);f.socket.onopen();
  for(let i=1;i<=8;i++)f.step(i*20);
  f.step(4900);await f.socket.onmessage({data:{}});f.step(5100);
  assert.equal(f.socket.readyState,1);assert.doesNotMatch(f.elements.message.textContent,/超时/);
  f.step(13001);assert.match(f.elements.message.textContent,/超时/);
});

test('FVV reduces bandwidth before a stalled receive window can fill',async()=>{
  const f=await fixture(true,8);f.elements.adaptivePreview.checked=true;f.socket.onopen();
  for(let i=1;i<=8;i++)f.step(i*20);
  for(const t of [700,1300,1900])f.step(t);
  await f.socket.onmessage({data:{}});f.step(2000);
  assert.equal(f.socket.sent.at(-1).width,640);
});
