'use strict';
(() => {
  const $ = id => document.getElementById(id), canvas = $('fvvCanvas'), ctx = canvas.getContext('2d');
  const query = location.search;
  $('back').href = '/' + query;
  let ws, metadata, position, yaw = 0, pitch = 0, current = 0, playing = true;
  let inFlight = [], windowSize = 1, nextSend = 0, sendTimer, decodeQueue = Promise.resolve(), frames = [];
  let last = performance.now(), lastDisplay = 0, generation = 0, connecting = false, lastResponseAt=0;
  let timelineRevision = 0, droppedFrames = 0, requestSerial=0;
  let clockAt = performance.now(), displayTimes = [], latencies = [], playoutDelay = 50;
  const loopCache = new FVVLoopCache(blob => createImageBitmap(blob));
  let loopState=null, idleSince=performance.now(), lastLoopIndex=-1, reconnectTimer, reconnectAttempts=0, resumeState=null;
  let displayIntervals=[], lastTick=0, receiveTimes=[], nextPresentation=0, presentationStarted=false, liveWidth=960, lastQualityChange=0;
  function clearFrames() {for(const frame of frames)frame.bitmap.close();frames=[];}
  function invalidateView() {
    idleSince=performance.now();
    if(loopState) {timelineRevision++;loopState=null;loopCache.clear();lastLoopIndex=-1;clearFrames();nextSend=0;}
    if($('cacheStatus'))$('cacheStatus').textContent=$('smoothLoop').checked ? '移动视角时实时渲染；停稳后预载当前视角。' : '实时渲染当前视角；可按需开启本地循环预载。';
    if(!ws && metadata && !connecting) {clearTimeout(reconnectTimer);reconnectTimer=setTimeout(connect,0);}
  }
  function savePose() {
    if(metadata)resumeState={sha:metadata.sha256,position:[...position],yaw,pitch,current,playing,fov:Number($('fov').value)};
  }
  function recover(message) {
    savePose();
    if(loopState?.phase==='playing') {
      generation++;if(ws){ws.onclose=null;ws.close();ws=null;}inFlight=[];clearTimeout(sendTimer);
      $('message').textContent='连接中断，当前视角继续本地播放；移动视角时重新连接。';return;
    }
    disconnect();$('message').textContent=message+'，正在自动重连…';
    reconnectTimer=setTimeout(()=>{if(!document.hidden)connect();},Math.min(10000,500*2**Math.min(4,reconnectAttempts++)));
  }
  function beginLoop() {
    timelineRevision++;clearFrames();
    const hz=displayIntervals.length>10 ? 1000/[...displayIntervals].sort((a,b)=>a-b)[Math.floor(displayIntervals.length/2)] : 60;
    const refresh=[30,50,60,75,90,120].reduce((best,value)=>Math.abs(value-hz)<Math.abs(best-hz) ? value : best,60);
    const rate=Math.min(Number($('frameRate').value)||60,refresh);
    const duration=metadata.time_duration[1]-metadata.time_duration[0];
    const count=Math.ceil(duration*rate), version=loopCache.begin(count);
    loopState={phase:'building',version,next:0,count,rate,started:performance.now(),origin:current,baseIndex:Math.min(count-1,Math.floor((current-metadata.time_duration[0])/duration*count))};
    $('message').textContent='';
  }
  async function acceptLoop(request,blob) {
    if(!loopState||request.cacheVersion!==loopState.version)return;
    const state=loopState;
    if(loopCache.put(request.cacheIndex,blob,state.version)) {
      state.phase='warming';
      const [lo,hi]=metadata.time_duration;
      const index=Math.min(state.count-1,Math.floor((state.origin-lo)/(hi-lo)*state.count));
      await loopCache.prepare(index);
      if(loopState!==state)return;
      clearFrames();state.phase='playing';state.preparation_ms=performance.now()-state.started;
      current=lo+index/state.count*(hi-lo);clockAt=performance.now();displayTimes=[];samples=[];lastDisplay=0;
      $('message').textContent='';$('cacheStatus').textContent=`当前视角已预载 · ${state.rate} FPS 时间采样 · ${(loopCache.bytes/1024/1024).toFixed(1)} MiB 内存；移动即重新渲染。`;
    } else {
      $('cacheStatus').textContent=`预载 ${loopCache.received}/${state.count} 帧（${Math.floor(100*loopCache.received/state.count)}%）· 完成后连续播放`;
    }
  }
  function present(bitmap,time,now,sentAt,owned=true) {
    $('time').value=time;$('timestamp').textContent=time.toFixed(3)+' s';displayTimes.push(now);
    if(canvas.width!==bitmap.width)canvas.width=bitmap.width;if(canvas.height!==bitmap.height)canvas.height=bitmap.height;
    ctx.drawImage(bitmap,0,0);if(owned)bitmap.close();
    if(lastDisplay){samples.push({interval_ms:now-lastDisplay,roundtrip_ms:now-sentAt,animation_time:time});if(samples.length>300)samples.shift();}
    lastDisplay=now;
  }
  let samples = [], keys = new Set(), drag = null;
  function advanceClock(now) {
    if (metadata && playing && (!loopState || loopState.phase==='playing')) {
      const [lo, hi] = metadata.time_duration;
      current = lo + ((current-lo + Math.max(0,now-clockAt)/1000) % (hi-lo));
    }
    clockAt = now;
  }
  function reset() {
    if (!metadata) return;
    position = [...metadata.position]; yaw = metadata.yaw; pitch = metadata.pitch;
    keys.clear();invalidateView();
  }
  function disconnect() {
    clearTimeout(reconnectTimer);receiveTimes=[];nextPresentation=0;presentationStarted=false;loopState=null;loopCache.clear();lastLoopIndex=-1;
    generation++; droppedFrames=0; metadata = null; displayTimes=[]; latencies=[]; playoutDelay=50; inFlight = []; decodeQueue = Promise.resolve(); keys.clear();
    clearTimeout(sendTimer); sendTimer=null; for (const frame of frames) frame.bitmap.close(); frames=[];
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
  }
  async function connect() {
    if (connecting) return;
    connecting = true; savePose();disconnect();
    const token = generation;
    $('message').textContent = '正在连接 Hybrid 模型…';
    try {
      const response = await fetch('/fvv/status' + query, {cache:'no-store', ...(typeof AbortSignal!=='undefined' ? {signal:AbortSignal.timeout(8000)} : {})});
      const info = await response.json();
      if (token !== generation) return;
      if (!info.ready) { recover(info.message); return; }
      metadata = info; windowSize = Math.max(1, Math.min(8, Number(info.max_in_flight)||1)); nextSend = 0; $('fov').value = info.fov || 60; reset(); samples = []; lastDisplay = 0; current = info.time_duration[0]; clockAt=performance.now();idleSince=clockAt;
      if(resumeState && resumeState.sha===info.sha256){position=resumeState.position;yaw=resumeState.yaw;pitch=resumeState.pitch;current=resumeState.current;playing=resumeState.playing;$('fov').value=resumeState.fov;}
      liveWidth=Math.min(liveWidth,Number($('resolution').value)||960);resumeState=null;
      $('time').min = info.time_duration[0]; $('time').max = info.time_duration[1];
      $('time').disabled = false; $('play').disabled = false;
      ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/fvv/ws${query}`);
      ws.binaryType = 'blob';
      ws.onopen = () => { if (token === generation) {lastResponseAt=performance.now();reconnectAttempts=0;$('message').textContent = '';} };
      ws.onmessage = event => {
        if(token===generation)lastResponseAt=performance.now();
        decodeQueue = decodeQueue.then(async () => {
          if (token !== generation) return;
          try {
            if (typeof event.data === 'string') throw new Error(JSON.parse(event.data).error);
            let payload=event.data;
            if(metadata.frame_protocol==='fvv-jpeg-v1') {
              const header=new DataView(await payload.slice(0,8).arrayBuffer());
              if(token!==generation)return;
              if(header.byteLength!==8||header.getUint32(0)!==0x46565631)throw Error('Invalid frame envelope');
              const index=inFlight.findIndex(request=>request.id===header.getUint32(4));
              if(index<0)throw Error('Unknown frame identifier');
              droppedFrames+=index;inFlight.splice(0,index);payload=payload.slice(8);
            }
            const request = inFlight[0];
            const sentAt = request?.sentAt;
            if (sentAt === undefined) throw new Error('Unexpected frame');
            if(request.cacheIndex!==undefined) {
              try {
                const state=loopState;
                await acceptLoop(request,payload);
                if(state && loopState===state && state.phase==='building') {
                  const preview=await createImageBitmap(payload);
                  if(token!==generation||loopState!==state)preview.close();
                  else {frames.push({bitmap:preview,sentAt,time:request.time});while(frames.length>12)frames.shift().bitmap.close();}
                }
              } catch(error) {
                $('smoothLoop').checked=false;invalidateView();$('message').textContent=error.message+'；已恢复实时模式。';
              }
              return;
            }
            const bitmap = await createImageBitmap(payload);
            if (token !== generation) { bitmap.close(); return; }
            receiveTimes.push(performance.now());if(receiveTimes.length>90)receiveTimes.shift();
            if($('adaptivePreview').checked && receiveTimes.length>=30 && performance.now()-lastQualityChange>2000) {
              const fps=(receiveTimes.length-1)*1000/(receiveTimes.at(-1)-receiveTimes[0]);
              const maximum=Number($('resolution').value)||960;
              if(fps<45)liveWidth=Math.max(Math.min(640,maximum),Math.floor(liveWidth*.85/32)*32);
              else if(fps>55)liveWidth=Math.min(maximum,liveWidth+32);
              lastQualityChange=performance.now();
            }
            latencies.push(performance.now()-sentAt); if(latencies.length>60)latencies.shift();
            if(request.revision===timelineRevision) frames.push({bitmap, sentAt, time:request.time});
            else bitmap.close();
            // Bound decoded memory and prefer fresh camera poses when rendering
            // faster than this display. Superseded bitmaps are always released.
            while (frames.length > 12) {frames.shift().bitmap.close();droppedFrames++;}
          } catch(error) { recover(`渲染失败：${error.message}`); }
          finally { if (token === generation) { inFlight.shift(); requestFrame(performance.now()); } }
        });
        return decodeQueue;
      };
      ws.onclose = event => { if (token === generation) { if(event?.code===1012){loopState=null;loopCache.clear();}recover('渲染连接已关闭'); } };
      ws.onerror = () => { if (token === generation) $('message').textContent = '无法连接 GPU 服务，请检查部署状态。'; };
    } catch(error) { recover(`连接失败：${error.message}`); }
    finally { connecting = false; }
  }
  function move(forward, right, up) {
    if(forward || right || up)invalidateView();
    const a = yaw*Math.PI/180, b = pitch*Math.PI/180;
    position[0] += forward*Math.sin(a)*Math.cos(b) + right*Math.cos(a);
    position[1] += forward*Math.sin(b) - up;
    position[2] += forward*Math.cos(a)*Math.cos(b) - right*Math.sin(a);
  }
  function requestFrame(now) {
    if (!metadata || document.hidden || (loopState && loopState.phase!=='building')) return;
    if(loopState && loopState.next>=loopState.count)return;
    if (ws?.readyState === WebSocket.OPEN && inFlight.length < windowSize) {
      clearTimeout(sendTimer); sendTimer=null;
      const interval=1000/(loopState ? 120 : Math.min(Number($('frameRate').value)||60,displayIntervals.length>10 ? 1000/[...displayIntervals].sort((a,b)=>a-b)[Math.floor(displayIntervals.length/2)] : 60));
      if (now < nextSend-.5) {
        sendTimer=setTimeout(()=>requestFrame(performance.now()),Math.max(1,nextSend-now)); return;
      }
      const width = loopState || !$('adaptivePreview').checked ? Number($('resolution').value) : Math.min(Number($('resolution').value),liveWidth), ratio = canvas.clientHeight/Math.max(1,canvas.clientWidth);
      advanceClock(now);
      const request={id:++requestSerial,sentAt:now,time:current,revision:timelineRevision};
      if(loopState) {
        request.cacheIndex=(loopState.baseIndex+loopState.next++)%loopState.count;request.cacheVersion=loopState.version;
        const [lo,hi]=metadata.time_duration;request.time=lo+request.cacheIndex/loopState.count*(hi-lo);
      }
      inFlight.push(request); nextSend=Math.max(now,nextSend)+interval;
      ws.send(JSON.stringify({...(metadata.frame_protocol==='fvv-jpeg-v1' ? {request_id:request.id,coalesce:!loopState} : {}),position, yaw, pitch, time:request.time, width, temporal_samples:playing && (loopState || !$('adaptivePreview').checked || now-idleSince>400) ? (Number($('temporalSamples').value)||3) : 1,
        height:Math.max(256,Math.min(1280,Math.round(width*ratio))), fov:Number($('fov').value),jpeg_quality:Number($('imageQuality').value)||95}));
      if (inFlight.length < windowSize) sendTimer=setTimeout(()=>requestFrame(performance.now()),interval);
    }
  }
  function tick(now) {
    const dt = Math.min((now-last)/1000, .1); last = now;
    if(lastTick && now-lastTick>2 && now-lastTick<100){displayIntervals.push(now-lastTick);if(displayIntervals.length>60)displayIntervals.shift();}lastTick=now;
    if (metadata && !document.hidden) {
      if($('smoothLoop').checked && !loopState && playing && !drag && !keys.size && now-idleSince>1200 && lastDisplay && ws?.readyState===WebSocket.OPEN) {
        try{beginLoop();}catch(error){$('smoothLoop').checked=false;$('message').textContent=error.message;}
      }
      if(loopState?.phase==='playing') {
        advanceClock(now);
        const [lo,hi]=metadata.time_duration;
        const index=Math.min(loopState.count-1,Math.floor((current-lo)/(hi-lo)*loopState.count));
        loopCache.prepare(index).catch(error=>{invalidateView();$('message').textContent='缓存解码失败，恢复实时渲染：'+error.message;});
        const bitmap=loopCache.get(index);
        if(bitmap && index!==lastLoopIndex) {present(bitmap,lo+index/loopState.count*(hi-lo),now,now,false);lastLoopIndex=index;}
      }
      // A late packet burst is drained on a presentation clock instead of
      // dropping every frame whose original send deadline already expired.
      const target=1000/Math.min(Number($('frameRate').value)||60,120);
      const measured=receiveTimes.length>10 ? (receiveTimes.at(-1)-receiveTimes[0])/(receiveTimes.length-1) : target;
      const cadence=Math.max(target,Math.min(100,measured));
      playoutDelay=cadence*2;
      if((!loopState || loopState.phase==='building') && frames.length) {
        if(!presentationStarted && (!lastDisplay || frames.length>=2 || now-frames[0].sentAt>100)) {
          presentationStarted=true;nextPresentation=now;
        }
        if(presentationStarted && now>=nextPresentation-.5) {
          // Bound obsolete camera latency, while retaining a short burst.
          while(frames.length>1 && now-frames[0].sentAt>400){frames.shift().bitmap.close();droppedFrames++;}
          const {bitmap,sentAt,time}=frames.shift();present(bitmap,time,now,sentAt);
          nextPresentation=Math.max(nextPresentation+cadence,now);
        }
      } else if(!loopState && lastDisplay && now-lastDisplay>250) presentationStarted=false;
      const speed = Math.max(.01, Math.min(100, Number($('speed').value)||1))*dt;
      move((Number(keys.has('w'))-Number(keys.has('s')))*speed,
           (Number(keys.has('d'))-Number(keys.has('a')))*speed,
           (Number(keys.has('e'))-Number(keys.has('q')))*speed);
      advanceClock(now);
      displayTimes=displayTimes.filter(t=>now-t<1000);
      const sortedIntervals=samples.slice(-120).map(s=>s.interval_ms).sort((a,b)=>a-b);
      const p95=sortedIntervals.length ? sortedIntervals[Math.floor((sortedIntervals.length-1)*.95)] : 0;
      $('metrics').textContent=!playing ? '已暂停 · 移动视角可查看当前时刻' : loopState && loopState.phase!=='playing' ? `${displayTimes.length} FPS 预览 · 正在预载，完成后按正常速度循环播放` : `${displayTimes.length} FPS 显示（最近1秒） · 帧间隔 P95 ${p95.toFixed(0)} ms · 缓冲 ${loopState?.phase==='playing' ? '本地循环' : playoutDelay.toFixed(0)+' ms · 实时 '+liveWidth+' px'}`;
      // A slow oldest request is not a dead connection: packets may still be
      // arriving. Reconnecting it creates a much longer status/retry outage.
      if(inFlight.length && $('adaptivePreview').checked && !loopState && frames.length<2 && now-inFlight[0].sentAt>350 && now-lastQualityChange>500) {
        liveWidth=Math.max(Math.min(640,Number($('resolution').value)),Math.floor(liveWidth*.85/32)*32);lastQualityChange=now;
      }
      if (inFlight.length && now-lastResponseAt > 8000) { recover('渲染等待超时'); }
      requestFrame(now);
    }
    requestAnimationFrame(tick);
  }
  canvas.addEventListener('pointerdown', e => { canvas.focus(); canvas.setPointerCapture(e.pointerId); drag = [e.clientX,e.clientY]; });
  canvas.addEventListener('pointermove', e => { if (!drag) return; invalidateView();yaw += (e.clientX-drag[0])*.2; pitch = Math.max(-89,Math.min(89,pitch+(e.clientY-drag[1])*.2)); drag = [e.clientX,e.clientY]; });
  canvas.addEventListener('pointerup', () => { drag=null; });
  canvas.addEventListener('pointercancel', () => { drag=null; });
  canvas.addEventListener('wheel', e => { e.preventDefault(); if (position) move(-Math.sign(e.deltaY)*.1*(Number($('speed').value)||1),0,0); }, {passive:false});
  canvas.addEventListener('keydown', e => { const key=e.key.toLowerCase(); if ('wasdqe'.includes(key) && key.length===1) { keys.add(key); e.preventDefault(); } if (key==='r') reset(); });
  window.addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());
  $('reset').onclick = reset;
  $('play').onclick = () => { advanceClock(performance.now()); playing = !playing; if(loopState && (!playing || loopState.phase!=='playing'))invalidateView(); $('play').textContent = playing ? '暂停' : '播放'; };
  $('time').oninput = () => { invalidateView();timelineRevision++;current = Number($('time').value); clockAt=performance.now(); playing=false; for(const frame of frames)frame.bitmap.close();frames=[]; $('play').textContent='播放'; };
  $('reconnect').onclick = connect;
  $('frameRate').onchange=()=>{invalidateView();nextSend=0;samples=[];lastDisplay=0;requestFrame(performance.now());};
  $('imageQuality').onchange=()=>{invalidateView();samples=[];lastDisplay=0;};
  $('resolution').onchange = () => { liveWidth=Number($('resolution').value);invalidateView();samples=[]; lastDisplay=0; };
  $('adaptivePreview').onchange=()=>{liveWidth=Number($('resolution').value);invalidateView();};$('temporalSamples').onchange=invalidateView;$('smoothLoop').onchange=invalidateView;$('fov').oninput=invalidateView;
  window.addEventListener('resize',invalidateView);
  $('report').onclick = () => {
    const report = {scene:metadata?.scene, revision:metadata?.revision, model_sha256:metadata?.sha256,
      loop_cache:loopState ? {phase:loopState.phase,rate:loopState.rate,frames:loopState.count,bytes:loopCache.bytes,preparation_ms:loopState.preparation_ms} : null, resolution:Number($('resolution').value),target_fps:Number($('frameRate').value),measurement:'requestAnimationFrame canvas presentation',playout_delay_ms:playoutDelay,dropped_frames:droppedFrames, max_in_flight:windowSize, transport:`JPEG quality=${Number($('imageQuality').value)||95} subsampling=0`, samples};
    const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const link = document.createElement('a'); link.href=url; link.download='fvv-performance.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  document.addEventListener('visibilitychange', () => { if (document.hidden) {savePose();disconnect();} else { lastDisplay=0; connect(); } });
  window.addEventListener('pagehide', disconnect);
  window.FVV = {connect, disconnect};
  connect(); requestAnimationFrame(tick);
})();
