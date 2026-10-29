'use strict';
(() => {
  const $ = id => document.getElementById(id), canvas = $('fvvCanvas'), ctx = canvas.getContext('2d');
  const query = location.search;
  $('back').href = '/' + query;
  let ws, metadata, position, yaw = 0, pitch = 0, current = 0, playing = true;
  let inFlight = [], windowSize = 1, lastSent = -Infinity, decodeQueue = Promise.resolve();
  let last = performance.now(), lastDisplay = 0, generation = 0, connecting = false;
  let samples = [], keys = new Set(), drag = null;
  function reset() {
    if (!metadata) return;
    position = [...metadata.position]; yaw = metadata.yaw; pitch = metadata.pitch;
    keys.clear();
  }
  function disconnect() {
    generation++; inFlight = []; decodeQueue = Promise.resolve(); keys.clear();
    if (ws) { ws.onclose = null; ws.close(); ws = null; }
  }
  async function connect() {
    if (connecting) return;
    connecting = true; disconnect();
    const token = generation;
    $('message').textContent = '正在连接 Hybrid 模型…';
    try {
      const response = await fetch('/fvv/status' + query, {cache:'no-store'});
      const info = await response.json();
      if (token !== generation) return;
      if (!info.ready) { $('message').textContent = info.message; return; }
      metadata = info; windowSize = Math.max(1, Math.min(3, Number(info.max_in_flight)||1)); lastSent = -Infinity; $('fov').value = info.fov || 60; reset(); samples = []; lastDisplay = 0; current = info.time_duration[0];
      $('time').min = info.time_duration[0]; $('time').max = info.time_duration[1];
      $('time').disabled = false; $('play').disabled = false;
      ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/fvv/ws${query}`);
      ws.binaryType = 'blob';
      ws.onopen = () => { if (token === generation) $('message').textContent = ''; };
      ws.onmessage = event => {
        decodeQueue = decodeQueue.then(async () => {
          if (token !== generation) return;
          try {
            if (typeof event.data === 'string') throw new Error(JSON.parse(event.data).error);
            const sentAt = inFlight[0];
            if (sentAt === undefined) throw new Error('Unexpected frame');
            const bitmap = await createImageBitmap(event.data);
            if (token !== generation) { bitmap.close(); return; }
            if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
            if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
            ctx.drawImage(bitmap, 0, 0); bitmap.close();
            const now = performance.now(), interval = lastDisplay ? now-lastDisplay : null;
            lastDisplay = now;
            if (interval) {
              samples.push({interval_ms:interval, roundtrip_ms:now-sentAt});
              if (samples.length > 300) samples.shift();
              const fps = 1000 / (samples.reduce((s,x)=>s+x.interval_ms,0)/samples.length);
              $('metrics').textContent = `${fps.toFixed(1)} FPS · ${(now-sentAt).toFixed(0)} ms 交互往返${fps < 30 ? ' · 未达到实时验收目标' : ''}`;
            }
          } catch(error) { $('message').textContent = `渲染失败：${error.message}`; disconnect(); }
          finally { if (token === generation) { inFlight.shift(); requestFrame(performance.now()); } }
        });
        return decodeQueue;
      };
      ws.onclose = () => { if (token === generation) { disconnect(); $('message').textContent = '连接已关闭（服务可能忙碌）。请重新连接。'; } };
      ws.onerror = () => { if (token === generation) $('message').textContent = '无法连接 GPU 服务，请检查部署状态。'; };
    } catch(error) { $('message').textContent = `连接失败：${error.message}`; }
    finally { connecting = false; }
  }
  function move(forward, right, up) {
    const a = yaw*Math.PI/180, b = pitch*Math.PI/180;
    position[0] += forward*Math.sin(a)*Math.cos(b) + right*Math.cos(a);
    position[1] += forward*Math.sin(b) - up;
    position[2] += forward*Math.cos(a)*Math.cos(b) - right*Math.sin(a);
  }
  function requestFrame(now) {
    if (!metadata || document.hidden) return;
    if (ws?.readyState === WebSocket.OPEN && inFlight.length < windowSize && now-lastSent >= 1000/60 - .5) {
      const width = Number($('resolution').value), ratio = canvas.clientHeight/Math.max(1,canvas.clientWidth);
      inFlight.push(now); lastSent = now;
      ws.send(JSON.stringify({position, yaw, pitch, time:current, width,
        height:Math.max(256,Math.min(1280,Math.round(width*ratio))), fov:Number($('fov').value)}));
    }
  }
  function tick(now) {
    const dt = Math.min((now-last)/1000, .1); last = now;
    if (metadata && !document.hidden) {
      const speed = Math.max(.01, Math.min(100, Number($('speed').value)||1))*dt;
      move((Number(keys.has('w'))-Number(keys.has('s')))*speed,
           (Number(keys.has('d'))-Number(keys.has('a')))*speed,
           (Number(keys.has('e'))-Number(keys.has('q')))*speed);
      if (playing) {
        const [lo,hi] = metadata.time_duration;
        current = lo + (current-lo+dt)%(hi-lo);
      }
      $('time').value = current; $('timestamp').textContent = current.toFixed(3)+' s';
      if (inFlight.length && now-inFlight[0] > 10000) { $('message').textContent = '渲染超时，请重新连接。'; disconnect(); }
      requestFrame(now);
    }
    requestAnimationFrame(tick);
  }
  canvas.addEventListener('pointerdown', e => { canvas.focus(); canvas.setPointerCapture(e.pointerId); drag = [e.clientX,e.clientY]; });
  canvas.addEventListener('pointermove', e => { if (!drag) return; yaw += (e.clientX-drag[0])*.2; pitch = Math.max(-89,Math.min(89,pitch+(e.clientY-drag[1])*.2)); drag = [e.clientX,e.clientY]; });
  canvas.addEventListener('pointerup', () => { drag=null; });
  canvas.addEventListener('pointercancel', () => { drag=null; });
  canvas.addEventListener('wheel', e => { e.preventDefault(); if (position) move(-Math.sign(e.deltaY)*.1*(Number($('speed').value)||1),0,0); }, {passive:false});
  canvas.addEventListener('keydown', e => { const key=e.key.toLowerCase(); if ('wasdqe'.includes(key) && key.length===1) { keys.add(key); e.preventDefault(); } if (key==='r') reset(); });
  window.addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());
  $('reset').onclick = reset;
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '暂停' : '播放'; };
  $('time').oninput = () => { current = Number($('time').value); playing=false; $('play').textContent='播放'; };
  $('reconnect').onclick = connect;
  $('resolution').onchange = () => { samples=[]; lastDisplay=0; };
  $('report').onclick = () => {
    const report = {scene:metadata?.scene, revision:metadata?.revision, model_sha256:metadata?.sha256,
      resolution:Number($('resolution').value), max_in_flight:windowSize, transport:'JPEG quality=92 subsampling=0', samples};
    const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const link = document.createElement('a'); link.href=url; link.download='fvv-performance.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  document.addEventListener('visibilitychange', () => { if (document.hidden) disconnect(); else { lastDisplay=0; connect(); } });
  window.addEventListener('pagehide', disconnect);
  window.FVV = {connect, disconnect};
  connect(); requestAnimationFrame(tick);
})();
