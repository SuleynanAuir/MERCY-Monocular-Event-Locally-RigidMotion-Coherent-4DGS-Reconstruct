/* Protocol + presentation-clock benchmark, NOT a browser/Canvas FPS test.
 * node scripts/viewer/benchmark_viewer_stream.cjs [URL] [seconds] [rate] [width] [warmup]
 */
const GaussianPlayback = require('../../viewer/playback.js');
const base = process.argv[2] || 'http://127.0.0.1:8892';
const seconds = Number(process.argv[3] || 15), rate = Number(process.argv[4] || 30);
const width = Number(process.argv[5] || 640), warmup = Number(process.argv[6] || Math.ceil(rate / 6));
const quantile = (values, q) => values.length ? [...values].sort((a, b) => a - b)[Math.floor((values.length - 1) * q)] : null;
(async () => {
  const status = await (await fetch(`${base}/api/status`)).json();
  if (!status.ready) throw Error(JSON.stringify(status));
  const quality = status.scene.default_stream_quality ?? 78;
  const socket = new WebSocket(base.replace(/^http/, 'ws') + '/ws');
  socket.binaryType = 'arraybuffer';
  const buffer = new GaussianPlayback(rate, warmup, 24);
  const received = [], displayed = [], render = [], gaps = [], bytes = [];
  const cache = new Set(), pendingCache = new Set();
  let first = null, last = null, ack = -1, hits = 0, clientHits = 0, timer, finishTimer;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(Error('Stream timed out')), (seconds + 40) * 1000);
    socket.onopen = () => socket.send(JSON.stringify({type: 'stream', id: 1, frame: 0, speed: 1,
      rate, loop: true, width, height: width, fovy: status.scene.fovy, background: 'black', quality}));
    socket.onerror = reject;
    socket.onmessage = ({data}) => {
      if (typeof data === 'string') {reject(Error(data)); return;}
      const now = performance.now(), length = new DataView(data).getUint32(0, true);
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, length)));
      if (meta.client_cache_hit) {
        if (!cache.has(meta.frame)) {reject(Error('Missing cached JPEG')); return;}
        clientHits++;
      } else if (!cache.has(meta.frame)) {
        cache.add(meta.frame); pendingCache.add(meta.frame);
      }
      if (meta.comparison && meta.left_frame !== meta.right_frame) {reject(Error('Panels out of sync')); return;}
      if (first === null) {
        first = now;
        finishTimer = setTimeout(() => {clearTimeout(timeout); resolve();}, seconds * 1000);
      }
      if (last !== null) gaps.push(now - last);
      last = now; received.push(now); render.push(meta.render_ms + meta.encode_ms);
      bytes.push(data.byteLength); hits += Number(!!meta.cache_hit);
      buffer.push({meta, bitmap: {close() {}}});
    };
    timer = setInterval(() => {
      const item = buffer.take(performance.now());
      if (!item || item.ended) return;
      displayed.push(performance.now());
      if (buffer.last - ack >= 4) {
        const retained = [...pendingCache].slice(0, 24);
        socket.send(JSON.stringify({type: 'ack', id: 1, seq: buffer.last, cached_frames: retained}));
        retained.forEach(frame => pendingCache.delete(frame)); ack = buffer.last;
      }
    }, 1000 / 60);
  }).finally(() => {clearInterval(timer); clearTimeout(finishTimer); socket.close(); buffer.close();});
  const intervals = displayed.slice(1).map((t, i) => t - displayed[i]);
  const tailBytes = bytes.reduce((sum, size, i) => sum + (received[i] >= last - 5000 ? size : 0), 0);
  console.log(JSON.stringify({kind: 'WebSocket + simulated presentation clock; excludes browser decode/draw',
    seconds, rate, width, quality, warmup, received: received.length, displayed: displayed.length,
    presentation_fps: (displayed.length - 1) * 1000 / (displayed.at(-1) - displayed[0]),
    underruns: buffer.underruns, dropped: buffer.dropped, cache_hits: hits, client_cache_hits: clientHits,
    render_encode_mean_ms: render.reduce((a,b) => a+b, 0) / render.length,
    render_encode_p95_ms: quantile(render, .95), wire_gap_p95_ms: quantile(gaps, .95),
    display_gap_p95_ms: quantile(intervals, .95), display_gap_max_ms: Math.max(...intervals),
    mean_packet_kib: bytes.reduce((a,b) => a+b,0) / bytes.length / 1024,
    received_mbps: bytes.reduce((a,b) => a+b,0) * 8 / seconds / 1e6,
    last_five_seconds_mbps: tailBytes * 8 / 5 / 1e6}, null, 2));
})().catch(error => {console.error(error); process.exit(1);});
