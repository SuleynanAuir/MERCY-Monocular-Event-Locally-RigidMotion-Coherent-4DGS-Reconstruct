// Capture both Gaussian sources at identical cameras for offline Mesh color fitting.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

(async () => {
  const [url = 'http://127.0.0.1:8892', output, mode = 'fit'] = process.argv.slice(2);
  if (!output || fs.existsSync(output)) throw Error('Provide a new output directory');
  if (!['fit', 'check', 'dynamic'].includes(mode)) throw Error('Capture mode must be fit, check or dynamic');
  const status = await (await fetch(`${url}/api/status`)).json();
  if (!status.ready || !status.scene.comparison) throw Error('Paired viewer is not ready');
  const plan = [];
  for (const pitch of [0, 30]) for (let yaw = 0; yaw < 360; yaw += 30) {
    plan.push({split: 'train', frame: (yaw / 30 % 2) * 100, yaw, pitch});
  }
  for (const [frame, yaw, pitch] of [[200,75,10], [300,225,25], [0,-30,-15], [200,135,55], [100,35,20], [100,180,35]]) {
    plan.push({split: 'validation', frame, yaw, pitch});
  }
  if (mode === 'check') {
    plan.length = 0;
    for (const [frame, yaw, pitch] of [[125,15,15], [275,165,45], [350,285,-10], [50,315,60]]) {
      plan.push({split: 'validation', frame, yaw, pitch});
    }
  }
  if (mode === 'dynamic') {
    plan.length = 0;
    // Held-out cameras and consecutive time samples expose motion-dependent
    // errors that a few attractive still images can hide.
    for (const [yaw, pitch] of [[15,15], [165,45], [285,-10]]) {
      for (let frame = 36; frame < 228; frame += 4) {
        plan.push({split: 'validation', frame, yaw, pitch});
      }
    }
  }
  fs.mkdirSync(output, {recursive: true});
  const ws = new WebSocket(`${url.replace(/^http/, 'ws')}/ws`);
  ws.binaryType = 'arraybuffer';
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
  const views = [];
  try {
    for (const [id, view] of plan.entries()) {
      const request = {id, render_mode: 'gaussian', frame: view.frame, yaw: view.yaw, pitch: view.pitch,
        width: mode === 'dynamic' ? 768 : 1024, height: mode === 'dynamic' ? 768 : 1024,
        fovy: status.scene.fovy, background: 'black', quality: 95};
      const received = new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(Error('Render timeout')), 30000);
        ws.onmessage = e => { clearTimeout(timer); resolve(e.data); };
      });
      ws.send(JSON.stringify(request));
      const data = await received;
      if (typeof data === 'string') throw Error(data);
      const buffer = Buffer.from(data), size = buffer.readUInt32LE(0);
      const header = JSON.parse(buffer.subarray(4, 4 + size));
      if (header.id !== id || !header.comparison || header.left_frame !== view.frame || header.right_frame !== view.frame) throw Error('Mismatched paired render');
      const jpeg = buffer.subarray(4 + size), file = `view-${String(id).padStart(2,'0')}.jpg`;
      fs.writeFileSync(path.join(output, file), jpeg);
      views.push({...view, request, file, sha256: crypto.createHash('sha256').update(jpeg).digest('hex')});
      console.log(`${id + 1}/${plan.length}: ${view.split} frame=${view.frame} yaw=${view.yaw} pitch=${view.pitch}`);
    }
    fs.writeFileSync(path.join(output, 'views.json'), JSON.stringify({format: 'gaussian_mesh_color_views_v1',
      purpose: mode, scene: status.scene, views}, null, 2) + '\n');
  } finally { ws.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
