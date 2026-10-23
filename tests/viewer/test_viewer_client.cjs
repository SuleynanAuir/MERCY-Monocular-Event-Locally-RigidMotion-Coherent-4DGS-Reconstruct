/* Frontend logic with a fake DOM/socket/decoder. Not a browser rendering test. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const viewer = path.join(__dirname, '../../viewer');

async function client(overrides = {}, catalog = null, search = '?autoplay=1&loop=1', referenceFrame = null) {
  const ids = [...fs.readFileSync(path.join(viewer, 'index.html'), 'utf8').matchAll(/id="([^"]+)"/g)].map(m => m[1]);
  const element = () => ({value: '', hidden: false, dataset: {}, checked: false, textContent: '',
    classList: {toggle() {}, remove() {}}, style: {setProperty(k,v) {this[k]=v;}}, children: [], setAttribute() {},
    replaceChildren() {this.children=[];}, append(...nodes) {this.children.push(...nodes);}, addEventListener() {},
    focus() {}, setPointerCapture() {},
    getContext() {return {drawImage() {},setTransform() {},clearRect() {},save() {},restore() {},strokeRect() {},fillRect() {},fillText() {}};}, getBoundingClientRect() {return {width: 800, height: 500};}});
  const elements = Object.fromEntries(ids.map(id => [id, element()]));
  const presets = [...fs.readFileSync(path.join(viewer, 'index.html'), 'utf8')
    .matchAll(/class="preset(?: active)?" data-yaw="([^"]+)" data-pitch="([^"]+)"/g)]
    .map(match => Object.assign(element(), {dataset: {yaw: match[1], pitch: match[2]}}));
  const scene = {comparison: true, frame_count: 20, prewarmed_frames: 20, fps: 60,
    duration_seconds: 20/60, fovy: 40, default_resolution: 640, default_background: 'black',
    stream_window: 24, buffer_seconds: .4, gaussian_count: 1841816, cache: 'gpu-lru', cache_gib: 5,
    times_seconds: Array.from({length: 20}, (_,i) => i/60), rpm: Array(20).fill(null)};
  Object.assign(scene, overrides);
  const callbacks = [], decoded = [], errors = [];
  let socket, resize, now = 0;
  class Socket {
    static OPEN = 1;
    constructor(url) {this.url = url; socket = this; this.readyState = 1; this.sent = [];}
    send(data) {this.sent.push(JSON.parse(data));}
    close() {}
  }
  const context = {
    document: {hidden: false, getElementById(id) {assert.ok(elements[id], `Missing HTML element ${id}`); return elements[id];},
      querySelectorAll(selector) {return selector === '.preset' ? presets : [];},
      createTextNode: text => text, createElement: element, addEventListener() {}},
    location: {search, protocol: 'http:', host: '127.0.0.1:8892'},
    fetch: async () => ({ok: true, json: async () => ({ready: true, scene, ...(catalog || {})})}),
    ViewerReferences: class {constructor() {if (referenceFrame?.prepare) this.prepare = referenceFrame.prepare;} frame(index) {return referenceFrame ? referenceFrame(index) : Promise.resolve({close() {}});}},
    WebSocket: Socket, performance: {now: () => now},
    requestAnimationFrame: callback => callbacks.push(callback),
    ResizeObserver: class {constructor(callback) {resize = callback;} observe() {}},
    createImageBitmap: async blob => {decoded.push(blob); return {close() {}};},
    URL, URLSearchParams, Blob, AbortSignal, TextDecoder, Uint8Array, DataView,
    setTimeout() {}, clearTimeout() {}, addEventListener() {}, console: {error: e => errors.push(e)},
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(viewer, 'playback.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(viewer, 'app.js'), 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  socket.onopen();
  const tick = value => {now = value; callbacks.shift()(value);};
  tick(0);
  const stream = () => socket.sent.filter(m => m.type === 'stream').at(-1);
  async function packet(seq, frame, reference = false, id = stream().id) {
    const header = Buffer.from(JSON.stringify({id, frame, stream: {seq}, width: 1280,
      height: 640, panel_width: 640, comparison: true, left_frame: frame, right_frame: frame,
      render_ms: 0, encode_ms: 0, cache_hit: true, client_cache_hit: reference}));
    const data = new ArrayBuffer(4 + header.length + (reference ? 0 : 4));
    new DataView(data).setUint32(0, header.length, true);
    new Uint8Array(data, 4, header.length).set(header);
    await socket.onmessage({data});
  }
  return {elements, presets, socket, stream, tick, packet, decoded, resize, errors, context};
}

test('default cached view selects 60 FPS, new camera selects 30, layout does not restart playback', async () => {
  const c = await client();
  assert.equal(c.stream().rate, 60);
  const id = c.stream().id;
  c.resize(); c.tick(16);
  assert.equal(c.stream().id, id);
  c.presets[1].onclick(); c.tick(32);
  assert.equal(c.stream().rate, 30);
  assert.notEqual(c.stream().id, id);
});

test('1024px Clay starts at 30 FPS without lowering quality or motion speed and honors manual 60', async () => {
  const c = await client({default_stream_quality: 95,
    clay: {available: true, triangle_count: 100, default_resolution: 1024, prewarmed_frames: 20}}, null,
    '?mode=clay&autoplay=1&loop=1');
  assert.equal(c.stream().render_mode, 'clay');
  assert.equal(c.stream().rate, 30);
  assert.equal(c.stream().width, 1024);
  assert.equal(c.stream().quality, 95);
  assert.equal(c.stream().speed, 1);
  c.elements.streamRate.value = '60'; c.elements.streamRate.onchange(); c.tick(16);
  assert.equal(c.stream().rate, 60);
  assert.equal(c.stream().speed, 1);
});

test('an obsolete preparation advertisement does not delay playback until all 400 frames arrive', async () => {
  const c = await client({frame_count: 400, prewarmed_frames: 400,
    times_seconds: Array.from({length: 400}, (_, i) => i / 60), rpm: Array(400).fill(null),
    camera_preparation: {enabled: true, max_frames: 600, max_bytes: 128 * 1024**2, window: 24}});
  assert.equal(c.stream().type, 'stream');
  assert.equal(c.socket.sent.some(message => message.type === 'prepare'), false);
  for (let i = 0; i < 20; i++) await c.packet(i, i);
  c.tick(100); c.tick(100 + 1000 / 60);
  assert.equal(c.context.gaussianViewerStats.presented, 2);
  assert.equal(c.elements.canvas.dataset.frame, '1');
  assert.equal(c.elements.comparisonCanvas.dataset.frame, '1');
  assert.equal(c.decoded.length, 20, 'short warmup starts playback without the other 380 frames');
});

test('back and elevation presets change actual render camera and keep source motion speed', async () => {
  const c = await client();
  for (const [index, yaw, pitch, label] of [[3, 180, 0, '背面'], [4, 0, 35, '俯视'], [5, 0, -35, '仰视']]) {
    c.presets[index].onclick(); c.tick(16 * index);
    assert.equal(c.stream().yaw, yaw);
    assert.equal(c.stream().pitch, pitch);
    assert.equal(c.stream().speed, 1);
    assert.equal(c.elements.viewLabel.textContent, `${label} · 透视`);
  }
});

test('scene quality settings preserve cached playback timing and single-frame quality', async () => {
  const c = await client({default_stream_quality: 92, default_still_quality: 95});
  assert.equal(c.stream().quality, 92);
  assert.equal(c.stream().rate, 60);
  assert.equal(c.stream().speed, 1);
  const firstStill = c.socket.sent.find(m => m.type === undefined);
  assert.equal(firstStill.quality, 95);
  const id = c.stream().id;
  c.resize(); c.tick(16);
  assert.equal(c.stream().id, id);
  c.presets[1].onclick(); c.tick(32);
  assert.equal(c.stream().quality, 92);
  assert.equal(c.stream().rate, 30);
  c.elements.play.onclick(); c.tick(48);
  assert.equal(c.socket.sent.at(-1).quality, 95);
});

test('legacy scenes keep the old stream and still quality defaults', async () => {
  const c = await client();
  assert.equal(c.stream().quality, 78);
  assert.equal(c.socket.sent.find(m => m.type === undefined).quality, 85);
  c.elements.play.onclick(); c.tick(16);
  assert.equal(c.socket.sent.at(-1).quality, 85);
});

test('camera interaction keeps its inexpensive quality and resumes the scene stream quality', async () => {
  const c = await client({default_stream_quality: 92, default_still_quality: 95});
  c.elements.stage.onpointerdown({target: {closest: () => null}, pointerId: 1, clientX: 0, clientY: 0});
  c.elements.stage.onpointermove({pointerId: 1, clientX: 10, clientY: 0}); c.tick(16);
  assert.equal(c.socket.sent.at(-1).quality, 72);
  c.elements.stage.onpointerup({pointerId: 1}); c.tick(32);
  assert.equal(c.stream().quality, 92);
  assert.equal(c.stream().rate, 30);
  assert.equal(c.stream().speed, 1);
});

test('both canvases consume the retained JPEG, and a replaced stream rejects stale references', async () => {
  const c = await client({default_stream_quality: 92}), id = c.stream().id;
  for (let i = 0; i < 20; i++) await c.packet(i, i);
  for (let i = 0; i < 4; i++) c.tick(100 + i * 1000 / 60);
  const ack = c.socket.sent.filter(m => m.type === 'ack').at(-1);
  assert.equal(ack.seq, 3); assert.equal(ack.cached_frames.length, 20);
  assert.equal(c.elements.canvas.dataset.frame, '3');
  assert.equal(c.elements.comparisonCanvas.dataset.frame, '3');
  await c.packet(20, 0, true);
  assert.equal(c.decoded[20], c.decoded[0]);
  c.presets[1].onclick(); c.tick(200);
  const count = c.decoded.length;
  await c.packet(21, 1, true, id);
  assert.equal(c.decoded.length, count);
  // New streams never claim a JPEG from an earlier camera.
  await c.packet(0, 0, true);
  assert.equal(c.errors.length, 1);
  assert.equal(c.elements.loadTitle.textContent, '画面解析失败');
});

test('automatic mode lowers the target after two buffer underruns without changing motion speed', async () => {
  const c = await client();
  for (let i = 0; i < 20; i++) await c.packet(i, i);
  for (let i = 0; i <= 20; i++) c.tick(100 + i * 1000 / 60);
  assert.equal(c.context.gaussianViewerStats.underruns, 1);
  for (let i = 20; i < 40; i++) await c.packet(i, i % 20);
  for (let i = 0; i <= 20; i++) c.tick(600 + i * 1000 / 60);
  assert.equal(c.stream().rate, 30);
  assert.equal(c.stream().speed, 1);
});

test('comparison keeps labels and playback while diagnostic panels are absent', async () => {
  const c = await client({left_label: '最原始结果 · 未校正初始动画',
    right_label: '进一步优化后 · 全局 + 局部细节 + 事件约束',
    event_validation: {approved: false, summary: {passed: 382, intervals: 399, cosine: .944},
      interval_checks: [{passed: false, reasons: ['relative_l1', 'polarity_agreement',
        'cosine', 'unsupported_change_fraction'], reference_weak: true}]}});
  assert.match(c.elements.leftCompareLabel.textContent, /^最原始结果/);
  assert.match(c.elements.rightCompareLabel.textContent, /^进一步优化后/);
  assert.match(c.elements.sequenceBadge.textContent, /同帧同视角/);
  assert.match(c.elements.snapshot.textContent, /原始 \/ 优化对比/);
  for (let i = 0; i < 20; i++) await c.packet(i, (i + 1) % 20);
  c.tick(100);
  assert.equal(c.elements.timeCurrent.textContent, (1 / 60).toFixed(2));
  const html = fs.readFileSync(path.join(viewer, 'index.html'), 'utf8');
  for (const id of ['localMetricPanel', 'eventNotice', 'trajectoryNotice']) {
    assert.ok(!html.includes(`id="${id}"`));
  }
});

test('Mesh switch preserves the two comparison panels, frame, camera and isolated streams', async () => {
  const c = await client({mesh: {available: true, triangle_count: 12345, description: '高斯深度融合',
    default_resolution: 768, left_label: 'V1 Mesh · 原始结果', right_label: 'V3 Mesh · 最新优化结果'}});
  const oldId = c.stream().id;
  c.elements.play.onclick(); c.tick(16);
  c.elements.timeline.oninput({target: {value: '7'}}); c.tick(32);
  c.presets[1].onclick(); c.tick(48);
  c.elements.renderMode.onchange({target: {value: 'mesh'}}); c.tick(64);
  const mesh = c.socket.sent.at(-1);
  assert.equal(mesh.render_mode, 'mesh');
  assert.equal(mesh.frame, 7);
  assert.equal(mesh.yaw, 35);
  assert.equal(c.elements.comparisonCanvas.hidden, false);
  assert.equal(c.elements.leftCompareLabel.textContent, 'V1 Mesh · 原始结果');
  assert.equal(c.elements.rightCompareLabel.textContent, 'V3 Mesh · 最新优化结果');
  assert.equal(c.elements.snapshot.textContent, '保存 V1 / V3 Mesh 对比图 ↓');
  assert.equal(c.elements.sceneName.textContent, 'V1 / V3 Mesh 对比');
  assert.equal(mesh.width, 768);
  assert.equal(mesh.height, 768);
  assert.equal(c.elements.primitiveLabel.textContent, 'TRIANGLES / 三角面数量');
  assert.equal(c.elements.gaussianCount.textContent, '12,345');
  const count = c.decoded.length;
  await c.packet(0, 0, true, oldId);
  assert.equal(c.decoded.length, count);
  c.elements.play.onclick(); c.tick(80);
  assert.equal(c.stream().render_mode, 'mesh');
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(96);
  assert.equal(c.stream().render_mode, 'gaussian');
  assert.equal(c.elements.comparisonCanvas.hidden, false);
  assert.equal(c.stream().yaw, 35);
});

test('scenes without a reconstructed mesh retain Gaussian mode', async () => {
  const c = await client();
  assert.equal(c.elements.meshOption.disabled, true);
  c.elements.renderMode.onchange({target: {value: 'mesh'}}); c.tick(16);
  assert.equal(c.stream().render_mode, 'gaussian');
});

test('hybrid mode is explicit, synchronized, and isolated from Gaussian and pure Mesh streams', async () => {
  const c = await client({hybrid: {available: true, default_resolution: 768, prewarmed_frames: 20,
    left_label: 'V1 · Mesh＋Gaussian', right_label: 'V3 · Mesh＋Gaussian', description: '混合渲染，并非纯三角网格'}});
  c.elements.play.onclick(); c.tick(16);
  c.elements.timeline.oninput({target: {value: '7'}}); c.tick(32);
  c.presets[1].onclick(); c.tick(48);
  const oldId = c.stream()?.id;
  c.elements.renderMode.onchange({target: {value: 'hybrid'}}); c.tick(64);
  c.elements.play.onclick(); c.tick(80);
  const stream = c.stream();
  assert.equal(stream.render_mode, 'hybrid');
  assert.equal(stream.frame, 7); assert.equal(stream.yaw, 35);
  assert.equal(stream.width, 768); assert.notEqual(stream.id, oldId);
  assert.equal(c.elements.leftCompareLabel.textContent, 'V1 · Mesh＋Gaussian');
  assert.equal(c.elements.rightCompareLabel.textContent, 'V3 · Mesh＋Gaussian');
  assert.match(c.elements.sceneDescription.textContent, /并非纯三角网格/);
  assert.equal(c.elements.comparisonCanvas.hidden, false);
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(96);
  assert.equal(c.stream().render_mode, 'gaussian');
});

test('expand comparison enters and exits fullscreen for the shared viewport', async () => {
  const c = await client();
  c.elements.viewerPanel.requestFullscreen = async () => {c.context.document.fullscreenElement = c.elements.viewerPanel;};
  c.context.document.exitFullscreen = async () => {c.context.document.fullscreenElement = null;};
  await c.elements.expandView.onclick();
  assert.equal(c.context.document.fullscreenElement, c.elements.viewerPanel);
  await c.elements.expandView.onclick();
  assert.equal(c.context.document.fullscreenElement, null);
});

test('hybrid quality changes restart streams without moving the camera or timeline', async () => {
  const c = await client({hybrid: {available: true, default_resolution: 768, prewarmed_frames: 20,
    default_quality: 'detail', description: '混合动态展示'}});
  assert.equal(c.elements.hybridQualityField.hidden, true);
  c.elements.renderMode.onchange({target: {value: 'hybrid'}}); c.tick(16);
  assert.equal(c.stream().hybrid_quality, 'detail');
  assert.equal(c.stream().rate, 60);
  assert.equal(c.elements.hybridQualityField.hidden, false);
  c.presets[1].onclick(); c.tick(32);
  const before = c.stream();
  c.elements.hybridQuality.onchange({target: {value: 'balanced'}}); c.tick(48);
  assert.equal(c.stream().hybrid_quality, 'balanced');
  assert.notEqual(c.stream().id, before.id);
  assert.equal(c.stream().frame, before.frame);
  assert.equal(c.stream().yaw, before.yaw);
  assert.equal(c.stream().rate, 30);
  c.elements.reset.onclick(); c.tick(64);
  assert.equal(c.stream().rate, 30); // Only detail quality was prewarmed.
  c.elements.hybridQuality.onchange({target: {value: 'detail'}}); c.tick(80);
  assert.equal(c.stream().rate, 60);
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(96);
  assert.equal(c.elements.hybridQualityField.hidden, true);
  assert.equal(c.stream().render_mode, 'gaussian');
  assert.deepEqual(c.errors, []);
});

test('surface and native hybrid use the same resolution as GS and both warmed qualities allow 60 FPS', async () => {
  const c = await client({hybrid: {available: true, default_resolution: 640, prewarmed_frames: 20,
    default_quality: 'surface', prewarmed_qualities: ['surface', 'native'], description: '表面柔化',
    quality_descriptions: {surface: '内部轻度去噪', native: '关闭表面柔化', detail: '旧版高采样对照'}}});
  c.elements.renderMode.onchange({target: {value: 'hybrid'}}); c.tick(16);
  assert.equal(c.stream().hybrid_quality, 'surface');
  assert.equal(c.stream().width, 640); assert.equal(c.stream().rate, 60);
  assert.equal(c.elements.sceneDescription.textContent, '内部轻度去噪');
  const before=c.stream();
  c.elements.hybridQuality.onchange({target: {value: 'native'}}); c.tick(32);
  assert.equal(c.stream().hybrid_quality, 'native'); assert.equal(c.stream().rate, 60);
  assert.equal(c.elements.sceneDescription.textContent, '关闭表面柔化');
  assert.equal(c.stream().frame, before.frame); assert.equal(c.stream().yaw, before.yaw);
  c.elements.hybridQuality.onchange({target: {value: 'detail'}}); c.tick(48);
  assert.equal(c.stream().rate, 30);
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(64);
  assert.equal(c.stream().width, 640); assert.equal(c.stream().rate, 60);
  assert.deepEqual(c.errors, []);
});


test('object picker scopes sockets and navigation and falls back when Mesh is unavailable', async () => {
  const c = await client({comparison: false}, {scene_id: 'fan', objects: [
    {id: 'fan', name: 'Fan'}, {id: 'lego', name: 'Lego'}]}, '?scene=fan&mode=hybrid&quality=surface');
  assert.ok(c.socket.url.endsWith('/ws?scene=fan'));
  assert.equal(c.elements.objectSelect.value, 'fan');
  assert.equal(c.elements.objectSelect.disabled, false);
  assert.equal(c.elements.renderMode.value, 'gaussian');
  c.elements.objectSelect.value = 'lego';
  c.elements.objectSelect.onchange();
  const next = new URLSearchParams(c.context.location.search);
  assert.equal(next.get('scene'), 'lego');
  assert.equal(next.get('mode'), 'gaussian');
  assert.equal(next.get('quality'), 'surface');
});

test('Fan ultra resolution uses real output pixels and stays within the server budget after resizing', async () => {
  const c = await client({comparison: false, default_resolution: 1280,
    default_stream_quality: 95, default_still_quality: 95});
  assert.equal(c.elements.resolution.value, '1280');
  assert.equal(c.elements.ultraResolution.disabled, false);
  assert.equal(c.stream().width, 1280);
  assert.equal(c.stream().height, 800);
  assert.equal(c.stream().quality, 95);
  c.elements.stage.getBoundingClientRect = () => ({width: 500, height: 500});
  c.resize(); c.tick(16);
  assert.equal(c.stream().width, 1108);
  assert.equal(c.stream().height, 1108);
  assert.ok(c.stream().width * c.stream().height <= 1280 * 960);
  c.elements.stage.getBoundingClientRect = () => ({width: 390, height: 844});
  c.resize(); c.tick(32);
  assert.equal(c.stream().width, 1108);
  assert.equal(c.stream().height, 1108, 'portrait keeps the complete calibrated square view');
  assert.ok(c.stream().width * c.stream().height <= 1280 * 960);
  const lego = await client();
  assert.equal(lego.elements.ultraResolution.disabled, true);
  assert.equal(lego.stream().width, 640);
  assert.equal(lego.stream().height, 640);
});

test('Fan Mesh modes use paired square panels and return to the original HD GS view', async () => {
  const c = await client({comparison: false, comparison_modes: ['mesh', 'hybrid', 'clay'], default_resolution: 1280,
    default_stream_rate: 30, mesh: {available: true, default_resolution: 1024, triangle_count: 100,
      left_label: 'V1 修复前', right_label: 'V3 修复后'},
    clay: {available: true, default_resolution: 1024},
    hybrid: {available: true, default_resolution: 1024, default_quality: 'native'}});
  assert.equal(c.elements.comparisonCanvas.hidden, true);
  assert.equal(c.stream().width, 1280);
  c.elements.timeline.oninput({target: {value: '7'}}); c.tick(16);
  c.presets[1].onclick(); c.tick(32);
  for (const mode of ['mesh', 'hybrid', 'clay']) {
    c.elements.renderMode.onchange({target: {value: mode}}); c.tick(48);
    c.elements.play.onclick(); c.tick(64);
    assert.equal(c.elements.comparisonCanvas.hidden, false);
    assert.equal(c.stream().width, 1024); assert.equal(c.stream().height, 1024);
    assert.equal(c.stream().frame, 7); assert.equal(c.stream().yaw, 35);
    assert.equal(c.stream().rate, 30);
    c.elements.play.onclick();
  }
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(80);
  c.elements.play.onclick(); c.tick(96);
  assert.equal(c.elements.comparisonCanvas.hidden, true);
  assert.equal(c.stream().width, 1280); assert.equal(c.elements.ultraResolution.disabled, false);
  assert.deepEqual(c.errors, []);
});

test('Clay switch preserves the two comparison panels, frame, camera and isolated streams', async () => {
  const c = await client({clay: {available: true, representation: 'gaussian', description: 'GS 原生轮廓与细节',
    default_resolution: 768, left_label: 'V1 · 灰模 · 原始结果', right_label: 'V3 · 灰模 · 最新优化结果'}});
  const oldId = c.stream().id;
  c.elements.play.onclick(); c.tick(16);
  c.elements.timeline.oninput({target: {value: '7'}}); c.tick(32);
  c.presets[1].onclick(); c.tick(48);
  c.elements.renderMode.onchange({target: {value: 'clay'}}); c.tick(64);
  const mesh = c.socket.sent.at(-1);
  assert.equal(mesh.render_mode, 'clay');
  assert.equal(mesh.frame, 7);
  assert.equal(mesh.yaw, 35);
  assert.equal(c.elements.comparisonCanvas.hidden, false);
  assert.equal(c.elements.leftCompareLabel.textContent, 'V1 · 灰模 · 原始结果');
  assert.equal(c.elements.rightCompareLabel.textContent, 'V3 · 灰模 · 最新优化结果');
  assert.equal(c.elements.snapshot.textContent, '保存 V1 / V3 白模对比图 ↓');
  assert.equal(c.elements.sceneName.textContent, 'V1 / V3 白模对比');
  assert.equal(mesh.width, 768);
  assert.equal(mesh.height, 768);
  assert.equal(c.elements.primitiveLabel.textContent, 'GAUSSIANS / 高斯数量');
  assert.equal(c.elements.gaussianCount.textContent, '1,841,816');
  const count = c.decoded.length;
  await c.packet(0, 0, true, oldId);
  assert.equal(c.decoded.length, count);
  c.elements.play.onclick(); c.tick(80);
  assert.equal(c.stream().render_mode, 'clay');
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(96);
  assert.equal(c.stream().render_mode, 'gaussian');
  assert.equal(c.elements.comparisonCanvas.hidden, false);
  assert.equal(c.stream().yaw, 35);
});

test('white surface wire toggle preserves camera and frame and restarts an isolated stream', async () => {
  const c = await client({clay: {available: true, default_resolution: 1024}}, null, '?autoplay=1&mode=clay&clay_style=wire');
  assert.equal(c.elements.clayWire.checked, true);
  assert.equal(c.elements.clayStyleField.hidden, false);
  assert.equal(c.stream().clay_style, 'wire');
  c.elements.timeline.oninput({target: {value: '7'}}); c.tick(16);
  c.presets[1].onclick(); c.tick(32);
  c.elements.play.onclick(); c.tick(40);
  const previous = c.stream().id;
  c.elements.clayWire.onchange({target: {checked: false}}); c.tick(48);
  assert.notEqual(c.stream().id, previous);
  assert.equal(c.stream().clay_style, 'solid');
  assert.equal(c.stream().frame, 7); assert.equal(c.stream().yaw, 35);
  c.elements.clayWire.onchange({target: {checked: true}}); c.tick(64);
  assert.equal(c.stream().clay_style, 'wire');
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(80);
  assert.equal(c.elements.clayStyleField.hidden, true);
  assert.equal(c.elements.meshClayOption, undefined);
  const html = fs.readFileSync(path.join(viewer, 'index.html'), 'utf8');
  assert.ok(!html.includes('clay_mesh'));
});

test('clean white view defaults to solid and keeps wire an explicit option', async () => {
  const c = await client({clay: {available: true, default_resolution: 1024}}, null, '?autoplay=1&mode=clay');
  assert.equal(c.elements.clayWire.checked, false);
  assert.equal(c.stream().clay_style, 'solid');
  assert.equal(c.elements.clayStyleField.hidden, false);
});

test('pseudocolor launches as paired color mode and returns to unmodified GS controls', async () => {
  const c = await client({comparison: false, comparison_modes: ['clay', 'pseudocolor'], default_stream_rate: 30,
    pseudocolor: {available: true, default_resolution: 1024, prewarmed_frames: 20,
      left_label: 'V1 · 伪彩色', right_label: 'V3 · 伪彩色', description: '固定空间渐变'},
    clay: {available: true, default_resolution: 1024}}, null, '?autoplay=1&mode=pseudocolor');
  assert.equal(c.stream().render_mode, 'pseudocolor');
  assert.equal(c.stream().width, 1024);
  assert.equal(c.stream().rate, 30);
  assert.equal(c.elements.comparisonCanvas.hidden, false);
  assert.equal(c.elements.clayStyleField.hidden, true);
  assert.equal(c.elements.hybridQualityField.hidden, true);
  assert.equal(c.elements.leftCompareLabel.textContent, 'V1 · 伪彩色');
  assert.match(c.elements.sceneDescription.textContent, /固定空间/);
  assert.match(c.elements.snapshot.textContent, /伪彩色/);
  const oldId = c.stream().id;
  c.elements.renderMode.onchange({target: {value: 'clay'}}); c.tick(48);
  assert.equal(c.stream().render_mode, 'clay');
  assert.ok(c.stream().id > oldId);
  assert.equal(c.elements.clayStyleField.hidden, false);
  c.elements.renderMode.onchange({target: {value: 'gaussian'}}); c.tick(80);
  assert.equal(c.stream().render_mode, 'gaussian');
  assert.equal(c.elements.comparisonCanvas.hidden, true);
  assert.equal(c.elements.clayStyleField.hidden, true);
});

test('unavailable pseudocolor falls back without enabling its control', async () => {
  const c = await client({}, null, '?autoplay=1&mode=pseudocolor');
  assert.equal(c.stream().render_mode, 'gaussian');
  assert.equal(c.elements.pseudocolorOption.disabled, true);
});


test('four quadrants wait for their reference pair and present the same timestamp atomically', async () => {
  let resolve, requested, closed = 0;
  const reference = {available: true, width: 640, height: 640};
  const c = await client({reference}, null, '?autoplay=1', index => {
    requested = index;
    return new Promise(r => {resolve = r;});
  });
  const pending = c.packet(0, 7);
  await new Promise(r => setImmediate(r));
  assert.equal(requested, 7);
  assert.equal(c.elements.canvas.dataset.frame, undefined, 'top cannot get ahead of the lower videos');
  resolve({close() {closed++;}}); await pending;
  for (const id of ['canvas', 'comparisonCanvas', 'rgbCanvas', 'eventCanvas']) {
    assert.equal(c.elements[id].dataset.frame, '7');
  }
  assert.equal(c.elements.eventCanvas.dataset.sourceFrame, '6', 'events end at the RGB frame');
  assert.equal(c.elements.rgbCell.hidden, false);
  assert.equal(c.elements.eventCell.hidden, false);
  c.elements.play.onclick();
  assert.equal(closed, 1, 'stopping closes the reference bitmap owned by the jitter buffer');
});

test('late reference decode is discarded after a stream is replaced', async () => {
  let resolve, closed = 0;
  const c = await client({reference: {available: true, width: 640, height: 640}}, null,
    '?autoplay=1', () => new Promise(r => {resolve = r;}));
  const pending = c.packet(0, 8);
  await new Promise(r => setImmediate(r));
  c.presets[1].onclick(); c.tick(16);
  resolve({close() {closed++;}}); await pending;
  assert.equal(closed, 1);
  assert.equal(c.elements.rgbCanvas.dataset.frame, undefined);
  assert.equal(c.elements.canvas.dataset.frame, undefined);
});

test('failed reference loading never displays an apparently synchronized top-only update', async () => {
  const c = await client({reference: {available: true, width: 640, height: 640}}, null,
    '?autoplay=1', async () => {throw new Error('Missing reference');});
  await c.packet(0, 2);
  assert.equal(c.elements.canvas.dataset.frame, undefined);
  assert.equal(c.elements.rgbCanvas.dataset.frame, undefined);
  assert.equal(c.elements.overlay.hidden, false);
  assert.equal(c.errors.length, 1);
});


test('cancelling first-play preparation cannot later restart animation', async () => {
  let resolve;
  const reference = async () => ({close() {}});
  reference.prepare = () => new Promise(r => {resolve = r;});
  const c = await client({reference: {available: true, width: 640, height: 640}}, null, '?autoplay=1', reference);
  assert.equal(c.stream(), undefined, 'playback waits for references instead of contending with their downloads');
  assert.equal(c.elements.play.textContent, '取消准备');
  c.elements.play.onclick(); resolve(); await new Promise(r => setImmediate(r)); c.tick(20);
  assert.equal(c.stream(), undefined);
});

test('multiple ROIs resize and move in shared image coordinates without interrupting playback', async () => {
  const c=await client();
  for(const id of ['canvas','comparisonCanvas']) c.elements[id].getBoundingClientRect=()=>({left:0,top:0,width:800,height:500});
  await c.packet(0,0);
  const sent=c.socket.sent.length,streamId=c.stream().id;
  c.elements.detailToggle.onclick();c.elements.detailAdd.onclick();
  const regions=()=>JSON.parse(c.elements.detailLeft.dataset.regions);
  assert.deepEqual(regions().map(r=>r.label),['A','B']);
  assert.notEqual(regions()[0].color,regions()[1].color);
  assert.equal(c.elements.detailRow.children.length,4);
  c.elements.detailSelect.value='0';c.elements.detailSelect.onchange();
  const target=c.elements.canvas;target.closest=()=>null;
  // Source is letterboxed: x=150 + normalized x*500.
  const before=regions()[0];
  c.elements.stage.onpointerdown({target,pointerId:1,clientX:150+(before.x+before.w)*500,clientY:(before.y+before.h)*500});
  c.elements.stage.onpointermove({target,pointerId:1,clientX:150+(before.x+before.w)*500+50,clientY:(before.y+before.h)*500+25});
  c.elements.stage.onpointerup({pointerId:1});
  let r=regions()[0];assert.ok(Math.abs(r.w-.3)<1e-9);assert.ok(Math.abs(r.h-.25)<1e-9);
  assert.equal(c.elements.detailLeft.dataset.regions,c.elements.detailRight.dataset.regions);
  // Drag interior beyond image bounds clamps the whole ROI inside the image.
  c.elements.stage.onpointerdown({target,pointerId:2,clientX:150+(r.x+r.w/2)*500,clientY:(r.y+r.h/2)*500});
  c.elements.stage.onpointermove({target,pointerId:2,clientX:1000,clientY:900});
  c.elements.stage.onpointerup({pointerId:2});
  r=regions()[0];assert.ok(r.x+r.w<=1);assert.ok(r.y+r.h<=1);
  assert.equal(c.socket.sent.length,sent);assert.equal(c.stream().id,streamId);
  for(let i=1;i<20;i++) await c.packet(i,i);
  c.tick(100);c.tick(100+1000/60);
  assert.equal(c.elements.detailLeft.dataset.frame,c.elements.canvas.dataset.frame);
  for(const cell of c.elements.detailRow.children) assert.equal(cell.children[1].children[0].dataset.frame,c.elements.canvas.dataset.frame);
  c.elements.detailRemove.onclick();assert.deepEqual(regions().map(r=>r.label),['B']);
  c.elements.detailAdd.onclick();assert.deepEqual(regions().map(r=>r.label),['B','C']);
  c.elements.detailClear.onclick();assert.equal(c.elements.detailRow.hidden,true);
  c.elements.detailToggle.onclick();assert.equal(c.elements.detailLeft.hidden,true);
});


test('overview fits a shared content envelope and does not shrink on later animation frames', async () => {
  const c=await client({default_background:'light',reference:{available:true,width:640,height:640}});
  let first=30,last=90;
  const create=c.context.document.createElement;
  c.context.document.createElement=()=>{
    const el=create();el.getContext=()=>({clearRect(){},drawImage(){},getImageData(){
      const data=new Uint8Array(128*128*4).fill(255);
      for(let y=first;y<=last;y++) for(let x=40;x<80;x++) data[(y*128+x)*4]=80;
      return {data};
    }});return el;
  };
  await c.packet(0,0);
  const top=c.elements.canvas.style.top,height=c.elements.stage.style['--overview-height'];
  assert.ok(parseFloat(top)<0);
  for(const id of ['canvas','comparisonCanvas','rgbCanvas','eventCanvas']) assert.equal(c.elements[id].style.top,top);
  first=45;last=70;
  for(let i=1;i<20;i++) await c.packet(i,i);
  c.tick(100);c.tick(100+1000/60);
  assert.equal(c.elements.canvas.style.top,top,'narrower motion does not make the row breathe');
  assert.equal(c.elements.stage.style['--overview-height'],height);
});

test('complete local playback needs no stream or ACK packets and camera changes invalidate its cache',async()=>{
 const c=await client({frame_count:4,prewarmed_frames:4,times_seconds:[0,1/60,2/60,3/60],rpm:[null,null,null,null]},null,'');
 c.elements.smoothPlayback.checked=true;c.elements.streamRate.value='60';c.elements.loop.checked=true;c.elements.play.onclick();
 await new Promise(r=>setImmediate(r));
 const requests=c.socket.sent.slice(-4);assert.deepEqual(requests.map(r=>r.frame),[0,1,2,3]);
 for(const req of requests){
  const header=Buffer.from(JSON.stringify({id:req.id,frame:req.frame,width:1280,height:640,panel_width:640,comparison:true,right_frame:req.frame,render_ms:0}));
  const data=new ArrayBuffer(4+header.length+4);new DataView(data).setUint32(0,header.length,true);new Uint8Array(data,4,header.length).set(header);await c.socket.onmessage({data});
 }
 await new Promise(r=>setImmediate(r));c.tick(100);await new Promise(r=>setImmediate(r));
 const sent=c.socket.sent.length;c.tick(116);c.tick(133);c.tick(150);
 assert.equal(c.context.gaussianViewerStats.local,true);assert.ok(c.context.gaussianViewerStats.presented>0);
 assert.equal(c.socket.sent.length,sent);assert.equal(c.stream(),undefined,'local playback never opens a server stream');
 c.presets[1].onclick();c.tick(166);await new Promise(r=>setImmediate(r));
 assert.equal(c.socket.sent.at(-1).yaw,35,'a new view requests its own complete sequence');
});

test('cancelling whole render preparation cannot restart playback when in-flight frames arrive',async()=>{
 const c=await client({frame_count:4,prewarmed_frames:4,times_seconds:[0,1/60,2/60,3/60],rpm:[null,null,null,null]},null,'');
 c.elements.smoothPlayback.checked=true;c.elements.play.onclick();await new Promise(r=>setImmediate(r));
 const requests=c.socket.sent.slice(-4);c.elements.play.onclick();
 for(const req of requests){
  const header=Buffer.from(JSON.stringify({id:req.id,frame:req.frame,width:1280,height:640,panel_width:640,comparison:true,right_frame:req.frame,render_ms:0}));
  const data=new ArrayBuffer(4+header.length+4);new DataView(data).setUint32(0,header.length,true);new Uint8Array(data,4,header.length).set(header);await c.socket.onmessage({data});
 }
 await new Promise(r=>setImmediate(r));c.tick(100);
 assert.equal(c.context.gaussianViewerStats,undefined);assert.equal(c.stream(),undefined);
});
