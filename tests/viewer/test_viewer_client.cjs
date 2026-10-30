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
  const experimentPanel = element();
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
      querySelector(selector) {return selector === '.experiment-panel' ? experimentPanel : null;},
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
    const mode = stream().render_mode, panels = (scene.mode_version_labels?.[mode] || scene.version_labels || [0,1]).length;
    const header = Buffer.from(JSON.stringify({id, frame, stream: {seq}, width: 640*panels, panel_count:panels, render_mode:mode,
      height: 640, panel_width: 640, comparison: true, left_frame: frame, right_frame: frame,
      render_ms: 0, encode_ms: 0, cache_hit: true, client_cache_hit: reference}));
    const data = new ArrayBuffer(4 + header.length + (reference ? 0 : 4));
    new DataView(data).setUint32(0, header.length, true);
    new Uint8Array(data, 4, header.length).set(header);
    await socket.onmessage({data});
  }
  return {elements, presets, socket, stream, tick, packet, decoded, resize, errors, context};
}

test('V1 V2 V3 stay synchronized through cached playback, ROI and a two-version mode switch', async () => {
  const c = await client({version_labels:['V1','V2 · 存档分支','V3'],
    mode_version_labels:{mesh:['V1','V3']}, reference:{available:true},
    mesh:{available:true,triangle_count:20,description:'Mesh',left_label:'V1',right_label:'V3'}});
  assert.equal(c.elements.middleRenderCell.hidden,false);
  assert.equal(c.elements.middleCompareLabel.textContent,'V2 · 存档分支');
  assert.equal(c.elements.snapshot.textContent,'保存 1×5 对比图 ↓');
  for(let i=0;i<20;i++)await c.packet(i,i);
  c.tick(100);c.tick(140);
  for(const id of ['middleCanvas','comparisonCanvas','rgbCanvas','eventCanvas'])
    assert.equal(c.elements[id].dataset.frame,c.elements.canvas.dataset.frame);
  c.elements.detailToggle.onclick();c.elements.detailAdd.onclick();
  assert.equal(c.elements.detailRow.children.length,6);
  assert.equal(c.elements.detailMiddle.dataset.regions,c.elements.detailLeft.dataset.regions);
  const decoded=c.decoded.length;
  await c.packet(20,0,true);
  assert.equal(c.decoded.length,decoded+1);
  assert.equal(c.decoded[decoded],c.decoded[0]);
  c.elements.renderMode.onchange({target:{value:'mesh'}});c.tick(160);
  assert.equal(c.elements.middleRenderCell.hidden,true);
  assert.equal(c.elements.detailMiddle.hidden,true);
  assert.equal(c.elements.snapshot.textContent,'保存 1×4 对比图 ↓');
  assert.equal(c.elements.detailRow.children.length,4);
  c.elements.renderMode.onchange({target:{value:'gaussian'}});c.tick(180);
  assert.equal(c.elements.middleRenderCell.hidden,false);
  assert.equal(c.elements.detailRow.children.length,6);
  assert.equal(c.errors.length,0);
});

test('renumbered triplet labels and two-version mesh controls follow the supplied mapping', async () => {
  const c=await client({version_presentation:{revision:'swap-v1-v2-20261027'},
    version_labels:['V1 · 观测校正存档','V2 · 未校正初始动画','V3 · 修复补全'],
    left_label:'V1 · 观测校正存档',right_label:'V3 · 修复补全',
    mode_version_labels:{mesh:['V2 · 原始网格','V3 · 修复网格']},
    mesh:{available:true,triangle_count:20,left_label:'V2 · 原始网格',right_label:'V3 · 修复网格'}});
  assert.equal(c.elements.leftCompareLabel.textContent,'V1 · 观测校正存档');
  assert.equal(c.elements.middleCompareLabel.textContent,'V2 · 未校正初始动画');
  assert.equal(c.elements.meshOption.textContent,'V2 / V3 纯 Mesh');
  c.elements.renderMode.onchange({target:{value:'mesh'}});c.tick(160);
  assert.equal(c.elements.leftCompareLabel.textContent,'V2 · 原始网格');
  assert.equal(c.elements.middleRenderCell.hidden,true);
  assert.equal(c.elements.sceneName.textContent,'V2 / V3 Mesh 对比');
});

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
  assert.ok(c.socket.url.endsWith('/ws?scene=fan&pipeline=live'));
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

async function answerStill(c,req) {
 const header=Buffer.from(JSON.stringify({id:req.id,frame:req.frame,width:1280,height:640,panel_width:640,comparison:true,right_frame:req.frame,render_ms:0}));
 const data=new ArrayBuffer(4+header.length+4);new DataView(data).setUint32(0,header.length,true);new Uint8Array(data,4,header.length).set(header);await c.socket.onmessage({data});
}
function memoryViewStore(c,failWrite=()=>false) {
 const records=new Map();
 c.context.GaussianPlayback.Media.ExportFrameStore.open=async()=>({
  async save(key,entry){if(failWrite())throw new Error('disk full');for(const [frame,record] of entry.frames)records.set(`${key}:${frame}`,record);},
  async get(key,frame){return records.get(`${key}:${frame}`);},close(){}
 });return records;
}
const flush=()=>new Promise(r=>setImmediate(r));

test('all six complete views gate presentation and stay locally playable after RAM eviction and disconnect',async()=>{
 const c=await client({frame_count:4,prewarmed_frames:4,times_seconds:[0,1/60,2/60,3/60],rpm:[null,null,null,null]},null,'');
 const records=memoryViewStore(c);c.elements.smoothPlayback.checked=true;
 // Force a one-view RAM budget: completed disk views must survive eviction.
 c.context.GaussianPlayback.Media.SequenceCache.prototype.put=function(entry,index,record){
  for(const [key,e] of this.entries)if(e!==entry)this.entries.delete(key);
  entry.frames.set(index,record);entry.complete=entry.frames.size===entry.count;
 };
 let offset=c.socket.sent.length;c.tick(16);await flush();
 assert.equal(c.elements.loadTitle.textContent,'渲染中');
 for(let view=0;view<6;view++) {
  const requests=c.socket.sent.slice(offset).filter(m=>m.frame!==undefined);offset=c.socket.sent.length;
  assert.equal(requests.length,4);assert.equal(c.elements.canvas.dataset.frame,undefined,'no partial group is presented');
  for(const req of requests)await answerStill(c,req);
  await flush();
 }
 assert.equal(records.size,24);assert.equal(c.elements.loadProgress.value,100);
 c.tick(32);await flush();assert.equal(c.elements.canvas.dataset.frame,'0');assert.equal(c.elements.overlay.hidden,true);
 c.socket.readyState=3;c.socket.onclose();
 c.elements.streamRate.value='60';c.elements.loop.checked=true;c.elements.play.onclick();
 const sent=c.socket.sent.length;
 for(let view=0;view<6;view++) {
  c.presets[view].onclick();c.tick(100+view*100);await flush();c.tick(116+view*100);await flush();
  assert.equal(c.context.gaussianViewerStats.local,true);assert.ok(c.context.gaussianViewerStats.presented>0);
 }
 assert.equal(c.socket.sent.length,sent);assert.equal(c.stream(),undefined);
 // A quality change must prepare a new group, not reuse mismatched images.
 c.socket.readyState=1;c.elements.resolution.value='512';c.elements.resolution.onchange();c.tick(800);await flush();
 assert.equal(c.elements.loadTitle.textContent,'渲染中');assert.equal(c.socket.sent.at(-1).width,512);
});

test('failed disk writes never publish a ready group and retry reuses downloaded frames',async()=>{
 const c=await client({frame_count:4,prewarmed_frames:4,times_seconds:[0,1/60,2/60,3/60],rpm:[null,null,null,null]},null,'');
 let fail=true;const records=memoryViewStore(c,()=>fail);c.elements.smoothPlayback.checked=true;
 c.tick(16);await flush();for(const req of c.socket.sent.slice(-4))await answerStill(c,req);await flush();
 assert.equal(c.elements.loadTitle.textContent,'渲染未完成');assert.equal(c.elements.canvas.dataset.frame,undefined);
 assert.equal(c.elements.retryViews.hidden,false);assert.equal(records.size,0);
 await answerStill(c,c.socket.sent.at(-1));
 assert.equal(c.elements.canvas.dataset.frame,undefined,'late preparation replies cannot reveal an incomplete group');
 assert.equal(c.elements.loadTitle.textContent,'渲染未完成');
 fail=false;const sent=c.socket.sent.length;c.elements.retryViews.onclick();c.tick(32);await flush();
 assert.equal(records.size,4,'retry commits the retained front sequence');
 assert.equal(c.socket.sent.slice(sent).filter(m=>m.frame!==undefined).length,4,'only the next view is requested');
 assert.equal(c.socket.sent.at(-1).yaw,35);
});

test('each display mode prepares six isolated views and returning to a completed mode reuses disk',async()=>{
 const mode={available:true,default_resolution:640,triangle_count:100,description:''};
 const c=await client({frame_count:2,prewarmed_frames:2,times_seconds:[0,1/60],rpm:[null,null],
  mesh:mode,clay:mode,pseudocolor:mode,hybrid:mode},null,'');
 const records=memoryViewStore(c);c.elements.smoothPlayback.checked=true;
 let now=10;
 for(const mode of ['gaussian','mesh','clay','pseudocolor','hybrid']) {
  if(mode!=='gaussian')c.elements.renderMode.onchange({target:{value:mode}});
  let offset=c.socket.sent.length;c.tick(now+=10);await flush();
  for(let view=0;view<6;view++) {
   const requests=c.socket.sent.slice(offset).filter(m=>m.frame!==undefined);offset=c.socket.sent.length;
   assert.equal(requests.length,2);assert.ok(requests.every(r=>r.render_mode===mode));
   for(const request of requests)await answerStill(c,request);await flush();
  }
  assert.equal(c.elements.loadProgress.value,100);c.tick(now+=10);await flush();
  assert.equal(c.elements.overlay.hidden,true);
 }
 assert.equal(records.size,60);
 const sent=c.socket.sent.length;c.elements.renderMode.onchange({target:{value:'gaussian'}});
 c.tick(now+=10);await flush();c.tick(now+=10);await flush();
 assert.equal(c.socket.sent.length,sent);assert.equal(c.elements.overlay.hidden,true);
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

test('ROI sets are independent per camera view and shared across render modes of that view',async()=>{
 const c=await client({mesh:{available:true,triangle_count:100,default_resolution:640}});
 await c.packet(0,0);c.elements.detailToggle.onclick();c.elements.detailAdd.onclick();
 const regions=()=>JSON.parse(c.elements.detailLeft.dataset.regions);
 const front=regions();assert.equal(front.length,2);
 c.presets[2].onclick();c.tick(16);await c.packet(0,0);
 assert.deepEqual(regions(),[],'side view starts with its own empty selection set');
 c.elements.detailAdd.onclick();const side=regions();assert.equal(side[0].label,'A');
 c.presets[0].onclick();c.tick(32);await c.packet(0,0);assert.deepEqual(regions(),front);
 c.elements.renderMode.onchange({target:{value:'mesh'}});c.tick(48);await c.packet(0,0);
 assert.deepEqual(regions(),front,'a mode change keeps the current camera selections');
 c.elements.detailAdd.onclick();const frontMesh=regions();assert.equal(frontMesh[2].label,'C');
 c.elements.renderMode.onchange({target:{value:'gaussian'}});c.tick(64);await c.packet(0,0);
 assert.deepEqual(regions(),frontMesh,'edits in Mesh also appear in Gaussian for this view');
 c.presets[2].onclick();c.tick(80);await c.packet(0,0);assert.deepEqual(regions(),side);
 c.elements.detailClear.onclick();assert.deepEqual(regions(),[]);
 c.presets[0].onclick();c.tick(96);await c.packet(0,0);assert.deepEqual(regions(),frontMesh,'clearing side does not clear front');
 assert.equal(c.elements.detailSelect.value,'2','selected ROI is restored with the view');
});

test('V1/V2 scene keeps the frozen stage labels in clay title and controls', async () => {
  const c = await client({left_label:'V1 · 原始高斯',right_label:'V2 · 观测校正',
    clay:{available:true,left_label:'V1 · 白模',right_label:'V2 · 白模',default_resolution:640}},null,'?mode=clay');
  assert.equal(c.elements.sceneName.textContent,'V1 / V2 白模对比');
  assert.equal(c.elements.rightCompareLabel.textContent,'V2 · 白模');
  assert.equal(c.elements.clayOption.textContent,'V1 / V2 白模 · 清晰柔光');
  assert.equal(c.errors.length,0);
});

test('evaluation link applies the same 640px and JPEG 95 settings as measured UI frames', async () => {
  const c=await client({default_stream_quality:78,clay:{available:true,default_resolution:1024}},null,'?mode=clay&autoplay=1&resolution=640&jpeg_quality=95');
  assert.equal(c.elements.resolution.value,'640');
  assert.equal(c.stream().width,640);
  assert.equal(c.stream().height,640);
  assert.equal(c.stream().quality,95);
  assert.equal(c.stream().render_mode,'clay');
});

test('experiment sequence reuses saved camera/time samples after the live view changes', async () => {
  const c=await client();
  const samples=[{frame:0,time_seconds:0,width:640,height:640,yaw:0,pitch:0,
    camera:{pose_opengl_c2w:[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]}}];
  const plan={format:'scgs_experiment_plan_v1',scene:c.elements.objectSelect.value,samples};
  c.elements.experimentPlanFile.files=[{size:100,text:async()=>JSON.stringify(plan)}];
  await c.elements.experimentPlanFile.onchange();
  assert.equal(c.elements.experimentSequence.disabled,false);
  c.presets[1].onclick();c.tick(20);
  c.elements.experimentWarmup.value='3';c.elements.experimentRepeats.value='2';
  let submitted;
  c.context.fetch=async(url,options)=>{submitted={url,data:JSON.parse(options.body)};
    return {ok:true,json:async()=>({id:'test-job',status:'complete'})};};
  await c.elements.experimentSequence.onclick();
  assert.equal(submitted.data.operation,'sequence');
  assert.deepEqual(submitted.data.samples,samples);
  assert.equal(c.elements.experimentDownload.hidden,false);
  assert.match(c.elements.experimentDownload.href,/test-job\/download/);
});

test('experiment benchmark forwards explicit warmup/repeats and errors do not expose a download', async () => {
  const c=await client();
  const plan={format:'scgs_experiment_plan_v1',scene:c.elements.objectSelect.value,
    samples:[{frame:0,time_seconds:0,width:640,height:640}]};
  c.elements.experimentPlanFile.files=[{size:100,text:async()=>JSON.stringify(plan)}];
  await c.elements.experimentPlanFile.onchange();
  c.elements.experimentWarmup.value='7';c.elements.experimentRepeats.value='4';
  let submitted;
  c.context.fetch=async(url,options)=>{submitted=JSON.parse(options.body);return {ok:false,text:async()=> '已有任务'};};
  await c.elements.experimentBenchmark.onclick();
  assert.equal(submitted.operation,'benchmark');assert.equal(submitted.warmup,7);assert.equal(submitted.repeats,4);
  assert.equal(c.elements.experimentStatus.textContent,'已有任务');
  assert.equal(c.elements.experimentDownload.hidden,true);
  assert.equal(c.elements.experimentBenchmark.disabled,false);
});

test('one-click save needs no saved plan, preserves camera and downloads the single folder archive', async () => {
  const c=await client();
  assert.equal(c.elements.experimentStep.value,'2');
  c.presets[1].onclick();c.tick(20);
  c.elements.experimentStart.value='invalid';c.elements.experimentEnd.value='invalid';
  c.elements.experimentStep.value='2';
  let submitted,downloads=0;
  c.elements.experimentDownload.click=()=>downloads++;
  c.context.fetch=async(url,options)=>{submitted=JSON.parse(options.body);
    return {ok:true,json:async()=>({id:'one-click',status:'complete',
      output_archive:'/exports/scene_20261026.zip',output_bytes:1048576,sampled_frames:5,sample_step:4,missing_data:['原始事件计数：0/4']})};};
  await c.elements.experimentAll.onclick();
  assert.equal(submitted.operation,'complete');
  assert.equal(submitted.settings.yaw,35);assert.equal(submitted.settings.pitch,20);
  assert.equal(submitted.warmup,20);assert.equal(submitted.repeats,5);
  assert.equal(submitted.samples,undefined,'server selects the complete source timeline');
  assert.equal(submitted.sample_step,2);
  assert.equal(downloads,1);assert.equal(c.elements.experimentLocation.hidden,false);
  assert.match(c.elements.experimentLocation.textContent,/\/exports\/scene_20261026/);
  assert.match(c.elements.experimentStatus.textContent,/data_availability.json/);
  assert.match(c.elements.experimentStatus.textContent,/python evaluate.py/);
  assert.match(c.elements.experimentStatus.textContent,/每 4 帧采样，共 5 帧/);
  assert.match(c.elements.experimentLocation.textContent,/仅保留 ZIP/);
});

test('sequence and benchmark automatically freeze a truth plan without importing a file', async () => {
  for(const id of ['experimentSequence','experimentBenchmark']) {
    const c=await client(),requests=[];
    assert.equal(c.elements[id].disabled,false);
    const samples=[{frame:0,time_seconds:0,camera:{fixture:'server truth'},background_rgb:[.27,.27,.27]}];
    c.context.fetch=async(url,options)=>{
      requests.push({url,body:JSON.parse(options.body)});
      return {ok:true,json:async()=>url.includes('/plan') ?
        {format:'scgs_experiment_plan_v1',scene:c.elements.objectSelect.value,samples} :
        {id:'automatic',status:'complete'}};
    };
    await c.elements[id].onclick();
    assert.equal(requests.length,2);
    assert.equal(requests[0].body.operation,'complete');
    assert.equal(requests[0].body.sample_step,2);
    assert.deepEqual(requests[1].body.samples,samples);
    assert.equal(requests[1].body.operation,id==='experimentSequence' ? 'sequence' : 'benchmark');
    assert.equal(c.elements.experimentDownload.hidden,false);
  }
});

test('one-click progress follows server stages and bytes until the archive is complete', async () => {
  const c=await client(), waits=[];
  c.context.setTimeout=callback=>waits.push(callback);
  let downloads=0;
  c.elements.experimentDownload.click=()=>downloads++;
  const jobs=[
    {stage:'warmup',message:'预热渲染器',stage_index:2,stage_count:10,percent:12,completed:2,total:10,unit:'renders'},
    {stage:'frames',message:'导出无损帧与模型状态',stage_index:4,stage_count:10,percent:35,completed:10,total:20,unit:'frames'},
    {stage:'archive',message:'打包下载文件',stage_index:10,stage_count:10,percent:95,completed:1048576,total:2097152,unit:'bytes',detail:'sources/model.npz'},
    {status:'complete',stage:'complete',message:'导出完成',percent:100,completed:1,total:1},
  ];
  c.context.fetch=async()=>({ok:true,json:async()=>({id:'progress',status:'running',...jobs.shift()})});
  const saving=c.elements.experimentAll.onclick();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(c.elements.experimentProgress.hidden,false);
  assert.equal(c.elements.experimentProgressBar.value,12);
  assert.match(c.elements.experimentStage.textContent,/阶段 2\/10 · 预热/);
  assert.equal(c.elements.experimentDetail.textContent,'2 / 10 次');
  assert.equal(c.elements.experimentAll.disabled,true);
  waits.shift()();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(c.elements.experimentProgressBar.value,35);
  assert.equal(c.elements.experimentDetail.textContent,'10 / 20 帧');
  waits.shift()();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(c.elements.experimentPercent.textContent,'95%');
  assert.match(c.elements.experimentDetail.textContent,/1.0 MiB \/ 2.0 MiB · sources\/model.npz/);
  assert.equal(downloads,0,'no download before the archive closes');
  waits.shift()();await saving;
  assert.equal(c.elements.experimentProgressBar.value,100);
  assert.equal(c.elements.experimentProgress.dataset.state,'complete');
  assert.match(c.elements.experimentDetail.textContent,/浏览器下载进度/);
  assert.equal(downloads,1);
  assert.equal(c.elements.experimentAll.disabled,false);
});

test('failed exports retain their progress and retry starts at zero', async () => {
  const c=await client();
  c.context.fetch=async()=>({ok:true,json:async()=>({id:'failed',status:'failed',stage:'sources',
    message:'导出失败：磁盘空间不足',percent:64,completed:4,total:10,unit:'files'})});
  await c.elements.experimentAll.onclick();
  assert.equal(c.elements.experimentProgressBar.value,64);
  assert.equal(c.elements.experimentProgress.dataset.state,'failed');
  assert.match(c.elements.experimentStage.textContent,/磁盘空间不足/);
  assert.equal(c.elements.experimentDownload.hidden,true);
  c.context.fetch=async()=>{
    assert.equal(c.elements.experimentProgressBar.value,0);
    assert.equal(c.elements.experimentProgress.dataset.state,'running');
    return {ok:false,text:async()=> '已有导出正在运行'};
  };
  await c.elements.experimentAll.onclick();
  assert.equal(c.elements.experimentAll.disabled,false);
  assert.match(c.elements.experimentStage.textContent,/已有导出正在运行/);
});

test('long experiment keeps the displayed frame, pauses preview requests and schedules reconnect after completion', async () => {
  const c=await client(),waits=[];
  c.elements.overlay.hidden=true;
  c.context.setTimeout=callback=>waits.push(callback);
  let finish;
  c.context.fetch=()=>new Promise(resolve=>{finish=resolve;});
  c.elements.experimentDownload.click=()=>{};
  const saving=c.elements.experimentAll.onclick();
  const sent=c.socket.sent.length;
  c.tick(20);c.tick(40);
  assert.equal(c.socket.sent.length,sent,'no preview queued behind a long export');
  c.socket.readyState=3;c.socket.onclose();
  assert.equal(c.elements.overlay.hidden,true,'retain the last rendered image');
  assert.equal(waits.length,0,'wait until the export releases the renderer');
  finish({ok:true,json:async()=>({id:'slow',status:'complete'})});
  await saving;
  assert.equal(waits.length,1,'reconnect after the export completes');
  assert.equal(c.elements.experimentDownload.hidden,false);
});

test('comparison link carries the editable object name without injecting query parameters', async()=>{
  const c=await client({},null,'?scene=lego');
  assert.equal(c.elements.comparisonObjectName.value,'lego');
  assert.equal(c.elements.comparisonVisualize.href,'/comparison?object_name=lego');
  c.elements.comparisonObjectName.value='吊机 A';c.elements.comparisonObjectName.oninput();
  assert.equal(new URL(c.elements.comparisonVisualize.href,'http://localhost').searchParams.get('object_name'),'吊机 A');
  c.elements.comparisonObjectName.value='a&scene=other';c.elements.comparisonVisualize.onclick();
  assert.equal(new URL(c.elements.comparisonVisualize.href,'http://localhost').searchParams.has('scene'),false);
});

test('free view starts directly and mode links retain the selected object', async()=>{
 const c=await client({},null,'?scene=lego');
 assert.equal(c.elements.smoothPlayback.checked,false);
 assert.ok(c.socket.url.endsWith('/ws?scene=lego&pipeline=live'));
 assert.ok(c.socket.sent.length>0);
 assert.ok(c.socket.sent.every(r=>r.yaw===0),'no six-view preparation');
 const q=new URLSearchParams(c.elements.observationViewLink.href.slice(1));
 assert.equal(q.get('scene'),'lego');assert.equal(q.get('pipeline'),'observations');
 c.elements.smoothPlayback.checked=true;c.elements.smoothPlayback.onchange();
 assert.equal(new URLSearchParams(c.context.location.search).get('pipeline'),'observations');
});

test('observation page uses its own socket and prevents arbitrary orbit', async()=>{
 const c=await client({},null,'?scene=lego&pipeline=observations');
 assert.equal(c.elements.smoothPlayback.checked,true);
 assert.ok(c.socket.url.endsWith('/ws?scene=lego&pipeline=observations'));
 assert.equal(c.elements.liveViewLink,undefined);
 assert.ok(c.elements.realtimeGaussian,'Hybrid entry remains');
 const before=c.socket.sent.length;
 c.elements.stage.onpointerdown({target:{closest:()=>null},pointerId:1,clientX:0,clientY:0});
 c.elements.stage.onpointermove({pointerId:1,clientX:100,clientY:40});
 c.tick(16);
 assert.equal(c.socket.sent.length,before);
});
