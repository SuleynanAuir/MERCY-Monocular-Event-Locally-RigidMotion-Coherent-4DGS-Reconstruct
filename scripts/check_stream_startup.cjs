/* Acceptance for restored short-buffer streaming, not a 60 FPS performance gate.
 * Isolated headless browser; no existing user profile is touched.
 * SCGS_PLAYWRIGHT_PATH=/path/to/playwright-core SCGS_CHROME_PATH=/path/to/chrome \
 * SCGS_VIEWER_URL=http://127.0.0.1:8892 \
 * SCGS_STREAM_STARTUP_OUTPUT=outputs/viewer-stream-rollback node scripts/check_stream_startup.cjs
 */
'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.SCGS_PLAYWRIGHT_PATH || 'playwright-core');
const STARTUP_TIMEOUT_MS = 30000, OBSERVE_MS = 8000;

function instrument() {
  const probe = window.streamStartupProbe = {armed: false, click_at: null, initial_frame: 0,
    sent: [], received: [], presentations: [], first_moving: null, stats_samples: []};
  const NativeSocket = window.WebSocket;
  window.WebSocket = class extends NativeSocket {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', ({data}) => {
        if (!(data instanceof ArrayBuffer) || data.byteLength < 4) return;
        const length = new DataView(data).getUint32(0, true);
        if (length > 65536 || length + 4 > data.byteLength) return;
        try {
          const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, length)));
          if (meta.stream) probe.received.push({at: performance.now(), id: meta.id, frame: meta.frame,
            seq: meta.stream.seq, bytes: data.byteLength, cache_hit: !!meta.cache_hit,
            client_cache_hit: !!meta.client_cache_hit});
        } catch (_) { /* The application remains responsible for protocol errors. */ }
      });
    }
    send(data) {
      if (typeof data === 'string') {
        try {probe.sent.push({at: performance.now(), message: JSON.parse(data)});} catch (_) { /* Non-JSON. */ }
      }
      return super.send(data);
    }
  };
  document.addEventListener('click', event => {
    if (probe.armed && event.target.closest('#play') && probe.click_at === null) probe.click_at = performance.now();
  }, true);
  const draw = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function (...args) {
    const result = draw.apply(this, args);
    if (this.canvas.id === 'canvas') {
      const at = performance.now();
      queueMicrotask(() => {
        if (!probe.armed || probe.click_at === null) return;
        const left = document.querySelector('#canvas'), right = document.querySelector('#comparisonCanvas');
        if (!left.dataset.requestId?.includes(':')) return;
        const item = {at, left_frame: Number(left.dataset.frame), right_frame: Number(right.dataset.frame),
          request_id: left.dataset.requestId};
        probe.presentations.push(item);
        if (!probe.first_moving && item.left_frame !== probe.initial_frame) {
          probe.first_moving = {...item, click_to_first_moving_ms: at - probe.click_at,
            stream_packets_received: probe.received.filter(packet => packet.at >= probe.click_at).length,
            stream_packet_bytes: probe.received.filter(packet => packet.at >= probe.click_at)
              .reduce((sum, packet) => sum + packet.bytes, 0)};
        }
      });
    }
    return result;
  };
  setInterval(() => {
    if (probe.armed && probe.click_at !== null && window.gaussianViewerStats) {
      probe.stats_samples.push({at: performance.now(), stats: structuredClone(window.gaussianViewerStats)});
    }
  }, 250);
  probe.obstructions = () => {
    const canvases = ['canvas', 'comparisonCanvas'].map(id => document.getElementById(id))
      .filter(element => element && !element.hidden).map(element => ({id: element.id, rect: element.getBoundingClientRect()}));
    const area = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
      Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    const matches = [], walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode, parent = node.parentElement, value = node.textContent.trim();
      if (!value || !parent || parent.closest('script,style,[hidden],.sr-only')) continue;
      const style = getComputedStyle(parent);
      if (style.visibility === 'hidden' || style.display === 'none' || +style.opacity === 0) continue;
      const range = document.createRange(); range.selectNodeContents(node);
      for (const rect of range.getClientRects()) for (const canvas of canvases) if (area(rect, canvas.rect) > 1) {
        matches.push({canvas: canvas.id, element: parent.id || parent.className || parent.tagName,
          text: value.slice(0, 120), overlap_area_px: area(rect, canvas.rect)});
      }
    }
    const selected = ['#leftCompareLabel', '#rightCompareLabel', '.view-toolbar', '.viewport-hint'];
    const labels_and_toolbar = selected.map(selector => {
      const element = document.querySelector(selector), rect = element?.getBoundingClientRect();
      return {selector, rect: rect ? {x: rect.x, y: rect.y, width: rect.width, height: rect.height} : null,
        overlaps: rect ? canvases.filter(canvas => area(rect, canvas.rect) > 1).map(canvas => canvas.id) : []};
    });
    return {text_matches: matches, labels_and_toolbar};
  };
}

function summarize(probe) {
  const frames = probe.presentations.filter(frame => frame.at >= probe.first_moving.at);
  const gaps = frames.slice(1).map((frame, i) => frame.at - frames[i].at), sorted = [...gaps].sort((a, b) => a - b);
  const percentile = fraction => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] : null;
  const streamRequests = probe.sent.filter(row => row.message.type === 'stream');
  const statsByStream = [];
  for (let i = 0; i < streamRequests.length; i++) {
    const start = streamRequests[i].at, end = streamRequests[i + 1]?.at ?? Infinity;
    const samples = probe.stats_samples.filter(sample => sample.at >= start && sample.at < end);
    statsByStream.push({id: streamRequests[i].message.id, requested_rate: streamRequests[i].message.rate,
      maximum_sampled_underruns: Math.max(0, ...samples.map(sample => sample.stats.underruns || 0)),
      last_sample: samples.at(-1)?.stats || null});
  }
  return {first_moving: probe.first_moving, observation_ms: OBSERVE_MS, presentation_count: frames.length,
    actual_canvas_fps: frames.length > 1 ? (frames.length - 1) * 1000 / (frames.at(-1).at - frames[0].at) : 0,
    frame_gap_ms: {p50: percentile(.5), p95: percentile(.95), max: sorted.at(-1) ?? null},
    gaps_over_100ms: gaps.filter(gap => gap > 100).length,
    pair_mismatches: frames.filter(frame => !Number.isInteger(frame.left_frame) || frame.left_frame !== frame.right_frame),
    stream_requests: streamRequests, stream_restarts: Math.max(0, streamRequests.length - 1),
    sampled_stats_by_stream: statsByStream, final_stats: probe.final_stats,
    prepare_requests: probe.sent.filter(row => row.message.type === 'prepare'),
    scope: 'Actual browser canvas draws. Startup is measured from the real Play click to the first changed streamed frame. FPS and underruns are observations, not a 60 FPS pass threshold.'};
}

(async () => {
  const output = path.resolve(process.env.SCGS_STREAM_STARTUP_OUTPUT || 'outputs/viewer-stream-rollback');
  await fs.mkdir(output, {recursive: true});
  const url = new URL(process.env.SCGS_VIEWER_URL || 'http://127.0.0.1:8892');
  url.searchParams.set('autoplay', '0'); url.searchParams.set('loop', '1');
  const report = {format: 'gaussian_legacy_stream_startup_check_v1', passed: false, url: url.toString(),
    started_at: new Date().toISOString(), startup_timeout_ms: STARTUP_TIMEOUT_MS, views: [], errors: [], failures: []};
  let browser;
  try {
    browser = await chromium.launch({headless: true,
      ...(process.env.SCGS_CHROME_PATH ? {executablePath: process.env.SCGS_CHROME_PATH} : {})});
    for (const view of [{name: 'front', yaw: 0}, {name: 'back', yaw: 180}]) {
      const result = {view: view.name, requested_yaw: view.yaw}; report.views.push(result);
      const context = await browser.newContext({viewport: {width: 1440, height: 1000}, deviceScaleFactor: 1});
      const page = await context.newPage(); page.setDefaultTimeout(STARTUP_TIMEOUT_MS);
      page.on('pageerror', error => report.errors.push({view: view.name, message: error.message}));
      try {
        await page.addInitScript(instrument); await page.goto(url.toString());
        await page.waitForFunction(() => document.querySelector('#canvas')?.dataset.frame === '0' &&
          document.querySelector('#comparisonCanvas')?.dataset.frame === '0', null, {timeout: 120000});
        const scene = await page.evaluate(async () => (await (await fetch('/api/status')).json()).scene);
        result.scene = scene;
        if (!scene.comparison || scene.frame_count !== 400) throw new Error('Expected 400-frame original/latest comparison');
        if (Object.hasOwn(scene, 'camera_preparation')) report.failures.push(`${view.name}: camera_preparation remains advertised`);
        if (scene.default_resolution !== 640 || scene.default_stream_quality !== 95) {
          report.failures.push(`${view.name}: default 640 px / JPEG95 quality has changed`);
        }
        await page.locator('#loop').check(); await page.locator('#speed').selectOption('1');
        await page.locator('#streamRate').selectOption('auto');
        const previousRequest = await page.locator('#canvas').getAttribute('data-request-id');
        await page.locator(`.preset[data-yaw="${view.yaw}"][data-pitch="0"]`).click();
        await page.waitForFunction(({old, yaw}) => {
          const probe = window.streamStartupProbe, canvas = document.querySelector('#canvas');
          const request = [...probe.sent].reverse().find(row => !row.message.type && row.message.yaw === yaw)?.message;
          return request && canvas.dataset.requestId !== old && canvas.dataset.requestId === String(request.id) &&
            canvas.dataset.frame === '0';
        }, {old: previousRequest, yaw: view.yaw});
        result.before_play_obstructions = await page.evaluate(() => window.streamStartupProbe.obstructions());
        await page.screenshot({path: path.join(output, `${view.name}-before-play.png`), fullPage: true});
        await page.evaluate(() => {
          const probe = window.streamStartupProbe; probe.initial_frame = Number(document.querySelector('#canvas').dataset.frame);
          probe.armed = true;
        });
        console.log(JSON.stringify({stage: 'click_play', view: view.name}));
        await page.locator('#play').click();
        await page.waitForFunction(() => window.streamStartupProbe.first_moving !== null, null, {timeout: STARTUP_TIMEOUT_MS});
        result.first_motion_obstructions = await page.evaluate(() => window.streamStartupProbe.obstructions());
        console.log(JSON.stringify({stage: 'first_motion', view: view.name,
          ...(await page.evaluate(() => window.streamStartupProbe.first_moving))}));
        // Fixed observation window intentionally reports sustained streaming, including stalls.
        await page.waitForTimeout(OBSERVE_MS);
        const probe = await page.evaluate(() => {
          const probe = window.streamStartupProbe; probe.armed = false;
          return {sent: probe.sent, received: probe.received, presentations: probe.presentations,
            first_moving: probe.first_moving, stats_samples: probe.stats_samples,
            final_stats: window.gaussianViewerStats, obstructions: probe.obstructions()};
        });
        result.playback = summarize(probe); result.after_play_obstructions = probe.obstructions;
        await fs.writeFile(path.join(output, `${view.name}-samples.json`), JSON.stringify(probe, null, 2) + '\n');
        await page.screenshot({path: path.join(output, `${view.name}-playing.png`), fullPage: true});
        if (result.playback.first_moving.stream_packets_received < 1 ||
          result.playback.first_moving.stream_packets_received >= scene.frame_count) {
          report.failures.push(`${view.name}: expected 1–399 stream packets before first motion`);
        }
        if (result.playback.first_moving.click_to_first_moving_ms > STARTUP_TIMEOUT_MS) report.failures.push(`${view.name}: startup exceeds 30 seconds`);
        if (result.playback.prepare_requests.length) report.failures.push(`${view.name}: client still sends prepare`);
        if (!result.playback.stream_requests.length) report.failures.push(`${view.name}: no legacy stream request`);
        if (result.playback.stream_requests.some(({message}) => message.yaw !== view.yaw || message.pitch !== 0 ||
          message.width !== 640 || message.height !== 640 || message.quality !== 95)) {
          report.failures.push(`${view.name}: actual stream camera or image quality differs from requested comparison`);
        }
        if (result.playback.pair_mismatches.length) report.failures.push(`${view.name}: left/right frame mismatch`);
        if (new Set(probe.presentations.map(frame => frame.left_frame)).size < 2) {
          report.failures.push(`${view.name}: no continuing frame progression during observation`);
        }
        for (const check of [result.before_play_obstructions, result.first_motion_obstructions, result.after_play_obstructions]) {
          if (check.text_matches.length || check.labels_and_toolbar.some(row => row.overlaps.length)) {
            report.failures.push(`${view.name}: text, labels, or toolbar overlap actual canvas`); break;
          }
        }
      } catch (error) {
        result.error = error.stack || error.message; report.failures.push(`${view.name}: ${error.message}`);
        try {await page.screenshot({path: path.join(output, `${view.name}-failure.png`), fullPage: true});} catch (_) { /* Browser may be unavailable. */ }
      } finally {await context.close();}
      await fs.writeFile(path.join(output, 'stream-startup.json'), JSON.stringify(report, null, 2) + '\n');
    }
    if (report.errors.length) report.failures.push('Browser emitted uncaught page errors');
    report.passed = report.failures.length === 0;
  } catch (error) {report.failures.push(error.stack || error.message);}
  finally {
    report.finished_at = new Date().toISOString();
    await fs.writeFile(path.join(output, 'stream-startup.json'), JSON.stringify(report, null, 2) + '\n');
    if (browser) await browser.close();
  }
  console.log(JSON.stringify({passed: report.passed, report: path.join(output, 'stream-startup.json'), failures: report.failures}, null, 2));
  if (!report.passed) process.exitCode = 1;
})().catch(error => {console.error(error); process.exitCode = 1;});
