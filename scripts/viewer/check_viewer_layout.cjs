/* Layout and synchronized-presentation regression for the real comparison UI.
 * Uses an isolated headless browser, never an existing browser profile.
 * SCGS_PLAYWRIGHT_PATH=/path/to/playwright-core SCGS_CHROME_PATH=/path/to/chrome \
 * SCGS_VIEWER_LAYOUT_OUTPUT=outputs/layout-before node scripts/viewer/check_viewer_layout.cjs
 */
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.SCGS_PLAYWRIGHT_PATH || 'playwright-core');

const FRAMES = [0, 1, 2, 50, 99, 100, 101, 104, 196, 198, 200, 204, 296, 299, 300, 399];
const SNAPSHOTS = new Set([104, 204, 399]);
const SELECTORS = ['#stage', '#canvas', '#comparisonCanvas', '.transport', '.viewport', '.inspector', 'footer'];
const TOLERANCE_PX = 1;

// Runs before application code. All wrappers preserve the original behavior.
function instrument(selectors) {
  const state = window.viewerLayoutProbe = {active: false, samples: [], presentations: [], shifts: [], streams: []};
  state.sample = () => {
    const rects = {};
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (!node) continue;
      const box = node.getBoundingClientRect();
      rects[selector] = {x: box.x + scrollX, y: box.y + scrollY, width: box.width, height: box.height};
    }
    const left = document.querySelector('#canvas'), right = document.querySelector('#comparisonCanvas');
    return {at: performance.now(), rects, scroll_height: document.documentElement.scrollHeight,
      scroll_width: document.documentElement.scrollWidth, viewport_width: innerWidth, scroll_y: scrollY,
      left_frame: left?.dataset.frame ?? null, right_frame: right?.dataset.frame ?? null,
      left_size: left ? [left.width, left.height] : null, right_size: right ? [right.width, right.height] : null,
      interval_text: document.querySelector('#eventInterval')?.textContent ?? null};
  };
  const draw = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function (...args) {
    const result = draw.apply(this, args);
    if (this.canvas.id === 'canvas') queueMicrotask(() => {
      if (!state.active) return;
      state.presentations.push({at: performance.now(), left_frame: this.canvas.dataset.frame,
        right_frame: document.querySelector('#comparisonCanvas')?.dataset.frame,
        request_id: this.canvas.dataset.requestId});
    });
    return result;
  };
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function (data) {
    if (state.active && typeof data === 'string') {
      try {
        const message = JSON.parse(data);
        if (message.type === 'stream') state.streams.push({at: performance.now(), ...message});
      } catch (_) { /* Non-JSON traffic is not a viewer command. */ }
    }
    return send.call(this, data);
  };
  try {
    new PerformanceObserver(list => {
      if (!state.active) return;
      for (const entry of list.getEntries()) state.shifts.push({at: entry.startTime,
        value: entry.value, had_recent_input: entry.hadRecentInput,
        sources: (entry.sources || []).map(source => ({
          element: source.node?.id || source.node?.className || source.node?.nodeName || null,
          previous: source.previousRect.toJSON(), current: source.currentRect.toJSON()}))});
    }).observe({type: 'layout-shift'});
  } catch (_) { state.layout_shift_observer_unavailable = true; }
  const sample = () => {
    if (state.active) state.samples.push(state.sample());
    requestAnimationFrame(sample);
  };
  requestAnimationFrame(sample);
}

function summarizeGeometry(samples) {
  const dimensions = {};
  for (const sample of samples) {
    const values = {scroll_height: sample.scroll_height, scroll_width: sample.scroll_width};
    for (const [selector, rect] of Object.entries(sample.rects)) {
      for (const [name, value] of Object.entries(rect)) values[`${selector}.${name}`] = value;
    }
    for (const [name, value] of Object.entries(values)) {
      const range = dimensions[name] ||= {min: value, max: value};
      range.min = Math.min(range.min, value); range.max = Math.max(range.max, value);
    }
  }
  for (const range of Object.values(dimensions)) range.deviation_px = range.max - range.min;
  const violations = Object.entries(dimensions).filter(([, value]) => value.deviation_px > TOLERANCE_PX)
    .map(([dimension, range]) => ({dimension, ...range}));
  return {samples: samples.length, tolerance_px: TOLERANCE_PX,
    maximum_deviation_px: Math.max(0, ...Object.values(dimensions).map(value => value.deviation_px)),
    horizontal_overflow: samples.some(sample => sample.scroll_width > sample.viewport_width + TOLERANCE_PX),
    violations, dimensions};
}

function summarizePlayback(probe, stats) {
  const frames = probe.presentations;
  const intervals = frames.slice(1).map((frame, index) => frame.at - frames[index].at);
  const sorted = [...intervals].sort((a, b) => a - b);
  const percentile = fraction => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] : null;
  return {geometry: summarizeGeometry(probe.samples), presentation_count: frames.length,
    presentation_mismatches: frames.filter(frame => frame.left_frame !== frame.right_frame),
    display_fps: frames.length > 1 ? (frames.length - 1) * 1000 / (frames.at(-1).at - frames[0].at) : null,
    presentation_interval_ms: {p50: percentile(.5), p95: percentile(.95), p99: percentile(.99),
      max: sorted.at(-1) ?? null},
    gaps_over_50_ms: intervals.filter(value => value > 50).length,
    stream_starts: probe.streams, stream_restarts: Math.max(0, probe.streams.length - 1),
    layout_shift_count: probe.shifts.length,
    cumulative_layout_shift: probe.shifts.filter(shift => !shift.had_recent_input).reduce((sum, shift) => sum + shift.value, 0),
    all_layout_shift_value: probe.shifts.reduce((sum, shift) => sum + shift.value, 0),
    layout_shifts: probe.shifts, stream_stats: stats,
    note: 'FPS is observed canvas presentation rate through the current connection, not a pass threshold.'};
}

async function snapshot(page, output, frame) {
  const images = await page.evaluate(() => {
    const left = document.querySelector('#canvas'), right = document.querySelector('#comparisonCanvas');
    const canvas = document.createElement('canvas'), header = 48;
    canvas.width = left.width + right.width; canvas.height = Math.max(left.height, right.height) + header;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#11151a'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#fff'; ctx.font = '20px sans-serif'; ctx.textBaseline = 'middle';
    const frame = left.dataset.frame;
    ctx.fillText(`最原始结果 / ORIGINAL · frame ${frame}`, 16, header / 2);
    ctx.fillText(`进一步优化后 / OPTIMIZED · frame ${frame}`, left.width + 16, header / 2);
    // Native pixels, no resizing, interpolation, contrast, or sharpening.
    ctx.drawImage(left, 0, header); ctx.drawImage(right, left.width, header);
    return {original: left.toDataURL('image/png'), optimized: right.toDataURL('image/png'),
      comparison: canvas.toDataURL('image/png'), frame, right_frame: right.dataset.frame,
      native_size: [left.width, left.height], labels: [document.querySelector('#leftCompareLabel').textContent,
        document.querySelector('#rightCompareLabel').textContent]};
  });
  const basename = `frame-${String(frame).padStart(4, '0')}`;
  for (const kind of ['original', 'optimized', 'comparison']) {
    await fs.writeFile(path.join(output, `${basename}-${kind}.png`), Buffer.from(images[kind].split(',')[1], 'base64'));
  }
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#snapshot').click();
  const download = await downloadPromise;
  const downloadedPath = path.join(output, `${basename}-ui-download.png`);
  await download.saveAs(downloadedPath);
  const downloaded = await fs.readFile(downloadedPath);
  const downloadSize = [downloaded.readUInt32BE(16), downloaded.readUInt32BE(20)];
  if (downloadSize[0] !== images.native_size[0] * 2 || downloadSize[1] < images.native_size[1]) {
    throw new Error(`Frame ${frame}: exported comparison does not preserve native panel resolution`);
  }
  if (process.env.SCGS_LAYOUT_PHASE === 'after' && downloadSize[1] !== images.native_size[1] + 56) {
    throw new Error(`Frame ${frame}: expected the 56px provenance/warning header in the UI download`);
  }
  await page.screenshot({path: path.join(output, `${basename}-page.png`), fullPage: true});
  return {frame: images.frame, right_frame: images.right_frame, native_size: images.native_size,
    labels: images.labels, comparison: `${basename}-comparison.png`, page: `${basename}-page.png`,
    application_download: `${basename}-ui-download.png`, application_download_size: downloadSize};
}

(async () => {
  const output = path.resolve(process.env.SCGS_VIEWER_LAYOUT_OUTPUT || 'outputs/viewer-layout-check');
  await fs.mkdir(output, {recursive: true});
  const url = new URL(process.env.SCGS_VIEWER_URL || 'http://127.0.0.1:8892');
  url.searchParams.set('autoplay', '0'); url.searchParams.set('loop', '1');
  const report = {passed: false, url: url.toString(), phase: process.env.SCGS_LAYOUT_PHASE || null,
    started_at: new Date().toISOString(), viewports: [], screenshots: [], errors: [], failures: []};
  let browser;
  try {
    browser = await chromium.launch({headless: true,
      ...(process.env.SCGS_CHROME_PATH ? {executablePath: process.env.SCGS_CHROME_PATH} : {})});
    for (const viewport of [{width: 1440, height: 1000}, {width: 1024, height: 900}, {width: 390, height: 844}]) {
      const context = await browser.newContext({viewport, deviceScaleFactor: 1, acceptDownloads: true});
      const page = await context.newPage();
      page.setDefaultTimeout(60000);
      page.on('pageerror', error => report.errors.push({viewport, message: error.message}));
      await page.addInitScript(instrument, SELECTORS);
      await page.goto(url.toString());
      await page.waitForFunction(() => document.querySelector('#canvas')?.dataset.frame === '0' &&
        document.querySelector('#comparisonCanvas')?.dataset.frame === '0', null, {timeout: 120000});
      await page.evaluate(async () => {
        await document.fonts.ready;
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      });
      const scene = await page.evaluate(async () => (await (await fetch('/api/status')).json()).scene);
      if (!scene.comparison) throw new Error('Expected the actual original/optimized comparison scene.');
      report.scene ||= scene;
      if (process.env.SCGS_LAYOUT_SURFACE_DETAIL === '1' && !scene.protected_surface_detail) {
        throw new Error('Expected the audited protected surface detail model, not the previous preview.');
      }
      if (process.env.SCGS_LAYOUT_PHASE === 'after') {
        const leftLabel = await page.locator('#leftCompareLabel').innerText();
        const rightLabel = await page.locator('#rightCompareLabel').innerText();
        if (!leftLabel.includes('最原始') || !rightLabel.includes('进一步优化')) {
          report.failures.push(`${viewport.width}px original/latest comparison labels are missing or reversed`);
        }
        if (!scene.event_validation?.approved &&
            (!(await page.locator('#eventDecision').isVisible()) ||
             !(await page.locator('#eventDecision').innerText()).includes('尚未通过整段验收'))) {
          report.failures.push(`${viewport.width}px unapproved event warning is not visible`);
        }
      }
      const samples = [];
      for (const frame of FRAMES) {
        await page.locator('#timeline').evaluate((input, selected) => {
          input.value = String(selected); input.dispatchEvent(new Event('input', {bubbles: true}));
        }, frame);
        await page.waitForFunction(selected => document.querySelector('#canvas').dataset.frame === String(selected) &&
          document.querySelector('#comparisonCanvas').dataset.frame === String(selected), frame);
        const sample = await page.evaluate(async () => {
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          return window.viewerLayoutProbe.sample();
        });
        samples.push(sample);
        const intervalCheck = scene.event_validation?.interval_checks?.[frame - 1];
        const failedIntervalVisible = intervalCheck ? (!intervalCheck.passed &&
          !!sample.interval_text && !sample.interval_text.includes('通过粗尺度') &&
          !sample.interval_text.includes('初始帧')) : sample.interval_text?.includes('未通过事件检查');
        if (scene.event_validation?.failed_intervals?.includes(frame - 1) && !failedIntervalVisible) {
          report.failures.push(`${viewport.width}px frame ${frame} hides a failed event interval`);
        }
        if (viewport.width === 1440 && SNAPSHOTS.has(frame)) report.screenshots.push(await snapshot(page, output, frame));
      }
      const geometry = summarizeGeometry(samples);
      const result = {viewport, frames: FRAMES, geometry, samples,
        frame_sync_passed: samples.every(sample => sample.left_frame === sample.right_frame)};
      report.viewports.push(result);
      if (geometry.violations.length || geometry.horizontal_overflow) report.failures.push(`${viewport.width}px deterministic seek changes page geometry`);
      if (!result.frame_sync_passed) report.failures.push(`${viewport.width}px left/right frame mismatch`);
      if (viewport.width === 1440) {
        await page.locator('#restart').click();
        await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '0' &&
          document.querySelector('#comparisonCanvas').dataset.frame === '0');
        await page.locator('#loop').check();
        await page.locator('#speed').selectOption('1');
        await page.evaluate(() => {window.viewerLayoutProbe.active = true;});
        await page.locator('#play').click();
        // A fixed observation window is necessary to measure layout during playback.
        await page.waitForTimeout(10000);
        const measured = await page.evaluate(() => {
          window.viewerLayoutProbe.active = false;
          const {sample, ...probe} = window.viewerLayoutProbe;
          return {probe, stats: window.gaussianViewerStats || null};
        });
        result.playback = summarizePlayback(measured.probe, measured.stats);
        await fs.writeFile(path.join(output, 'playback-samples.json'), JSON.stringify(measured, null, 2) + '\n');
        if (result.playback.geometry.violations.length || result.playback.geometry.horizontal_overflow) {
          report.failures.push('1440px loop playback changes page geometry');
        }
        if (result.playback.presentation_mismatches.length) report.failures.push('1440px playback left/right frame mismatch');
        if (!result.playback.presentation_count) report.failures.push('1440px playback presented no frames');
        if (await page.locator('#play').getAttribute('aria-label') === '暂停') await page.locator('#play').click();
        if (process.env.SCGS_LAYOUT_SURFACE_DETAIL === '1') {
          await page.locator('#timeline').evaluate(input => {
            input.value = '100'; input.dispatchEvent(new Event('input', {bubbles: true}));
          });
          await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '100' &&
            document.querySelector('#comparisonCanvas').dataset.frame === '100');
          result.novel_view_playback = [];
          for (const [name, yaw, pitch] of [['back', 180, 0], ['roof', 0, 35]]) {
            await page.locator('#timeline').evaluate(input => {
              input.value = '100'; input.dispatchEvent(new Event('input', {bubbles: true}));
            });
            await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '100' &&
              document.querySelector('#comparisonCanvas').dataset.frame === '100');
            const previousId = await page.locator('#canvas').getAttribute('data-request-id');
            await page.locator(`.preset[data-yaw="${yaw}"][data-pitch="${pitch}"]`).click();
            await page.waitForFunction(old => document.querySelector('#canvas').dataset.requestId !== old &&
              document.querySelector('#canvas').dataset.requestId === document.querySelector('#comparisonCanvas').dataset.requestId,
              previousId);
            const destination = path.join(output, name);
            await fs.mkdir(destination, {recursive: true});
            report.screenshots.push({view: name, ...(await snapshot(page, destination, 100))});
            await page.evaluate(() => {
              const probe = window.viewerLayoutProbe;
              for (const key of ['samples', 'presentations', 'shifts', 'streams']) probe[key] = [];
              probe.active = true;
            });
            await page.locator('#play').click();
            await page.waitForTimeout(5000);
            const viewMeasured = await page.evaluate(() => {
              window.viewerLayoutProbe.active = false;
              const {sample, ...probe} = window.viewerLayoutProbe;
              return {probe, stats: window.gaussianViewerStats || null};
            });
            const playback = summarizePlayback(viewMeasured.probe, viewMeasured.stats);
            result.novel_view_playback.push({view: name, ...playback});
            if (!playback.presentation_count || playback.presentation_mismatches.length ||
                playback.geometry.violations.length || playback.geometry.horizontal_overflow) {
              report.failures.push(`${name} playback is absent, desynchronized, or changes layout`);
            }
            if (await page.locator('#play').getAttribute('aria-label') === '暂停') await page.locator('#play').click();
          }
        }
      }
      await page.screenshot({path: path.join(output, `viewport-${viewport.width}.png`), fullPage: true});
      console.log(JSON.stringify({viewport, seek_max_shift_px: geometry.maximum_deviation_px,
        playback_max_shift_px: result.playback?.geometry.maximum_deviation_px,
        stream_restarts: result.playback?.stream_restarts}));
      await context.close();
    }
    if (report.errors.length) report.failures.push('Browser application errors');
    report.passed = !report.failures.length;
  } catch (error) {
    report.failures.push(error.stack || String(error));
  } finally {
    if (browser) await browser.close();
    report.finished_at = new Date().toISOString();
    await fs.writeFile(path.join(output, 'layout-report.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({passed: report.passed, output, failures: report.failures}, null, 2));
    if (!report.passed) process.exitCode = 1;
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
