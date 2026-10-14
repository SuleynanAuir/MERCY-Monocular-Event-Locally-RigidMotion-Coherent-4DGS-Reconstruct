// Measure actual canvas presentations, not packet arrivals or source FPS.
const {chromium} = require(process.env.SCGS_PLAYWRIGHT_PATH || 'playwright-core');
const fs = require('node:fs/promises');
const path = require('node:path');
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.SCGS_CHROME_PATH});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      window.viewerPresentations = [];
      const draw = CanvasRenderingContext2D.prototype.drawImage;
      CanvasRenderingContext2D.prototype.drawImage = function (...args) {
        const result = draw.apply(this, args);
        if (this.canvas.id === 'canvas') {
          const at = performance.now();
          queueMicrotask(() => window.viewerPresentations.push({at, frame: +this.canvas.dataset.frame}));
        }
        return result;
      };
    });
    await page.goto(process.env.SCGS_VIEWER_URL || 'http://127.0.0.1:8890');
    await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '0', null, {timeout: 120000});
    const scene = await page.evaluate(async () => (await (await fetch('/api/status')).json()).scene);
    await page.locator('#resolution').selectOption(process.env.SCGS_RESOLUTION || '768');
    if (process.env.SCGS_STREAM_RATE) await page.locator('#streamRate').selectOption(process.env.SCGS_STREAM_RATE);
    await page.locator('#loop').check();
    await page.locator('#speed').selectOption('1');
    await page.getByRole('button', {name: '播放', exact: true}).click();
    // Fixed observation period is intentional for this frame-pacing benchmark.
    await page.waitForTimeout(10000);
    await page.getByRole('button', {name: '暂停', exact: true}).click();
    const report = await page.evaluate(scene => {
      const all = window.viewerPresentations;
      const end = performance.now() - 100, start = end - 7000;
      const frames = all.filter(f => f.at >= start && f.at <= end);
      const intervals = frames.slice(1).map((f, i) => f.at - frames[i].at);
      const sorted = [...intervals].sort((a, b) => a - b);
      const percentile = p => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
      const expectedStep = scene.fps / Number(document.querySelector('#streamRate').value);
      return {frames: frames.length, display_fps: (frames.length - 1) * 1000 / (frames.at(-1).at - frames[0].at),
        frame_interval_ms: {p50: percentile(.5), p95: percentile(.95), p99: percentile(.99), max: Math.max(...intervals)},
        gaps_over_50ms: intervals.filter(t => t > 50).length,
        frames_skipped: frames.slice(1).reduce((total, f, i) => total + Math.max(0, (f.frame - frames[i].frame + scene.frame_count) % scene.frame_count - expectedStep), 0),
        source_fps: scene.fps, resolution: document.querySelector('#resolution').value,
        stream_stats: window.gaussianViewerStats || null};
    }, scene);
    report.console_errors = errors;
    console.log(JSON.stringify(report, null, 2));
    if (process.env.SCGS_MEASUREMENT_OUTPUT) {
      const dest = path.resolve(process.env.SCGS_MEASUREMENT_OUTPUT);
      await fs.mkdir(path.dirname(dest), {recursive: true});
      await fs.writeFile(dest, JSON.stringify(report, null, 2) + '\n');
    }
    if (errors.length) process.exitCode = 1;
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
