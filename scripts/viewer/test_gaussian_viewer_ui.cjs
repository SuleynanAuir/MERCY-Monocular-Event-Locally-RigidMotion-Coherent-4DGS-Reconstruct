/* Optional end-to-end test. Uses an isolated headless browser, never a user's profile.
 * npm install --prefix /tmp/scgs-browser-test playwright-core
 * SCGS_PLAYWRIGHT_PATH=/tmp/scgs-browser-test/node_modules/playwright-core \
 * SCGS_CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
 * node scripts/viewer/test_gaussian_viewer_ui.cjs
 */
const {chromium} = require(process.env.SCGS_PLAYWRIGHT_PATH || 'playwright-core');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

(async () => {
  const output = path.resolve(process.env.SCGS_VIEWER_TEST_OUTPUT || 'outputs/fan-viewer-ui');
  await fs.mkdir(output, {recursive: true});
  const browser = await chromium.launch({headless: true,
    ...(process.env.SCGS_CHROME_PATH ? {executablePath: process.env.SCGS_CHROME_PATH} : {})});
  const errors = [], checks = [];
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}, acceptDownloads: true});
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  const url = process.env.SCGS_VIEWER_URL || 'http://127.0.0.1:8890';
  const imageHash = async () => crypto.createHash('sha256').update(
    await page.locator('#canvas').evaluate(canvas => canvas.toDataURL())).digest('hex');
  const requestId = () => page.locator('#canvas').getAttribute('data-request-id');
  const changed = async (id) => page.waitForFunction(old => document.querySelector('#canvas').dataset.requestId !== old, id);
  try {
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '0', null, {timeout: 120000});
    const scene = await page.evaluate(async () => (await (await fetch('/api/status')).json()).scene);
    assert.equal(await page.locator('#overlay').isVisible(), false);
    assert.equal(await page.locator('#gaussianCount').innerText(), scene.gaussian_count.toLocaleString('en-US'));
    const initial = await imageHash();
    checks.push(`Actual ${scene.name} loaded: ${scene.gaussian_count} Gaussians, initial frame rendered`);
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});

    await page.locator('#timeline').evaluate(input => {input.value = '75'; input.dispatchEvent(new Event('input', {bubbles: true}));});
    await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '75');
    const frame75 = await imageHash();
    assert.notEqual(frame75, initial);
    assert.match(await page.locator('#timeLabel').innerText(), /1\.25/);
    checks.push('Timeline seek changes baked Gaussian frame and image');

    let previous = await requestId();
    const beforeOrbit = await page.locator('#canvas').evaluate(canvas => canvas.toDataURL());
    await page.getByRole('button', {name: '斜视', exact: true}).click();
    // A resize/seek response can still be in flight. Wait for the image change,
    // not merely the next response ID, before asserting the camera result.
    await page.waitForFunction(before => document.querySelector('#canvas').toDataURL() !== before, beforeOrbit);
    assert.notEqual(await imageHash(), frame75);
    assert.equal(await page.locator('#canvas').getAttribute('data-frame'), '75');
    await page.screenshot({path: path.join(output, 'orbit.png'), fullPage: true});
    checks.push('Camera preset re-renders the same frame from a new 3D viewpoint');

    const bounds = await page.locator('#stage').boundingBox();
    previous = await requestId();
    await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    await page.mouse.down(); await page.mouse.move(bounds.x + bounds.width / 2 + 80, bounds.y + bounds.height / 2 + 30, {steps: 5}); await page.mouse.up();
    await changed(previous);
    checks.push('Pointer drag orbits camera');
    previous = await requestId();
    await page.mouse.wheel(0, -150); await changed(previous);
    checks.push('Mouse wheel zooms camera');
    previous = await requestId();
    await page.keyboard.down('Shift'); await page.mouse.down(); await page.mouse.move(bounds.x + bounds.width / 2 + 105, bounds.y + bounds.height / 2 + 55); await page.mouse.up(); await page.keyboard.up('Shift');
    await changed(previous);
    checks.push('Shift-drag pans camera');

    previous = await requestId();
    await page.getByRole('button', {name: '浅色背景', exact: true}).click();
    await changed(previous);
    assert.equal(await page.getByRole('button', {name: '浅色背景', exact: true}).getAttribute('aria-pressed'), 'true');
    await page.locator('#resolution').selectOption('512');
    await page.waitForFunction(() => Math.max(document.querySelector('#canvas').width, document.querySelector('#canvas').height) === 512);
    checks.push('Background and resolution controls work');

    await page.locator('#reset').click();
    await page.locator('#restart').click();
    await page.waitForFunction(() => document.querySelector('#canvas').dataset.frame === '0');
    await page.locator('#loop').check();
    await page.locator('#speed').selectOption('0.5');
    await page.getByRole('button', {name: '播放', exact: true}).click();
    await page.waitForFunction(() => +document.querySelector('#canvas').dataset.frame >= 40, null, {timeout: 15000});
    const playingFps = await page.locator('#displayFps').innerText();
    const gpuMs = await page.locator('#renderMs').innerText();
    await page.getByRole('button', {name: '暂停', exact: true}).click();
    const stopped = await page.locator('#timeline').inputValue();
    await page.waitForFunction(frame => document.querySelector('#canvas').dataset.frame === frame, stopped);
    await page.locator('#next').click();
    await page.waitForFunction(frame => +document.querySelector('#canvas').dataset.frame === +frame + 1, stopped);
    await page.locator('#previous').click();
    await page.waitForFunction(frame => document.querySelector('#canvas').dataset.frame === frame, stopped);
    checks.push('Playback, pause, playback rate, previous/next frame work');
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#snapshot').click();
    const download = await downloadPromise;
    await download.saveAs(path.join(output, 'snapshot.png'));
    checks.push('Rendered image snapshot downloads');

    await page.locator('#stage').focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(frame => +document.querySelector('#canvas').dataset.frame === +frame + 1, stopped);
    await page.keyboard.press('r');
    checks.push('Keyboard frame stepping and reset work');

    // Two-frame tail exercises the short-stream path (less than the warmup).
    await page.locator('#loop').uncheck();
    await page.locator('#speed').selectOption('1');
    await page.locator('#timeline').evaluate((input, frame) => {input.value = String(frame); input.dispatchEvent(new Event('input', {bubbles: true}));}, scene.frame_count - 2);
    await page.waitForFunction(frame => document.querySelector('#canvas').dataset.frame === String(frame), scene.frame_count - 2);
    await page.getByRole('button', {name: '播放', exact: true}).click();
    await page.waitForFunction(frame => document.querySelector('#canvas').dataset.frame === String(frame) && document.querySelector('#play').getAttribute('aria-label') === '播放', scene.frame_count - 1);
    checks.push('Short stream reaches the exact last frame and stops automatically');

    await page.locator('#streamRate').selectOption('30');
    await page.locator('#loop').check();
    await page.getByRole('button', {name: '播放', exact: true}).click();
    await page.waitForFunction(() => window.gaussianViewerStats?.target_fps === 30 && +document.querySelector('#canvas').dataset.frame > 12);
    const movingBounds = await page.locator('#stage').boundingBox();
    await page.mouse.move(movingBounds.x + movingBounds.width / 2, movingBounds.y + movingBounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(movingBounds.x + movingBounds.width / 2 + 50, movingBounds.y + movingBounds.height / 2 + 25, {steps: 5});
    await page.mouse.up();
    await page.waitForFunction(() => window.gaussianViewerStats?.presented > 15 && document.querySelector('#canvas').dataset.requestId.includes(':'));
    assert.equal(await page.locator('#play').getAttribute('aria-label'), '暂停');
    checks.push('30 FPS streaming works; camera interaction flushes old frames and resumes playback');
    await page.getByRole('button', {name: '暂停', exact: true}).click();
    await page.waitForFunction(() => !document.querySelector('#canvas').dataset.requestId.includes(':'));
    const idleId = await requestId();
    await page.waitForTimeout(300);
    assert.equal(await requestId(), idleId);
    checks.push('Paused viewer stops continuous rendering and retains the exact displayed frame');
    await page.locator('#streamRate').selectOption('60');
    await page.getByRole('button', {name: '深色背景', exact: true}).click();
    await page.locator('#resolution').selectOption('768');
    previous = await requestId();
    await page.setViewportSize({width: 390, height: 844});
    await changed(previous);
    await page.waitForFunction(() => {
      const c = document.querySelector('#canvas'), rect = document.querySelector('#stage').getBoundingClientRect();
      return Math.abs(c.width / c.height - rect.width / rect.height) < .02;
    });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    checks.push('Mobile layout has no horizontal overflow and rendering matches the viewport aspect');

    const retry = await context.newPage();
    retry.on('pageerror', error => errors.push(error.message));
    await retry.route('**/api/status', route => route.abort());
    await retry.goto(url);
    await retry.getByRole('heading', {name: '无法连接渲染服务器'}).waitFor();
    await retry.unroute('**/api/status');
    await retry.waitForFunction(() => !!document.querySelector('#canvas').dataset.frame, null, {timeout: 15000});
    checks.push('Connection failure is visible; automatic reconnection succeeds');
    await retry.close();
    assert.deepEqual(errors, []);
    const report = {passed: true, checks, console_errors: errors, sample_display_fps: +playingFps,
      sample_render_ms: +gpuMs, url, scene, note: 'Display FPS is sampled through the SSH tunnel at 512 px with the scene default stream rate; motion speed 0.5x.'};
    await fs.writeFile(path.join(output, 'browser-tests.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    await page.screenshot({path: path.join(output, 'failure.png'), fullPage: true});
    throw error;
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
