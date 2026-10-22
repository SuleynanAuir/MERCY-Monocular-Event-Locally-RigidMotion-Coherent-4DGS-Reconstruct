"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  // Avoid replacing identical live text on every animation frame.
  const text = (id, value) => {if ($(id).textContent !== value) $(id).textContent = value;};
  const canvas = $("canvas"), stage = $("stage"), context = canvas.getContext("2d", {alpha: false});
  const rightCanvas = $("comparisonCanvas"), rightContext = rightCanvas.getContext("2d", {alpha: false});
  const state = {scene: null, socket: null, playing: false, frame: 0, dirty: true, pending: new Set(),
    id: 0, yaw: 0, pitch: 0, zoom: 1, pan_x: 0, pan_y: 0, background: "dark", speed: 1,
    lastReceived: 0, displayedFrame: -1, lastDrawnId: 0, delivered: [], retry: null,
    stream: null, minimumId: 0, interacting: false, lastStats: 0, lastLabels: 0, lastStreamStatus: 0,
    autoRateCap: 60, renderMode: "gaussian", hybridQuality: "surface"};
  let pointer = null, wheelTimer = null;
  const launchOptions = new URLSearchParams(window.location.search);
  const sceneQuery = launchOptions.has('scene') ? `?scene=${encodeURIComponent(launchOptions.get('scene'))}` : '';
  $('objectSelect').onchange = () => {
    setPlaying(false);
    const query = new URLSearchParams(window.location.search);
    query.set('scene', $('objectSelect').value);
    query.set('mode', state.renderMode);
    query.set('quality', state.hybridQuality);
    // A new document clears decoded frames, camera and time together. Other
    // browser tabs retain their own object and WebSocket renderer.
    window.location.search = query.toString();
  };
  let autoStartPending = launchOptions.get("autoplay") === "1";

  function comparisonView() {
    return !!state.scene?.comparison || (state.scene?.comparison_modes || []).includes(state.renderMode);
  }
  function modeResolution(mode = state.renderMode) {
    return (["mesh", "hybrid"].includes(mode) ? state.scene?.[mode]?.default_resolution : null) || state.scene?.default_resolution || 768;
  }
  function updateModeUI() {
    const scene = state.scene;
    if (!scene) return;
    const mesh = state.renderMode === "mesh", hybrid = state.renderMode === "hybrid", paired = comparisonView();
    const modeInfo = hybrid ? scene.hybrid : scene.mesh;
    const modeDescription = hybrid ? (scene.hybrid.quality_descriptions?.[state.hybridQuality] || scene.hybrid.description) : modeInfo?.description;
    $("renderMode").value = state.renderMode;
    $("renderMode").disabled = false;
    $("meshOption").disabled = !scene.mesh?.available;
    $("hybridOption").disabled = !scene.hybrid?.available;
    $("ultraResolution").disabled = paired;
    $("hybridQualityField").hidden = !hybrid;
    $("hybridQuality").value = state.hybridQuality;
    $("meshNotice").textContent = hybrid ? modeDescription : scene.mesh?.available ? scene.mesh.description :
      (scene.mesh?.description || "当前场景未生成高斯重建网格。");
    stage.classList.toggle("comparison", paired);
    $("expandView").textContent = paired ? "放大对比" : "放大画面";
    $("expandView").title = paired ? "全屏显示两个对比视窗" : "全屏显示当前物件";
    rightCanvas.hidden = !paired;
    $("leftCompareLabel").hidden = $("rightCompareLabel").hidden = !paired;
    $("eventNotice").hidden = mesh || hybrid || !scene.event_validation;
    $("sceneKind").textContent = hybrid ? "MESH + GAUSSIAN" : mesh ? "MESH SURFACE" : "GAUSSIAN SCENE";
    $("sceneName").textContent = hybrid ? "V1 / V3 混合动态对比" : mesh ? "V1 / V3 Mesh 对比" : scene.name;
    $("leftCompareLabel").textContent = mesh || hybrid ? (modeInfo.left_label || "V1 Mesh · 原始结果") : scene.left_label;
    $("rightCompareLabel").textContent = mesh || hybrid ? (modeInfo.right_label || "V3 Mesh · 最新优化结果") : scene.right_label;
    $("sequenceBadge").textContent = hybrid ? "Mesh＋Gaussian · 同帧同视角" : mesh ? "V1 / V3 Mesh · 同帧同视角" : paired ?
      "原始 / 优化 · 同帧同视角" : scene.event_validation ? "原始事件校正 · 候选预览" : "逐帧高斯 · 实时渲染";
    $("gaussianCount").textContent = (mesh ? scene.mesh.triangle_count : scene.gaussian_count).toLocaleString("en-US");
    $("primitiveLabel").textContent = mesh ? "TRIANGLES / 三角面数量" : "GAUSSIANS / 高斯数量";
    $("sceneDescription").textContent = mesh || hybrid ? modeDescription :
      (scene.description || "真实高斯序列，随时间更新位置与朝向。改变视角时重新进行 3D 渲染。");
    $("snapshot").textContent = hybrid ? "保存 Mesh＋Gaussian 动态对比图 ↓" : mesh ? "保存 V1 / V3 Mesh 对比图 ↓" : paired ? "保存原始 / 优化对比图 ↓" : "保存当前画面 ↓";
    canvas.setAttribute("aria-label", hybrid ? "V1 Mesh 绑定驱动的高斯混合渲染" : mesh ? "V1 原始高斯重建的 Mesh，与右侧同帧同视角" : "GPU 实时渲染的动态高斯");
    rightCanvas.setAttribute("aria-label", hybrid ? "V3 Mesh 绑定驱动的高斯混合渲染" : mesh ? "V3 最新优化高斯重建的 Mesh，与左侧同帧同视角" : "优化高斯，同步右视窗");
  }
  function stopStream() {
    if (!state.stream) return;
    state.stream.buffer.close(); state.stream = null;
    state.minimumId = ++state.id; state.pending.clear();
    if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({type: "stop"}));
  }

  function status(text, kind = "") {
    $("connectionText").textContent = text;
    $("statusDot").className = `dot ${kind}`;
  }
  function overlay(title, message, error = false) {
    $("overlay").hidden = false;
    $("overlay").classList.toggle("error", error);
    $("loadTitle").textContent = title;
    $("loadMessage").textContent = message;
  }
  function controls(enabled) {
    ["play", "previous", "next", "restart", "timeline", "snapshot"].forEach(id => $(id).disabled = !enabled);
  }
  function setPlaying(value) {
    stopStream();
    state.playing = value && !!state.scene && state.socket?.readyState === WebSocket.OPEN;
    if (state.playing && state.frame >= state.scene.frame_count - 1) state.frame = 0;
    $("play").replaceChildren(document.createTextNode(state.playing ? "Ⅱ " : "▶ "));
    const label = document.createElement("span");
    label.textContent = state.playing ? "暂停" : "播放";
    $("play").append(label);
    $("play").setAttribute("aria-label", label.textContent);
    state.dirty = true;
    labels();
  }
  function setFrame(frame) {
    if (!state.scene) return;
    state.frame = Math.max(0, Math.min(state.scene.frame_count - 1, Math.round(frame)));
    state.dirty = true;
    labels();
  }
  function labels() {
    if (!state.scene) return;
    const scene = state.scene, seconds = scene.times_seconds[state.frame];
    $("timeline").value = state.frame;
    $("timeline").setAttribute("aria-valuetext", `${seconds.toFixed(2)} 秒，第 ${state.frame + 1} 帧`);
    text("timeCurrent", seconds.toFixed(2));
    text("timeTotal", `/ ${scene.duration_seconds.toFixed(2)} s`);
    $("frameLabel").textContent = `帧 ${String(state.frame + 1).padStart(3, "0")} / ${scene.frame_count}`;
    const rpm = scene.rpm[state.frame];
    $("rpm").textContent = Number.isFinite(rpm) ? `${rpm.toFixed(1)} RPM` : "—";
    $("motionLabel").textContent = state.renderMode === "hybrid" ? "Mesh 绑定运动 · Gaussian 外观" : state.renderMode === "mesh" ? "高斯重建 Mesh · 同步部件动画" : scene.motion_label || (scene.ramp_seconds == null ? "逐帧高斯序列" :
      seconds === 0 ? "静止 · 起始状态" : seconds < scene.ramp_seconds ? "平滑加速" : "匀速旋转");
    if (scene.event_validation) {
      const check = scene.event_validation.interval_checks?.[state.frame - 1];
      const names = {relative_l1: '变化幅度偏差', polarity_agreement: '事件极性不一致',
        cosine: '变化分布不一致', unsupported_change_fraction: '存在事件未支持的变化',
        insufficient_event_evidence: '事件证据不足'};
      const intervalText = state.frame === 0 ? "当前：初始帧" : check ?
        `当前：${check.passed ? '通过粗尺度检查' : check.reasons.map(reason => names[reason] || reason).join(' / ')}${check.reference_weak ? ' · 参考证据偏弱' : ''}` :
        scene.event_validation.failed_intervals?.includes(state.frame - 1) ? "当前时间段：未通过事件检查" :
          "当前时间段：通过粗尺度事件检查";
      text("eventInterval", intervalText);
      $("eventInterval").title = intervalText;
    }
  }
  function reset(yaw = 0, pitch = 0) {
    Object.assign(state, {yaw, pitch, zoom: 1, pan_x: 0, pan_y: 0, dirty: true});
    document.querySelectorAll(".preset").forEach(button => button.classList.toggle("active",
      +button.dataset.yaw === yaw && +button.dataset.pitch === pitch));
    const viewName = yaw === 0 && pitch === 0 ? "正视" : yaw === 180 && pitch === 0 ? "背面" :
      yaw === 0 && pitch === 35 ? "俯视" : yaw === 0 && pitch === -35 ? "仰视" : "自由视角";
    $("viewLabel").textContent = `${viewName} · 透视`;
  }
  function dimensions() {
    const rect = stage.getBoundingClientRect(), longest = +$("resolution").value;
    // Keep the calibrated square view in both panels. A narrow panel with the
    // same vertical FOV would crop the bucket and look like missing geometry.
    if (comparisonView()) return {width: Math.min(longest, 1024), height: Math.min(longest, 1024)};
    // Preserve the calibrated horizontal field in portrait layouts: rendering
    // a narrower camera would cut off the outer fan blades. CSS contains the
    // square render in the taller canvas instead.
    const ratio = Math.max(1, Math.min(5, rect.width / rect.height));
    const size = ratio >= 1 ? {width: longest, height: Math.max(128, Math.round(longest / ratio))} :
      {width: Math.max(128, Math.round(longest * ratio)), height: longest};
    // The 1280 preset follows the canvas aspect while respecting the server's
    // pixel budget, including nearly square windows and portrait mobile views.
    const scale = Math.min(1, Math.sqrt(1280 * 960 / (size.width * size.height)));
    return {width: Math.floor(size.width * scale), height: Math.floor(size.height * scale)};
  }
  function requestFrame() {
    if (state.playing && !state.interacting) {
      if (state.dirty) startStream();
      return;
    }
    // Camera previews and exact single-frame inspection keep the old bounded
    // request mode. Continuous playback uses a credit-based push stream.
    if (!state.scene || state.pending.size >= 4 || !state.dirty || state.socket?.readyState !== WebSocket.OPEN || document.hidden) return;
    state.dirty = false;
    const {yaw, pitch, zoom, pan_x, pan_y, background} = state;
    const id = ++state.id;
    state.pending.add(id);
    state.socket.send(JSON.stringify({id, frame: state.frame, yaw, pitch, zoom, pan_x, pan_y,
      background, render_mode: state.renderMode, hybrid_quality: state.hybridQuality, fovy: state.scene.fovy,
      quality: state.interacting ? 72 : (state.scene.default_still_quality ?? 85), ...dimensions()}));
  }
  function startStream() {
    if (!state.scene || state.socket?.readyState !== WebSocket.OPEN || document.hidden) return;
    stopStream(); state.pending.clear();
    const id = ++state.id;
    const prewarmed = ["mesh", "hybrid"].includes(state.renderMode) ? state.scene[state.renderMode]?.prewarmed_frames : state.scene.prewarmed_frames;
    const defaultView = prewarmed === state.scene.frame_count &&
      (state.renderMode !== "hybrid" || (state.scene.hybrid.prewarmed_qualities || [state.scene.hybrid.default_quality || "surface"]).includes(state.hybridQuality)) &&
      state.yaw === 0 && state.pitch === 0 && state.zoom === 1 && state.pan_x === 0 && state.pan_y === 0 &&
      state.background === state.scene.default_background && +$("resolution").value === modeResolution();
    const rate = $("streamRate").value === 'auto' ? Math.min(state.autoRateCap,
      comparisonView() && !defaultView ? 30 : 60) : +$("streamRate").value;
    state.minimumId = id; state.dirty = false; state.delivered = [];
    const windowSize = state.scene.stream_window || 24;
    const warmup = Math.min(windowSize - 4, Math.ceil(rate * (state.scene.buffer_seconds || .4)));
    state.stream = {id, buffer: new GaussianPlayback(rate, warmup, windowSize), ack: -1,
      jpeg: new Map(), jpegBytes: 0, cachePending: new Set()};
    window.gaussianViewerStats = {presented: 0, underruns: 0, dropped: 0, buffered: 0,
      target_fps: rate, cache_hits: 0, client_cache_hits: 0, decode_ms: 0, draw_ms: 0};
    const {yaw, pitch, zoom, pan_x, pan_y, background} = state;
    state.socket.send(JSON.stringify({type: "stream", id, frame: state.frame, speed: state.speed,
      rate, loop: $("loop").checked, yaw, pitch, zoom, pan_x, pan_y, background, render_mode: state.renderMode, hybrid_quality: state.hybridQuality,
      fovy: state.scene.fovy, quality: state.scene.default_stream_quality ?? 78, ...dimensions()}));
    $("streamStatus").textContent = "预缓冲…";
  }
  function present(item, close = true) {
    const {meta, bitmap} = item, now = performance.now();
    $("snapshot").disabled = false;
    const panelWidth = meta.comparison ? meta.panel_width : meta.width;
    if (canvas.width !== panelWidth || canvas.height !== meta.height) {
      canvas.width = panelWidth; canvas.height = meta.height;
    }
    if (meta.comparison) {
      if (rightCanvas.width !== panelWidth || rightCanvas.height !== meta.height) {
        rightCanvas.width = panelWidth; rightCanvas.height = meta.height;
      }
      context.drawImage(bitmap, 0, 0, panelWidth, meta.height, 0, 0, panelWidth, meta.height);
      rightContext.drawImage(bitmap, panelWidth, 0, panelWidth, meta.height, 0, 0, panelWidth, meta.height);
      rightCanvas.dataset.frame = String(meta.right_frame);
    } else context.drawImage(bitmap, 0, 0);
    if (close) bitmap.close();
    canvas.dataset.frame = String(meta.frame);
    canvas.dataset.requestId = meta.stream ? `${meta.id}:${meta.stream.seq}` : String(meta.id);
    state.displayedFrame = meta.frame; state.lastReceived = now;
    state.delivered.push(now); state.delivered = state.delivered.filter(t => t >= now - 1500);
    if (now - state.lastStats >= 250) {
      const elapsed = now - state.delivered[0];
      $("displayFps").textContent = elapsed > 0 ? ((state.delivered.length - 1) * 1000 / elapsed).toFixed(1) : "—";
      $("renderMs").textContent = meta.render_ms.toFixed(1);
      $("renderKind").textContent = meta.client_cache_hit ? "ms · 本地帧缓存" :
        meta.cache_hit ? "ms · 同视角缓存" : "ms 渲染";
      state.lastStats = now;
    }
    if (meta.stream) {
      state.frame = meta.frame;
      if (now - state.lastLabels > 80 || meta.frame === state.scene.frame_count - 1) {
        labels(); state.lastLabels = now;
      }
    }
    if (!$("overlay").hidden) {$("overlay").hidden = true; status("已连接", "ready");}
    if (meta.stream && window.gaussianViewerStats) {
      window.gaussianViewerStats.draw_ms = performance.now() - now;
      window.gaussianViewerStats.cache_hits += Number(!!meta.cache_hit);
      window.gaussianViewerStats.client_cache_hits += Number(!!meta.client_cache_hit);
    }
  }
  function scheduleRetry() {
    clearTimeout(state.retry);
    state.retry = setTimeout(poll, 2000);
  }
  function connect() {
    const socket = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws${sceneQuery}`);
    state.socket = socket; socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      status("已连接", "ready"); state.pending.clear(); state.dirty = true;
      controls(true); requestFrame();
      if (autoStartPending) {
        autoStartPending = false;
        $("loop").checked = launchOptions.get("loop") === "1";
        setPlaying(true);
      }
    };
    socket.onmessage = async (event) => {
      if (state.socket !== socket) return;
      let responseId;
      try {
        if (typeof event.data === "string") {
          const error = JSON.parse(event.data);
          if (error.type === "stream_end") {
            if (state.stream?.id === error.id) state.stream.buffer.finish(error.seq);
            return;
          }
          setPlaying(false); status("渲染错误", "error");
          overlay("暂时无法渲染", error.message || "请检查服务器日志", true);
          state.dirty = false; state.pending.clear(); return;
        }
        const data = event.data;
        const length = new DataView(data).getUint32(0, true);
        if (length > 65536 || length + 4 > data.byteLength) throw new Error("Invalid image packet");
        const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, length)));
        responseId = meta.id;
        // Keep compressed JPEGs (<= 64 MiB), not hundreds of decoded bitmaps.
        // Stream lifetime fixes frame/camera/quality; replacements start empty.
        const stream = state.stream?.id === meta.id ? state.stream : null;
        if (meta.id < state.minimumId || (meta.stream && !stream)) return;
        let blob;
        if (meta.client_cache_hit) {
          blob = stream?.jpeg.get(meta.frame);
          if (!blob) throw new Error("Missing retained stream frame");
        } else {
          if (length + 4 === data.byteLength) throw new Error("Missing JPEG");
          blob = new Blob([new Uint8Array(data, 4 + length)], {type: "image/jpeg"});
          if (stream && !stream.jpeg.has(meta.frame) && stream.jpegBytes + blob.size <= 64 * 1024**2) {
            stream.jpeg.set(meta.frame, blob); stream.jpegBytes += blob.size;
            stream.cachePending.add(meta.frame);
          }
        }
        const decodeStart = performance.now();
        const bitmap = await createImageBitmap(blob);
        if (state.socket !== socket || meta.id < state.minimumId) {bitmap.close(); return;}
        if (meta.stream) {
          if (state.stream?.id !== meta.id) {bitmap.close(); return;}
          window.gaussianViewerStats.decode_ms = performance.now() - decodeStart;
          const item = {meta, bitmap}, buffer = state.stream.buffer;
          if (buffer.push(item) && meta.stream.seq === 0) present(item, false);
        } else {
          if (meta.id <= state.lastDrawnId || state.stream) {bitmap.close(); return;}
          state.lastDrawnId = meta.id; present({meta, bitmap});
        }
      } catch (error) {
        console.error("Viewer image error", error);
        overlay("画面解析失败", "请刷新页面重试。", true); setPlaying(false); state.dirty = false; state.pending.clear();
      } finally {
        if (state.socket === socket) state.pending.delete(responseId);
      }
    };
    socket.onclose = () => {
      if (state.socket !== socket) return;
      state.pending.clear(); setPlaying(false); controls(false);
      status("连接已断开", "error");
      overlay("正在重新连接", "请保持 SSH 隧道和 AutoDL 渲染服务运行。"); scheduleRetry();
    };
    socket.onerror = () => status("连接异常", "error");
  }
  async function poll() {
    try {
      const response = await fetch(`/api/status${sceneQuery}`, {cache: "no-store", signal: AbortSignal.timeout(8000)});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.objects) {
        const select = $('objectSelect');
        select.replaceChildren(...data.objects.map(item => {
          const option = document.createElement('option');
          option.value = item.id; option.textContent = item.name;
          return option;
        }));
        select.value = data.scene_id;
        select.disabled = data.objects.length < 2;
      }
      $("loadProgress").value = data.progress || 0;
      if (data.error) {
        overlay("场景加载失败", `${data.message}（${data.detail}）`, true);
        status("加载失败", "error"); return;
      }
      if (!data.ready) {
        overlay("加载动态高斯", data.message); status("加载中"); scheduleRetry(); return;
      }
      const firstLoad = !state.scene;
      state.scene = data.scene; const scene = data.scene;
      if (firstLoad) {
        $("streamRate").value = scene.comparison ? 'auto' : String(scene.default_stream_rate || 60);
        $("resolution").value = String(scene.default_resolution || 768);
        state.background = scene.default_background || "dark";
        document.querySelectorAll(".swatch").forEach(button => {
          const selected = button.dataset.background === state.background;
          button.classList.toggle("active", selected);
          button.setAttribute("aria-pressed", String(selected));
        });
      }
      state.frame = Math.min(state.frame, scene.frame_count - 1);
      $("sceneName").textContent = scene.name;
      stage.classList.toggle("comparison", !!scene.comparison);
      rightCanvas.hidden = !scene.comparison;
      $("leftCompareLabel").hidden = $("rightCompareLabel").hidden = !scene.comparison;
      if (scene.comparison) {
        $("leftCompareLabel").textContent = scene.left_label;
        $("rightCompareLabel").textContent = scene.right_label;
        $("snapshot").textContent = "保存原始 / 优化对比图 ↓";
      }
      const event = scene.event_validation;
      $("eventNotice").hidden = !event;
      $("geometryNotice").hidden = !scene.geometry_repaired;
      $("geometryDetails").hidden = !scene.geometry_repaired;
      $("geometryNotice").textContent = scene.geometry_repaired ? scene.geometry_repair_scope : "";
      $("sequenceBadge").textContent = scene.comparison ? "原始 / 优化 · 同帧同视角" : event ? "原始事件校正 · 候选预览" : "逐帧高斯 · 实时渲染";
      if (event) {
        $("eventDecision").textContent = event.approved ? "事件校正候选 · 粗尺度检查通过" : "事件校正后 · 尚未通过整段验收";
        $("eventSummary").textContent = `区间通过 ${event.summary.passed}/${event.summary.intervals} · 事件余弦 ${event.summary.cosine.toFixed(3)}` +
          (event.validation_summary ? ` · 留出事件 ${event.validation_summary.passed}/${event.validation_summary.intervals}` : '') +
          (event.reference_weak_intervals?.length ? ` · 参考弱证据 ${event.reference_weak_intervals.length} 段（仍计入验收）` : '');
      }
      $("gaussianCount").textContent = scene.gaussian_count.toLocaleString("en-US");
      $("frameCount").textContent = `${scene.frame_count} 帧`;
      $("sourceFps").textContent = `${scene.fps} FPS`;
      $("duration").textContent = `${scene.duration_seconds.toFixed(2)} s`;
      $("tickMid").textContent = `${(scene.duration_seconds / 2).toFixed(1)} s`;
      $("tickEnd").textContent = `${scene.duration_seconds.toFixed(1)} s`;
      $("timeline").max = scene.frame_count - 1;
      $("gpuName").textContent = scene.gpu;
      $("cacheInfo").textContent = `${scene.cache.toUpperCase()} ${scene.comparison ? "缓存上限" : "缓存"} · ${scene.cache_gib.toFixed(2)} GiB 运动数据`;
      $("gpuDot").className = "dot ready";
      const launchMode = launchOptions.get("mode");
      if (firstLoad && ["mesh", "hybrid"].includes(launchMode) && scene[launchMode]?.available) state.renderMode = launchMode;
      if (["mesh", "hybrid"].includes(state.renderMode) && !scene[state.renderMode]?.available) state.renderMode = "gaussian";
      if (firstLoad) {
        $("resolution").value = String(modeResolution());
        state.hybridQuality = ["surface", "native", "balanced", "detail"].includes(launchOptions.get("quality")) ?
          launchOptions.get("quality") : (scene.hybrid?.default_quality || "surface");
      }
      updateModeUI(); labels(); connect();
    } catch (error) {
      status("等待服务器", "error");
      overlay("无法连接渲染服务器", "确认服务已启动，并通过 SSH 将端口转发到本机。"); scheduleRetry();
    }
  }
  function tick(now) {
    if (state.playing && state.stream && !document.hidden && !state.interacting) {
      const stream = state.stream, buffer = stream.buffer, item = buffer.take(now);
      if (item?.ended) {setPlaying(false); labels();}
      else if (item) {
        present(item);
        if (buffer.last - stream.ack >= 4 || buffer.last === buffer.end) {
          const retained = [...stream.cachePending].slice(0, buffer.limit);
          state.socket.send(JSON.stringify({type: "ack", id: stream.id, seq: buffer.last, cached_frames: retained}));
          retained.forEach(frame => stream.cachePending.delete(frame));
          stream.ack = buffer.last;
        }
        window.gaussianViewerStats.presented++;
      }
      Object.assign(window.gaussianViewerStats, {underruns: buffer.underruns, dropped: buffer.dropped,
        buffered: buffer.frames.size});
      if ($("streamRate").value === 'auto' && buffer.rate === 60 && buffer.underruns >= 2) {
        state.autoRateCap = 30; state.dirty = true;
      }
      if (now - state.lastStreamStatus >= 250) {
        text("streamStatus", buffer.buffering ? "网络缓冲…" :
          `${buffer.rate} FPS 同步流 · 缓存 ${buffer.frames.size} 帧 · 断流 ${buffer.underruns}`);
        state.lastStreamStatus = now;
      }
    } else if (!state.playing) {
      text("streamStatus", "逐帧查看");
    } else if (state.interacting) {
      text("streamStatus", "视角交互 · 松开恢复播放");
    }
    if (now - state.lastReceived > 1500) $("displayFps").textContent = "0.0";
    requestFrame(); requestAnimationFrame(tick);
  }
  $("play").onclick = () => setPlaying(!state.playing);
  $("restart").onclick = () => {setPlaying(false); setFrame(0);};
  $("previous").onclick = () => {setPlaying(false); setFrame(state.frame - 1);};
  $("next").onclick = () => {setPlaying(false); setFrame(state.frame + 1);};
  $("timeline").oninput = event => {
    const frame = +event.target.value; // Pausing updates labels, including this input.
    setPlaying(false); setFrame(frame);
  };
  $("speed").onchange = event => {state.speed = +event.target.value; state.dirty = true;};
  $("loop").onchange = () => state.dirty = true;
  $("streamRate").onchange = () => {state.autoRateCap = 60; state.dirty = true;};
  $("resolution").onchange = () => state.dirty = true;
  $("renderMode").onchange = event => {
    const mode = event.target.value;
    if (!state.scene || !["gaussian", "mesh", "hybrid"].includes(mode) ||
        (["mesh", "hybrid"].includes(mode) && !state.scene[mode]?.available)) return;
    stopStream(); state.pending.clear();
    state.minimumId = ++state.id;
    const wasDefaultResolution = +$("resolution").value === modeResolution();
    state.renderMode = mode; state.autoRateCap = 60; state.dirty = true;
    if (wasDefaultResolution) $("resolution").value = String(modeResolution());
    state.displayedFrame = -1;
    $("snapshot").disabled = true;
    overlay("切换显示模式", mode === "hybrid" ? "正在渲染 Mesh＋Gaussian 动态效果…" : mode === "mesh" ? "正在渲染高斯重建网格…" : "正在渲染高斯…");
    updateModeUI(); labels();
  };
  $("hybridQuality").onchange = event => {
    const quality = event.target.value;
    if (state.renderMode !== "hybrid" || !["surface", "native", "balanced", "detail"].includes(quality)) return;
    stopStream(); state.pending.clear(); state.minimumId = ++state.id;
    state.hybridQuality = quality; state.autoRateCap = 60; state.dirty = true;
    state.displayedFrame = -1; $("snapshot").disabled = true;
    overlay("切换混合画质", quality === "surface" ? "正在渲染表面柔化效果…" : quality === "native" ? "正在渲染 GS 原生外观…" : "正在渲染旧版效果…");
    updateModeUI();
  };
  $("expandView").onclick = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await $("viewerPanel").requestFullscreen();
    } catch (error) {$("expandView").title = "浏览器未允许全屏，可放大窗口查看";}
  };
  document.addEventListener("fullscreenchange", () => {
    $("expandView").textContent = document.fullscreenElement ? "退出全屏" : "放大对比";
  });
  $("reset").onclick = $("resetFloating").onclick = () => reset();
  document.querySelectorAll(".preset").forEach(button => button.onclick = () => reset(+button.dataset.yaw, +button.dataset.pitch));
  document.querySelectorAll(".swatch").forEach(button => button.onclick = () => {
    state.background = button.dataset.background; state.dirty = true;
    document.querySelectorAll(".swatch").forEach(other => {
      other.classList.toggle("active", other === button); other.setAttribute("aria-pressed", String(other === button));
    });
  });
  stage.oncontextmenu = event => event.preventDefault();
  stage.onpointerdown = event => {
    if (event.target.closest("button") || event.target.closest(".overlay")) return;
    stage.focus({preventScroll: true}); stage.setPointerCapture(event.pointerId);
    state.interacting = true; stopStream();
    pointer = {x: event.clientX, y: event.clientY, id: event.pointerId, pan: event.button === 2 || event.shiftKey};
  };
  stage.onpointermove = event => {
    if (!pointer || pointer.id !== event.pointerId) return;
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    pointer.x = event.clientX; pointer.y = event.clientY;
    if (pointer.pan) {
      const factor = state.zoom * 1.2 / stage.clientHeight;
      state.pan_x = Math.max(-5, Math.min(5, state.pan_x - dx * factor));
      state.pan_y = Math.max(-5, Math.min(5, state.pan_y + dy * factor));
    } else {
      state.yaw = ((state.yaw - dx * .35 + 540) % 360) - 180;
      state.pitch = Math.max(-85, Math.min(85, state.pitch + dy * .35));
    }
    document.querySelectorAll(".preset").forEach(button => button.classList.remove("active"));
    $("viewLabel").textContent = "自由视角 · 透视"; state.dirty = true;
  };
  stage.onpointerup = stage.onpointercancel = event => {
    if (pointer?.id === event.pointerId) {pointer = null; state.interacting = false; state.dirty = true;}
  };
  stage.ondblclick = () => reset();
  stage.addEventListener("wheel", event => {
    state.interacting = true; stopStream(); clearTimeout(wheelTimer);
    wheelTimer = setTimeout(() => {state.interacting = false; state.dirty = true;}, 180);
    event.preventDefault(); state.zoom = Math.max(.2, Math.min(8, state.zoom * Math.exp(event.deltaY * .001)));
    state.dirty = true;
  }, {passive: false});
  document.addEventListener("keydown", event => {
    if (event.target.closest("input,select,textarea,button,a") || !state.scene) return;
    if (event.code === "Space") {event.preventDefault(); setPlaying(!state.playing);}
    if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
      event.preventDefault(); setPlaying(false); setFrame(state.frame + (event.code === "ArrowRight" ? 1 : -1));
    }
    if (event.code === "KeyR") reset();
  });
  document.addEventListener("visibilitychange", () => {if (document.hidden) setPlaying(false); else state.dirty = true;});
  // Square calibrated comparison renders do not depend on CSS panel size.
  // Don't restart their stream when surrounding text or layout changes size.
  new ResizeObserver(() => {if (!comparisonView()) state.dirty = true;}).observe(stage);
  $("snapshot").onclick = () => {
    if (state.displayedFrame < 0) return;
    let snapshot = canvas;
    if (comparisonView()) {
      snapshot = document.createElement("canvas");
      const header = 56;
      snapshot.width = canvas.width + rightCanvas.width; snapshot.height = canvas.height + header;
      const ctx = snapshot.getContext("2d");
      ctx.fillStyle = "#171c23"; ctx.fillRect(0, 0, snapshot.width, header);
      ctx.fillStyle = "#e6ebf2"; ctx.font = "16px sans-serif";
      ctx.fillText($("leftCompareLabel").textContent, 12, 22, canvas.width - 24);
      ctx.fillText($("rightCompareLabel").textContent, canvas.width + 12, 22, rightCanvas.width - 24);
      ctx.font = "12px sans-serif"; ctx.fillStyle = "#b6aa8f";
      const frame = state.displayedFrame;
      const stamp = `帧 ${frame + 1} / ${state.scene.frame_count} · ${state.scene.times_seconds[frame].toFixed(3)} s · 同帧同视角`;
      ctx.fillText(stamp, 12, 43);
      ctx.fillText(state.renderMode === "hybrid" ? "Mesh 绑定驱动＋Gaussian 外观；混合渲染" : state.renderMode === "mesh" ? "V1 与 V3 各自高斯重建的近似表面" : state.scene.event_validation?.approved ? "粗尺度事件检查通过；新视角未经验证" :
        "尚未通过整段事件验收；背面完整性不保证", canvas.width + 12, 43, rightCanvas.width - 24);
      ctx.drawImage(canvas, 0, header); ctx.drawImage(rightCanvas, canvas.width, header);
    }
    snapshot.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `${state.renderMode}-frame-${String(state.displayedFrame).padStart(4, "0")}.png`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/png");
  };
  window.addEventListener("beforeunload", () => state.socket?.close());
  reset(); poll(); requestAnimationFrame(tick);
})();
