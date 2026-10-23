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
    preparing: false, preparationId: 0, autoRateCap: 60, renderMode: "gaussian", hybridQuality: "surface", clayStyle: "solid"};
  const rgbCanvas = $("rgbCanvas"), eventCanvas = $("eventCanvas");
  const rgbContext = rgbCanvas.getContext("2d", {alpha: false});
  const eventContext = eventCanvas.getContext("2d", {alpha: false});
  // ROIs are shared image coordinates. Crops always read the displayed frame.
  const detail = {enabled:false, regions:[], selected:null, next:0, drag:null, views:[]};
  const detailPairs = [[canvas,$("detailLeft")],[rightCanvas,$("detailRight")]];
  const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
  // Shared vertical framing preserves RGB/event alignment and ROI coordinates.
  // The envelope only expands during a camera view, avoiding frame-to-frame breathing.
  const overviewFit={top:0,bottom:1,key:null,measured:false,sample:null};
  function fitOverview() {
    if(!comparisonView() || !state.scene?.reference?.available) return;
    const key=[state.renderMode,state.yaw,state.pitch,state.zoom,state.pan_x,state.pan_y,state.background].join('/');
    if(key!==overviewFit.key) Object.assign(overviewFit,{top:0,bottom:1,key,measured:false});
    if(state.background!=="light" || state.interacting) return;
    if(!overviewFit.sample) {overviewFit.sample=document.createElement("canvas");overviewFit.sample.width=128;overviewFit.sample.height=128;}
    const ctx=overviewFit.sample.getContext("2d",{willReadFrequently:true});
    if(!ctx.getImageData) return;
    let first=128,last=-1;
    for(const source of [canvas,rightCanvas,rgbCanvas,eventCanvas]) {
      ctx.clearRect(0,0,128,128);ctx.drawImage(source,0,0,128,128);
      const pixels=ctx.getImageData(0,0,128,128).data;
      for(let y=0;y<128;y++) {
        let count=0;
        for(let x=0;x<128;x++){const i=(y*128+x)*4;if(pixels[i+3]>128 && Math.min(pixels[i],pixels[i+1],pixels[i+2])<235) count++;}
        if(count>=2){first=Math.min(first,y);last=Math.max(last,y);}
      }
    }
    if(last<first) return; // Empty event/reference frames never collapse the view.
    const margin=8; // ~6% image padding, including motion and sampling safety.
    const top=Math.max(0,(first-margin)/128),bottom=Math.min(1,(last+1+margin)/128);
    if(!overviewFit.measured) Object.assign(overviewFit,{top,bottom,measured:true});
    else {overviewFit.top=Math.min(overviewFit.top,top);overviewFit.bottom=Math.max(overviewFit.bottom,bottom);}
  }
  function overviewBounds() {
    let top=overviewFit.top,bottom=overviewFit.bottom;
    if(detail.enabled) for(const r of detail.regions) {top=Math.min(top,r.y-.035);bottom=Math.max(bottom,r.y+r.h+.02);}
    return {top:Math.max(0,top),bottom:Math.min(1,bottom)};
  }
  function layoutOverview() {
    if(!comparisonView() || !state.scene?.reference?.available) {
      for(const source of [canvas,rightCanvas,rgbCanvas,eventCanvas]) {source.style.height="";source.style.top="";}
      return;
    }
    const {top,bottom}=overviewBounds(),width=$("leftRenderCell").getBoundingClientRect().width;
    if(!width || !canvas.width || !canvas.height) return;
    const fullHeight=width*canvas.height/canvas.width;
    const header=$("leftCompareLabel").getBoundingClientRect().height || 28;
    stage.style.setProperty("--overview-height",`${header+fullHeight*(bottom-top)}px`);
    for(const source of [canvas,rightCanvas,rgbCanvas,eventCanvas]) {
      source.style.height=`${fullHeight}px`;source.style.top=`${-top*fullHeight}px`;
    }
  }
  function detailImageHeight(cellWidth,overviewHeight) {
    const pad=Math.min(12,Math.max(4,cellWidth*.025));
    const ratios=detail.regions.map(r=>r.h*canvas.height/(r.w*canvas.width));
    const natural=Math.max(0,...ratios)*(cellWidth-2*pad)+2*pad;
    // Extreme tall/narrow selections cannot force every other crop into a tall column.
    return Math.ceil(Math.max(2*pad+24,Math.min(natural,cellWidth*1.6,overviewHeight*1.15)));
  }
  function detailGeometry(source) {
    const rect=source.getBoundingClientRect(),area=source.parentElement?.getBoundingClientRect() || rect;
    const scale=Math.min(rect.width/source.width,rect.height/source.height);
    const width=source.width*scale,height=source.height*scale;
    return {rect:area,x:rect.left-area.left+(rect.width-width)/2,y:rect.top-area.top+(rect.height-height)/2,width,height};
  }
  function detailLabel(index) {
    let label="";
    do {label=String.fromCharCode(65+index%26)+label;index=Math.floor(index/26)-1;} while(index>=0);
    return label;
  }
  function activeRegion() {return detail.regions.find(r=>r.id===detail.selected);}
  function paintRegions(ctx,box,unit=1,handles=true) {
    for(const r of detail.regions) {
      const x=box.x+r.x*box.width,y=box.y+r.y*box.height,w=r.w*box.width,h=r.h*box.height;
      ctx.save();ctx.strokeStyle=r.color;ctx.lineWidth=(r.id===detail.selected?2:1.5)*unit;
      ctx.strokeRect(x,y,w,h);
      ctx.fillStyle=r.color;ctx.fillRect(x,Math.max(0,y-20*unit),24*unit,20*unit);
      ctx.fillStyle="#fff";ctx.font=`600 ${12*unit}px sans-serif`;ctx.textAlign="center";
      ctx.fillText(r.label,x+12*unit,Math.max(0,y-20*unit)+14*unit);
      if(handles && r.id===detail.selected) for(const [hx,hy] of [[0,0],[.5,0],[1,0],[1,.5],[1,1],[.5,1],[0,1],[0,.5]]) {
        ctx.fillStyle="#fff";ctx.fillRect(x+hx*w-4*unit,y+hy*h-4*unit,8*unit,8*unit);
        ctx.strokeRect(x+hx*w-4*unit,y+hy*h-4*unit,8*unit,8*unit);
      }
      ctx.restore();
    }
  }
  function paintCrop(ctx,source,r,width,height) {
    ctx.fillStyle="#fff";ctx.fillRect(0,0,width,height);
    const pad=Math.min(12,Math.max(4,width*.025));
    const sw=r.w*source.width,sh=r.h*source.height,scale=Math.min((width-2*pad)/sw,(height-2*pad)/sh);
    const w=sw*scale,h=sh*scale;
    ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality="high";
    ctx.drawImage(source,r.x*source.width,r.y*source.height,sw,sh,(width-w)/2,(height-h)/2,w,h);
  }
  function rebuildDetail() {
    $("detailSelect").replaceChildren();$("detailRow").replaceChildren();detail.views=[];
    for(const r of detail.regions) {
      const option=document.createElement("option");option.value=String(r.id);option.textContent=r.label;
      $("detailSelect").append(option);
      for(const [i,source] of [canvas,rightCanvas].entries()) {
        const cell=document.createElement("section"),title=document.createElement("div"),area=document.createElement("div"),view=document.createElement("canvas");
        cell.className="detail-cell";cell.style.setProperty("--roi-color",r.color);cell.dataset.region=r.label;
        title.className="detail-title";title.textContent=`${r.label} · ${i===0?"V1":"V3"}`;
        area.className="detail-image";view.setAttribute("aria-label",`${r.label} 选区 ${i===0?"V1":"V3"} 动态细节`);
        area.append(view);cell.append(title,area);$("detailRow").append(cell);
        detail.views.push({region:r,source,view});
      }
    }
    $("detailSelect").value=String(detail.selected);
    $("detailSelect").disabled=$("detailRemove").disabled=!detail.regions.length;
    $("detailRow").style.setProperty("--detail-count",Math.max(1,detail.views.length));
    refreshDetail();
  }
  function addDetail() {
    const id=detail.next++,offset=(id%5)*.055;
    const colors=["#007e91","#d04a39","#7753bf","#b17a00","#22864f","#c44191","#315cb6"];
    detail.regions.push({id,label:detailLabel(id),color:colors[id] || `hsl(${(id*137.508)%360} 65% 38%)`,x:.28+offset,y:.28+offset,w:.2,h:.2});
    detail.selected=id;rebuildDetail();
  }
  function refreshDetail() {
    layoutOverview();
    const active=detail.enabled && comparisonView() && state.displayedFrame>=0;
    $("detailRow").hidden=!active || !detail.regions.length;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    for(const [source,layer] of detailPairs) {
      layer.hidden=!active;
      if(!active) continue;
      const box=detailGeometry(source),w=Math.round(box.rect.width*dpr),h=Math.round(box.rect.height*dpr);
      if(!w || !h) continue;
      if(layer.width!==w || layer.height!==h){layer.width=w;layer.height=h;}
      const ctx=layer.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,box.rect.width,box.rect.height);
      paintRegions(ctx,box);layer.dataset.frame=String(state.displayedFrame);
      layer.dataset.regions=JSON.stringify(detail.regions);
    }
    if(!active) return;
    if(detail.views.length) {
      const cellWidth=detail.views[0].view.getBoundingClientRect().width;
      const topHeight=canvas.height*$("leftRenderCell").getBoundingClientRect().width/canvas.width*(overviewBounds().bottom-overviewBounds().top);
      if(cellWidth>0) $("detailRow").style.setProperty("--detail-image-height",`${detailImageHeight(cellWidth,topHeight)}px`);
    }
    for(const {region,source,view} of detail.views) {
      const rect=view.getBoundingClientRect(),w=Math.round(rect.width*dpr),h=Math.round(rect.height*dpr);
      if(!w || !h) continue;
      if(view.width!==w || view.height!==h){view.width=w;view.height=h;}
      paintCrop(view.getContext("2d"),source,region,w,h);view.dataset.frame=String(state.displayedFrame);
    }
  }
  function detailPoint(event,source,bounded=false) {
    const box=detailGeometry(source);
    let x=(event.clientX-box.rect.left-box.x)/box.width,y=(event.clientY-box.rect.top-box.y)/box.height;
    if(!bounded && (x<0 || x>1 || y<0 || y>1)) return null;
    return {x:clamp(x,0,1),y:clamp(y,0,1),box};
  }
  function hitDetail(p) {
    const regions=[...detail.regions].sort((a,b)=>(a.id===detail.selected?1:0)-(b.id===detail.selected?1:0)).reverse();
    for(const r of regions) {
      const tx=9/p.box.width,ty=9/p.box.height;
      if(p.x<r.x-tx || p.x>r.x+r.w+tx || p.y<r.y-ty || p.y>r.y+r.h+ty) continue;
      const horizontal=Math.abs(p.x-r.x)<tx?"w":Math.abs(p.x-r.x-r.w)<tx?"e":"";
      const vertical=Math.abs(p.y-r.y)<ty?"n":Math.abs(p.y-r.y-r.h)<ty?"s":"";
      return {region:r,handle:vertical+horizontal || "move"};
    }
    return null;
  }
  function startDetail(event,source) {
    const p=detailPoint(event,source);if(!p) return;
    let hit=hitDetail(p);
    if(!hit) {
      const r=activeRegion();if(!r) return;
      r.x=clamp(p.x-r.w/2,0,1-r.w);r.y=clamp(p.y-r.h/2,0,1-r.h);
      hit={region:r,handle:"move"};
    }
    detail.selected=hit.region.id;$("detailSelect").value=String(detail.selected);
    detail.drag={id:event.pointerId,source,...hit,start:p,original:{...hit.region}};
    stage.focus({preventScroll:true});stage.setPointerCapture(event.pointerId);refreshDetail();
  }
  function moveDetail(event) {
    const d=detail.drag;if(!d || d.id!==event.pointerId) return;
    const box=d.start.box;
    const p={x:clamp((event.clientX-box.rect.left-box.x)/box.width,0,1),y:clamp((event.clientY-box.rect.top-box.y)/box.height,0,1),box};
    const dx=p.x-d.start.x,dy=p.y-d.start.y,o=d.original,r=d.region;
    if(d.handle==="move") {r.x=clamp(o.x+dx,0,1-o.w);r.y=clamp(o.y+dy,0,1-o.h);}
    else {
      let left=o.x,right=o.x+o.w,top=o.y,bottom=o.y+o.h;
      const minW=Math.max(.015,12/p.box.width),minH=Math.max(.015,12/p.box.height);
      if(d.handle.includes("w")) left=clamp(o.x+dx,0,right-minW);
      if(d.handle.includes("e")) right=clamp(right+dx,left+minW,1);
      if(d.handle.includes("n")) top=clamp(o.y+dy,0,bottom-minH);
      if(d.handle.includes("s")) bottom=clamp(bottom+dy,top+minH,1);
      Object.assign(r,{x:left,y:top,w:right-left,h:bottom-top});
    }
    refreshDetail();
  }
  function updateDetailHint() {
    $("viewHint").textContent=detail.enabled && comparisonView()
      ? "选区：拖动框内移动 · 拖动边框 / 角点调整大小 · 下排同步对比 V1 / V3"
      : comparisonView() && state.scene?.reference?.available
        ? "V1 / V3：拖动旋转 · Shift 平移 · 滚轮缩放；RGB / 事件：固定相机视角"
        : "拖动旋转 · Shift 平移 · 滚轮缩放";
  }
  $("detailToggle").onclick=()=>{
    detail.enabled=!detail.enabled;detail.drag=null;
    canvas.style.cursor=rightCanvas.style.cursor="";
    $("detailToggle").setAttribute("aria-pressed",String(detail.enabled));
    $("detailToggle").textContent=detail.enabled?"关闭细节对比":"细节放大镜";
    $("detailOptions").hidden=!detail.enabled;stage.classList.toggle("detail-active",detail.enabled);
    if(detail.enabled && !detail.regions.length) addDetail();
    updateDetailHint();refreshDetail();
  };
  $("detailAdd").onclick=addDetail;
  $("detailSelect").onchange=()=>{detail.selected=Number($("detailSelect").value);refreshDetail();};
  $("detailRemove").onclick=()=>{detail.drag=null;detail.regions=detail.regions.filter(r=>r.id!==detail.selected);detail.selected=detail.regions.at(-1)?.id??null;rebuildDetail();};
  $("detailClear").onclick=()=>{detail.drag=null;detail.regions=[];detail.selected=null;rebuildDetail();};

  let references = null;
  let pointer = null, wheelTimer = null;
  const launchOptions = new URLSearchParams(window.location.search);
  const sceneQuery = launchOptions.has('scene') ? `?scene=${encodeURIComponent(launchOptions.get('scene'))}` : '';
  $('objectSelect').onchange = () => {
    setPlaying(false);
    const query = new URLSearchParams(window.location.search);
    query.set('scene', $('objectSelect').value);
    query.set('mode', state.renderMode);
    query.set('quality', state.hybridQuality);
    query.set('clay_style', state.clayStyle);
    // A new document clears decoded frames, camera and time together. Other
    // browser tabs retain their own object and WebSocket renderer.
    window.location.search = query.toString();
  };
  const sequences=new GaussianPlayback.Media.SequenceCache();
  const frameRequests=new Map();
  let exportJob=null;
  function sequenceSettings(mode=state.renderMode) {
    const {yaw,pitch,zoom,pan_x,pan_y,background}=state;
    return {yaw,pitch,zoom,pan_x,pan_y,background,render_mode:mode,hybrid_quality:state.hybridQuality,
      clay_style:state.clayStyle,fovy:state.scene.fovy,
      quality:state.scene.default_stream_quality??95,...dimensions()};
  }
  function sequenceKey(settings){return JSON.stringify(settings);}
  function readySequence(){if(!state.scene)return null;const e=sequences.get(sequenceKey(sequenceSettings()));return e?.complete && (!references || references.prepared)?e:null;}
  function rejectFrameRequests(error){for(const {reject,timer} of frameRequests.values()){clearTimeout(timer);reject(error);}frameRequests.clear();}
  function fetchFrame(settings,frame) {
    if(state.socket?.readyState!==WebSocket.OPEN)return Promise.reject(new Error('渲染连接已断开，请重连后重试'));
    const id=++state.id;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{frameRequests.delete(id);reject(new Error('帧下载超时，请重试；已缓存帧会保留'));},60000);
      frameRequests.set(id,{resolve,reject,timer});state.socket.send(JSON.stringify({...settings,id,frame}));
    });
  }
  async function ensureConnection(valid) {
    if(state.socket?.readyState===WebSocket.OPEN)return;
    if(state.socket?.readyState!==0)connect();
    const deadline=performance.now()+15000;
    while(state.socket?.readyState!==WebSocket.OPEN){
      if(!valid())throw new Error("已取消");
      if(performance.now()>deadline)throw new Error("无法连接渲染服务器，请恢复网络后重试");
      await new Promise(resolve=>setTimeout(resolve,50));
    }
  }
  async function prepareSequence(settings,valid,progress=()=>{}) {
    const entry=sequences.create(sequenceKey(settings),state.scene.frame_count);
    if(entry.complete)return entry;
    await ensureConnection(valid);
    let next=0,failure=null;
    await Promise.all(Array.from({length:4},async()=>{
      while(!failure && valid() && next<entry.count) {
        const index=next++;if(entry.frames.has(index))continue;
        try {const record=await fetchFrame(settings,index);sequences.put(entry,index,record);if(valid())progress(entry.frames.size,entry.count);}
        catch(error){failure=error;}
      }
    }));
    if(failure)throw failure;if(!valid())throw new Error('已取消');
    if(!entry.complete)throw new Error('整段缓存未完成，请重试');return entry;
  }
  async function decodeRecord(record) {
    const decoded=await Promise.allSettled([createImageBitmap(record.blob),...(references?[references.frame(record.meta.frame)]:[])]);
    const failure=decoded.find(r=>r.status==='rejected');
    if(failure){decoded.forEach(r=>{if(r.status==='fulfilled')r.value.close();});throw failure.reason;}
    const image=decoded[0].value,reference=decoded[1]?.value;
    return reference?{image,reference,close(){image.close();reference.close();}}:image;
  }
  async function pumpLocal(stream) {
    if(stream.pumping || stream.finished || state.stream!==stream)return;
    stream.pumping=true;
    try {
      while(state.stream===stream && stream.buffer.frames.size<stream.buffer.limit-2 && !stream.finished) {
        const seq=stream.seq++,position=stream.startFrame+Math.floor(seq*state.scene.fps*stream.speed/stream.buffer.rate+1e-9);
        const last=!stream.loop && position>=state.scene.frame_count-1;
        const frame=stream.loop?position%state.scene.frame_count:Math.min(position,state.scene.frame_count-1);
        const record=stream.entry.frames.get(frame);if(!record)throw new Error('本地缓存帧不完整');
        const bitmap=await decodeRecord(record);
        if(state.stream!==stream){bitmap.close();break;}
        stream.buffer.push({meta:{...record.meta,id:stream.id,frame,client_cache_hit:true,stream:{seq}},bitmap});
        if(last){stream.buffer.finish(seq);stream.finished=true;}
      }
    }catch(error){if(state.stream===stream){setPlaying(false);text('streamStatus',`本地解码失败：${error.message}`);}}
    finally{stream.pumping=false;}
  }
  function beginLocal() {
    const entry=readySequence();if(!entry){setPlaying(true);return;}
    stopStream();state.pending.clear();const id=++state.id;
    const rate=$('streamRate').value==='auto'?Math.min(30,state.scene.fps):Number($('streamRate').value);
    const stream={id,local:true,entry,seq:0,startFrame:state.frame,speed:state.speed,loop:$('loop').checked,
      buffer:new GaussianPlayback(rate,Math.min(12,Math.ceil(rate*.35)),20)};
    state.stream=stream;state.minimumId=id;state.dirty=false;state.delivered=[];
    window.gaussianViewerStats={presented:0,underruns:0,dropped:0,buffered:0,target_fps:rate,cache_hits:0,client_cache_hits:0,decode_ms:0,draw_ms:0,local:true};
    text('streamStatus','整段已缓存 · 正在本地解码');pumpLocal(stream);
  }

  let autoStartPending = launchOptions.get("autoplay") === "1";

  function comparisonView() {
    return !!state.scene?.comparison || (state.scene?.comparison_modes || []).includes(state.renderMode);
  }
  function modeResolution(mode = state.renderMode) {
    return (["mesh", "hybrid", "clay", "pseudocolor"].includes(mode) ? state.scene?.[mode]?.default_resolution : null) || state.scene?.default_resolution || 768;
  }
  function updateModeUI() {
    const scene = state.scene;
    if (!scene) return;
    const color = state.renderMode === "pseudocolor";
    const clay = (state.renderMode === "clay"), mesh = state.renderMode === "mesh", hybrid = state.renderMode === "hybrid", paired = comparisonView();
    const modeInfo = clay || color ? scene[state.renderMode] : hybrid ? scene.hybrid : scene.mesh;
    const modeDescription = hybrid ? (scene.hybrid.quality_descriptions?.[state.hybridQuality] || scene.hybrid.description) : modeInfo?.description;
    $("renderMode").value = state.renderMode;
    $("renderMode").disabled = false;
    $("clayOption").disabled = !scene.clay?.available;
    $("pseudocolorOption").disabled = !scene.pseudocolor?.available;
    $("meshOption").disabled = !scene.mesh?.available;
    $("hybridOption").disabled = !scene.hybrid?.available;
    $("ultraResolution").disabled = paired;
    $("clayStyleField").hidden = !clay;
    $("clayWire").checked = state.clayStyle === "wire";
    $("hybridQualityField").hidden = !hybrid;
    $("hybridQuality").value = state.hybridQuality;
    $("meshNotice").textContent = clay || color || hybrid ? modeDescription : scene.mesh?.available ? scene.mesh.description :
      (scene.mesh?.description || "当前场景未生成高斯重建网格。");
    stage.classList.toggle("comparison", paired);
    $("detailTools").hidden = !paired;
    refreshDetail();
    $("expandView").textContent = paired ? "放大对比" : "放大画面";
    $("expandView").title = paired ? "全屏显示两个对比视窗" : "全屏显示当前物件";
    rightCanvas.hidden = $("rightRenderCell").hidden = !paired;
    const quad = !!scene.reference?.available && paired;
    stage.classList.toggle("quad", quad);
    $("rgbCell").hidden = $("eventCell").hidden = !quad;
    if (quad) {
      $("referenceAlignment").textContent = "对齐 RGB · 同步";
      $("referenceAlignment").title = "与 RGB 使用相同像素坐标；事件区间结束时刻对应当前 RGB 帧，首帧无前序事件。";
      $("viewHint").textContent = "V1 / V3：拖动旋转 · Shift 平移 · 滚轮缩放；RGB / 事件：固定相机视角";
      $("expandView").textContent = "放大四列对比";
      $("expandView").title = "全屏显示四个同步视窗";
    }
    updateDetailHint();
    $("leftCompareLabel").hidden = $("rightCompareLabel").hidden = !paired;
    $("sceneKind").textContent = color ? "PSEUDOCOLOR" : clay ? "CLAY SURFACE" : hybrid ? "MESH + GAUSSIAN" : mesh ? "MESH SURFACE" : "GAUSSIAN SCENE";
    $("sceneName").textContent = color ? "V1 / V3 伪彩色对比" : clay ? "V1 / V3 白模对比" : hybrid ? "V1 / V3 混合动态对比" : mesh ? "V1 / V3 Mesh 对比" : scene.name;
    $("leftCompareLabel").textContent = clay || color || mesh || hybrid ? (modeInfo.left_label || "V1 Mesh · 原始结果") : scene.left_label;
    $("rightCompareLabel").textContent = clay || color || mesh || hybrid ? (modeInfo.right_label || "V3 Mesh · 最新优化结果") : scene.right_label;
    $("sequenceBadge").textContent = color ? "伪彩色 · 同帧同视角" : clay ? "白模 · 同帧同视角" : hybrid ? "Mesh＋Gaussian · 同帧同视角" : mesh ? "V1 / V3 Mesh · 同帧同视角" : paired ?
      "原始 / 优化 · 同帧同视角" : scene.event_validation ? "原始事件校正 · 候选预览" : "逐帧高斯 · 实时渲染";
    $("gaussianCount").textContent = (mesh ? modeInfo.triangle_count : scene.gaussian_count).toLocaleString("en-US");
    $("primitiveLabel").textContent = mesh ? "TRIANGLES / 三角面数量" : "GAUSSIANS / 高斯数量";
    $("sceneDescription").textContent = clay || color || mesh || hybrid ? modeDescription :
      (scene.description || "真实高斯序列，随时间更新位置与朝向。改变视角时重新进行 3D 渲染。");
    $("snapshot").textContent = color ? "保存 V1 / V3 伪彩色对比图 ↓" : clay ? "保存 V1 / V3 白模对比图 ↓" : hybrid ? "保存 Mesh＋Gaussian 动态对比图 ↓" : mesh ? "保存 V1 / V3 Mesh 对比图 ↓" : paired ? "保存原始 / 优化对比图 ↓" : "保存当前画面 ↓";
    if (quad) $("snapshot").textContent = "保存 1×4 对比图 ↓";
    canvas.setAttribute("aria-label", color ? "伪彩色表面，同步部件动画与对比视角" : clay ? "白色表面，同步部件动画与对比视角" : hybrid ? "V1 Mesh 绑定驱动的高斯混合渲染" : mesh ? "V1 原始高斯重建的 Mesh，与右侧同帧同视角" : "GPU 实时渲染的动态高斯");
    rightCanvas.setAttribute("aria-label", color ? "伪彩色表面，同步部件动画与对比视角" : clay ? "白色表面，同步部件动画与对比视角" : hybrid ? "V3 Mesh 绑定驱动的高斯混合渲染" : mesh ? "V3 最新优化高斯重建的 Mesh，与左侧同帧同视角" : "优化高斯，同步右视窗");
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
    const preparationId = ++state.preparationId;
    state.preparing = false;
    stopStream();
    if(state.exporting)return;
    if(value && $("smoothPlayback").checked && !readySequence()) {
      state.playing=false;state.preparing=true;$("play").textContent="取消准备";$("play").setAttribute("aria-label","取消准备");
      const settings=sequenceSettings(),key=sequenceKey(settings);
      const valid=()=>state.preparationId===preparationId && key===sequenceKey(sequenceSettings());
      (async()=>{
        if(references)await references.prepare(state.scene.frame_count,(done,total)=>{if(valid())text("streamStatus",`整段缓存 RGB / 事件 ${done}/${total}`);},valid);
        if(!valid())return;
        await prepareSequence(settings,valid,(done,total)=>text("streamStatus",`整段缓存 ${done}/${total} · 完成后本地播放`));
        if(valid())setPlaying(true);
      })().catch(error=>{if(state.preparationId===preparationId){setPlaying(false);text("streamStatus",error.message);}}).finally(()=>{
        if(state.preparationId===preparationId && !valid()){setPlaying(false);text("streamStatus","画面设置已变化，请重新播放");}
      });return;
    }
    if (value && references?.prepare && !references.prepared) {
      state.playing = false; state.preparing = true;
      $("play").textContent = "取消准备";
      $("play").setAttribute("aria-label", "取消准备");
      references.prepare(state.scene.frame_count, (done, total) => {
        if (state.preparationId === preparationId) text("streamStatus", `准备 RGB / 事件视频 ${done}/${total} · 首次播放缓存中`);
      }).then(() => {
        if (state.preparationId === preparationId) setPlaying(true);
      }).catch(error => {
        if (state.preparationId !== preparationId) return;
        setPlaying(false);
        overlay("参考视频准备失败", "请重试播放；已下载的帧会保留。", true);
        console.error("Reference preparation failed", error);
      });
      return;
    }
    state.playing = value && !!state.scene && (state.socket?.readyState === WebSocket.OPEN || !!readySequence());
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
    $("motionLabel").textContent = state.renderMode === "pseudocolor" ? "伪彩色 · 同步部件动画" : (state.renderMode === "clay") ? "白模 · 同步部件动画" : state.renderMode === "hybrid" ? "Mesh 绑定运动 · Gaussian 外观" : state.renderMode === "mesh" ? "高斯重建 Mesh · 同步部件动画" : scene.motion_label || (scene.ramp_seconds == null ? "逐帧高斯序列" :
      seconds === 0 ? "静止 · 起始状态" : seconds < scene.ramp_seconds ? "平滑加速" : "匀速旋转");
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
    if(state.exporting || state.preparing)return;
    if(!state.playing && state.dirty && $("smoothPlayback").checked && readySequence() && !state.localStill) {
      const entry=readySequence(),id=++state.id,record=entry.frames.get(state.frame);state.localStill=true;state.dirty=false;
      decodeRecord(record).then(bitmap=>{if(!state.playing && !state.dirty && sequenceKey(sequenceSettings())===entry.key)present({meta:{...record.meta,id,client_cache_hit:true},bitmap});else bitmap.close();})
        .catch(error=>text("streamStatus",error.message)).finally(()=>{state.localStill=false;});return;
    }
    if(state.localStill)return;
    if(state.dirty && !state.playing && state.socket?.readyState===3 && !state.reconnecting && performance.now()-(state.lastReconnect||0)>3000){
      state.lastReconnect=performance.now();state.reconnecting=true;
      poll().finally(()=>{state.reconnecting=false;});return;
    }
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
      background, render_mode: state.renderMode, hybrid_quality: state.hybridQuality, clay_style: state.clayStyle, fovy: state.scene.fovy,
      quality: state.interacting ? 72 : (state.scene.default_still_quality ?? 85), ...dimensions()}));
  }
  function startStream() {
    if($("smoothPlayback").checked){beginLocal();return;}
    if (!state.scene || state.socket?.readyState !== WebSocket.OPEN || document.hidden) return;
    stopStream(); state.pending.clear();
    const id = ++state.id;
    const prewarmed = ["mesh", "hybrid", "clay", "pseudocolor"].includes(state.renderMode) ? state.scene[state.renderMode]?.prewarmed_frames : state.scene.prewarmed_frames;
    const defaultView = prewarmed === state.scene.frame_count &&
      (state.renderMode !== "hybrid" || (state.scene.hybrid.prewarmed_qualities || [state.scene.hybrid.default_quality || "surface"]).includes(state.hybridQuality)) &&
      state.yaw === 0 && state.pitch === 0 && state.zoom === 1 && state.pan_x === 0 && state.pan_y === 0 &&
      state.background === state.scene.default_background && +$("resolution").value === modeResolution();
    const highResolutionClay = (["clay", "pseudocolor"].includes(state.renderMode)) && +$("resolution").value >= 1024;
    const rate = $("streamRate").value === 'auto' ? Math.min(state.autoRateCap,
      highResolutionClay || (comparisonView() && !defaultView) ? 30 : 60) : +$("streamRate").value;
    state.minimumId = id; state.dirty = false; state.delivered = [];
    const windowSize = state.scene.stream_window || 24;
    const bufferSeconds = Math.max(state.scene.buffer_seconds || .4, highResolutionClay ? .6 : 0);
    const warmup = Math.min(windowSize - 4, Math.ceil(rate * bufferSeconds));
    state.stream = {id, buffer: new GaussianPlayback(rate, warmup, windowSize), ack: -1,
      jpeg: new Map(), jpegBytes: 0, cachePending: new Set()};
    window.gaussianViewerStats = {presented: 0, underruns: 0, dropped: 0, buffered: 0,
      target_fps: rate, cache_hits: 0, client_cache_hits: 0, decode_ms: 0, draw_ms: 0};
    const {yaw, pitch, zoom, pan_x, pan_y, background} = state;
    state.socket.send(JSON.stringify({type: "stream", id, frame: state.frame, speed: state.speed,
      rate, loop: $("loop").checked, yaw, pitch, zoom, pan_x, pan_y, background, render_mode: state.renderMode, hybrid_quality: state.hybridQuality, clay_style: state.clayStyle,
      fovy: state.scene.fovy, quality: state.scene.default_stream_quality ?? 78, ...dimensions()}));
    $("streamStatus").textContent = "预缓冲…";
  }
  function present(item, close = true) {
    const {meta} = item, bitmap = item.bitmap.image || item.bitmap, now = performance.now();
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
    if (item.bitmap.reference) {
      const {width, height} = state.scene.reference, reference = item.bitmap.reference;
      for (const c of [rgbCanvas, eventCanvas]) {
        if (c.width !== width || c.height !== height) {c.width = width; c.height = height;}
        c.dataset.frame = String(meta.frame);
        c.dataset.time = String(state.scene.times_seconds[meta.frame]);
      }
      rgbContext.drawImage(reference, 0, 0, width, height, 0, 0, width, height);
      eventContext.drawImage(reference, width, 0, width, height, 0, 0, width, height);
      eventCanvas.dataset.sourceFrame = meta.frame === 0 ? "empty" : String(meta.frame - 1);
      text("referenceAlignment", meta.frame === 0 ? "对齐 RGB · 起始无事件" : "对齐 RGB · 同步");
    }
    if (close) item.bitmap.close();
    canvas.dataset.frame = String(meta.frame);
    canvas.dataset.requestId = meta.stream ? `${meta.id}:${meta.stream.seq}` : String(meta.id);
    state.displayedFrame = meta.frame; state.lastReceived = now;
    fitOverview(); refreshDetail();
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
      controls(!state.exporting); requestFrame();
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
          if(frameRequests.size){rejectFrameRequests(new Error(error.message||"渲染失败"));return;}
          setPlaying(false); status("渲染错误", "error");
          overlay("暂时无法渲染", error.message || "请检查服务器日志", true);
          state.dirty = false; state.pending.clear(); return;
        }
        const data = event.data;
        const length = new DataView(data).getUint32(0, true);
        if (length > 65536 || length + 4 > data.byteLength) throw new Error("Invalid image packet");
        const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, length)));
        responseId = meta.id;
        if(frameRequests.has(meta.id)) {
          const request=frameRequests.get(meta.id);frameRequests.delete(meta.id);clearTimeout(request.timer);
          if(length+4===data.byteLength)request.reject(new Error("缺少图像数据"));
          else request.resolve({meta,blob:new Blob([new Uint8Array(data,4+length)],{type:"image/jpeg"})});
          return;
        }
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
        let bitmap;
        if (references) {
          // Decode both sources before queueing this timestamp. No independent
          // HTML video clocks or seeking races; present all four atomically.
          const decoded = await Promise.allSettled([createImageBitmap(blob), references.frame(meta.frame)]);
          const failure = decoded.find(result => result.status === "rejected");
          if (failure) {
            decoded.forEach(result => {if (result.status === "fulfilled") result.value.close();});
            throw failure.reason;
          }
          const [image, reference] = decoded.map(result => result.value);
          bitmap = {image, reference, close() {image.close(); reference.close();}};
        } else bitmap = await createImageBitmap(blob);
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
        if (state.socket !== socket || responseId < state.minimumId) return;
        console.error("Viewer image error", error);
        overlay("画面解析失败", "请刷新页面重试。", true); setPlaying(false); state.dirty = false; state.pending.clear();
      } finally {
        if (state.socket === socket) state.pending.delete(responseId);
      }
    };
    socket.onclose = () => {
      if (state.socket !== socket) return;
      rejectFrameRequests(new Error("渲染连接中断；已完成缓存会保留"));
      if($("smoothPlayback").checked && readySequence()) {status("本地缓存可播放 · 服务器连接断开","ready");return;}
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
      if (firstLoad && ["mesh", "hybrid", "clay", "pseudocolor"].includes(launchMode) && scene[launchMode]?.available) state.renderMode = launchMode;
      if (["mesh", "hybrid", "clay", "pseudocolor"].includes(state.renderMode) && !scene[state.renderMode]?.available) state.renderMode = "gaussian";
      if (firstLoad) {
        state.clayStyle = launchOptions.get("clay_style") === "wire" ? "wire" : "solid";
        $("resolution").value = String(modeResolution());
        state.hybridQuality = ["surface", "native", "balanced", "detail"].includes(launchOptions.get("quality")) ?
          launchOptions.get("quality") : (scene.hybrid?.default_quality || "surface");
      }
      references = scene.reference?.available ? (references || new ViewerReferences(sceneQuery,160*1024**2)) : null;
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
        if (!stream.local && (buffer.last - stream.ack >= 4 || buffer.last === buffer.end)) {
          const retained = [...stream.cachePending].slice(0, buffer.limit);
          state.socket.send(JSON.stringify({type: "ack", id: stream.id, seq: buffer.last, cached_frames: retained}));
          retained.forEach(frame => stream.cachePending.delete(frame));
          stream.ack = buffer.last;
        }
        window.gaussianViewerStats.presented++;
      }
      if(stream.local)pumpLocal(stream);
      Object.assign(window.gaussianViewerStats, {underruns: buffer.underruns, dropped: buffer.dropped,
        buffered: buffer.frames.size});
      if ($("streamRate").value === 'auto' && buffer.rate === 60 && buffer.underruns >= 2) {
        state.autoRateCap = 30; state.dirty = true;
      }
      if (now - state.lastStreamStatus >= 250) {
        text("streamStatus", stream.local ? `${buffer.rate} FPS · 整段本地播放 · 解码缓存 ${buffer.frames.size} 帧` : buffer.buffering ? "网络缓冲…" :
          `${buffer.rate} FPS 同步流 · 缓存 ${buffer.frames.size} 帧 · 断流 ${buffer.underruns}`);
        state.lastStreamStatus = now;
      }
    } else if (!state.playing && !state.preparing && !state.exporting) {
      text("streamStatus", "逐帧查看");
    } else if (state.interacting) {
      text("streamStatus", "视角交互 · 松开恢复播放");
    }
    if (now - state.lastReceived > 1500) $("displayFps").textContent = "0.0";
    requestFrame(); requestAnimationFrame(tick);
  }
  $("smoothPlayback").onchange=()=>{setPlaying(false);state.dirty=true;};
  $("play").onclick = () => setPlaying(!(state.playing || state.preparing));
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
    if (!state.scene || !["gaussian", "mesh", "hybrid", "clay", "pseudocolor"].includes(mode) ||
        (["mesh", "hybrid", "clay", "pseudocolor"].includes(mode) && !state.scene[mode]?.available)) return;
    stopStream(); state.pending.clear();
    state.minimumId = ++state.id;
    const wasDefaultResolution = +$("resolution").value === modeResolution();
    state.renderMode = mode; state.autoRateCap = 60; state.dirty = true;
    if (wasDefaultResolution) $("resolution").value = String(modeResolution());
    state.displayedFrame = -1;
    $("snapshot").disabled = true;
    overlay("切换显示模式", mode === "pseudocolor" ? "正在渲染伪彩色…" : (mode === "clay") ? "正在渲染白模…" : mode === "hybrid" ? "正在渲染 Mesh＋Gaussian 动态效果…" : mode === "mesh" ? "正在渲染高斯重建网格…" : "正在渲染高斯…");
    updateModeUI(); labels();
  };
  $("clayWire").onchange = event => {
    if (state.renderMode !== "clay") return;
    stopStream(); state.pending.clear(); state.minimumId = ++state.id;
    state.clayStyle = event.target.checked ? "wire" : "solid";
    state.dirty = true; state.displayedFrame = -1;
    $("snapshot").disabled = true;
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
    if(state.exporting)return;
    if (event.target.closest("button") || event.target.closest(".overlay") || event.target.closest(".reference-cell") || event.target.closest(".compare-label")) return;
    if (detail.enabled && comparisonView()) {
      if (event.target === canvas || event.target === rightCanvas) startDetail(event,event.target);
      return;
    }
    stage.focus({preventScroll: true}); stage.setPointerCapture(event.pointerId);
    state.interacting = true; stopStream();
    pointer = {x: event.clientX, y: event.clientY, id: event.pointerId, pan: event.button === 2 || event.shiftKey};
  };
  stage.onpointermove = event => {
    if (detail.enabled && comparisonView()) {
      if (detail.drag) moveDetail(event);
      else if (event.target === canvas || event.target === rightCanvas) {
        const p=detailPoint(event,event.target),hit=p && hitDetail(p);
        const h=hit?.handle;
        event.target.style.cursor=!h?"crosshair":h==="move"?"move":`${h==="ne"||h==="sw"?"nesw":h==="nw"||h==="se"?"nwse":h==="n"||h==="s"?"ns":"ew"}-resize`;
      }
      return;
    }
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
    if (detail.drag?.id === event.pointerId) detail.drag = null;
    if (pointer?.id === event.pointerId) {pointer = null; state.interacting = false; state.dirty = true;}
  };
  stage.ondblclick = event => {if (!state.exporting && !(detail.enabled && comparisonView()) && !event.target.closest(".reference-cell")) reset();};
  stage.addEventListener("wheel", event => {
    if(state.exporting)return;
    if (event.target.closest(".reference-cell") || event.target.closest(".compare-label")) return;
    if (detail.enabled && comparisonView()) return;
    state.interacting = true; stopStream(); clearTimeout(wheelTimer);
    wheelTimer = setTimeout(() => {state.interacting = false; state.dirty = true;}, 180);
    event.preventDefault(); state.zoom = Math.max(.2, Math.min(8, state.zoom * Math.exp(event.deltaY * .001)));
    state.dirty = true;
  }, {passive: false});
  document.addEventListener("keydown", event => {
    if(state.exporting)return;
    if (event.target.closest("input,select,textarea,button,a") || !state.scene) return;
    if (event.code === "Space") {event.preventDefault(); setPlaying(!(state.playing || state.preparing));}
    if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
      event.preventDefault(); setPlaying(false); setFrame(state.frame + (event.code === "ArrowRight" ? 1 : -1));
    }
    if (event.code === "KeyR") reset();
  });
  document.addEventListener("visibilitychange", () => {if (document.hidden) setPlaying(false); else state.dirty = true;});
  // Square calibrated comparison renders do not depend on CSS panel size.
  // Don't restart their stream when surrounding text or layout changes size.
  new ResizeObserver(() => {if (!comparisonView()) state.dirty = true; refreshDetail();}).observe(stage);
  $("snapshot").onclick = () => {
    if (state.displayedFrame < 0) return;
    let snapshot = canvas;
    if (comparisonView()) {
      const sources=state.scene.reference?.available?[canvas,rightCanvas,rgbCanvas,eventCanvas]:[canvas,rightCanvas];
      const bounds=state.scene.reference?.available?overviewBounds():{top:0,bottom:1};
      const width=canvas.width,unit=width/$("leftRenderCell").getBoundingClientRect().width;
      const header=Math.round(28*unit),imageHeight=Math.round(canvas.height*(bounds.bottom-bounds.top));
      const topHeight=imageHeight+header,withDetails=detail.enabled && detail.regions.length>0,count=detail.regions.length*2;
      const totalWidth=width*sources.length,cropHeight=withDetails?detailImageHeight(totalWidth/count,imageHeight):0;
      snapshot=document.createElement("canvas");snapshot.width=totalWidth;snapshot.height=topHeight+(withDetails?1+header+cropHeight:0);
      const ctx=snapshot.getContext("2d");ctx.fillStyle="#fff";ctx.fillRect(0,0,snapshot.width,snapshot.height);
      const names=[$("leftCompareLabel").textContent,$("rightCompareLabel").textContent,"RGB 视频 · 固定相机","事件相机视频 · 对齐 RGB"];
      for(const [i,source] of sources.entries()) {
        const x=i*width;ctx.fillStyle="#263543";ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(names[i],x+10*unit,18*unit,width-95*unit);
        ctx.save();ctx.fillStyle="#677987";ctx.font=`${9*unit}px sans-serif`;ctx.textAlign="right";ctx.fillText(`${state.scene.times_seconds[state.displayedFrame].toFixed(3)} s`,x+width-10*unit,18*unit);ctx.restore();
        ctx.drawImage(source,0,bounds.top*source.height,source.width,(bounds.bottom-bounds.top)*source.height,x,header,width,imageHeight);
        if(detail.enabled && i<2){ctx.save();ctx.beginPath();ctx.rect(x,header,width,imageHeight);ctx.clip();ctx.translate(x,header);
          paintRegions(ctx,{x:0,y:-bounds.top*canvas.height,width,height:canvas.height},width/detailGeometry(source).width,false);ctx.restore();}
        if(i){ctx.fillStyle="#dfe4e8";ctx.fillRect(x,0,1,topHeight);}
      }
      if(withDetails) {
        ctx.fillStyle="#dfe4e8";ctx.fillRect(0,topHeight,totalWidth,1);
        for(const [i,r] of detail.regions.entries()) for(const [j,source] of [canvas,rightCanvas].entries()) {
          const index=i*2+j,x=Math.round(index*totalWidth/count),end=Math.round((index+1)*totalWidth/count),w=end-x;
          ctx.save();ctx.translate(x,topHeight+1);ctx.fillStyle=r.color;ctx.fillRect(0,0,w,3);
          ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(`${r.label} · ${j===0?"V1":"V3"}`,10*unit,18*unit,w-20*unit);
          ctx.translate(0,header);paintCrop(ctx,source,r,w,cropHeight);ctx.restore();
          if(index){ctx.fillStyle="#dfe4e8";ctx.fillRect(x,topHeight+1,1,header+cropHeight);}
        }
      }
    }
    snapshot.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = `${state.renderMode}-frame-${String(state.displayedFrame).padStart(4, "0")}.png`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, "image/png");
  };
  const videoModes=[['gaussian','Gaussian 高斯'],['mesh','V1/V3 纯 Mesh'],['clay','V1/V3 白模 · 清晰柔光'],['pseudocolor','V1/V3 伪彩色 · 柔光'],['hybrid','Mesh＋GS']];
  let exportURL=null;
  $('cancelExport').onclick=()=>{if(exportJob){exportJob.cancelled=true;text('exportStatus','正在取消；已缓存的帧会保留。');}};
  $('exportVideo').onclick=async()=>{
    if(exportJob || !state.scene)return;
    if(!window.VideoEncoder || !window.VideoFrame){text('exportStatus','请使用新版 Chrome 或 Edge 导出完整视频。');return;}
    const missing=videoModes.filter(([mode])=>mode!=='gaussian' && !state.scene[mode]?.available);
    if(missing.length){text('exportStatus',`当前物件尚未提供：${missing.map(([,name])=>name).join('、')}`);return;}
    setPlaying(false);state.preparing=false;state.exporting=true;
    const job={cancelled:false};exportJob=job;
    const controls=[...document.querySelectorAll('button,input,select')].filter(el=>el!==$('cancelExport')).map(el=>[el,el.disabled]);
    controls.forEach(([el])=>{el.disabled=true;});$('cancelExport').hidden=false;$('exportDownload').hidden=true;
    const regions=detail.enabled?detail.regions.map(r=>({...r})):[];
    const valid=()=>exportJob===job && !job.cancelled;
    const check=()=>{if(!valid())throw new Error('已取消导出');};
    const report=message=>{text('exportStatus',message);};
    const settings=videoModes.map(([mode])=>({...sequenceSettings(mode),width:1024,height:1024,quality:95,clay_style:'solid'}));
    const scene=state.scene,count=scene.frame_count,fps=scene.fps;
    const sample=document.createElement('canvas');sample.width=128;sample.height=128;
    const sampleContext=sample.getContext('2d',{willReadFrequently:true});let first=128,last=-1;
    const measure=image=>{
      if(state.background!=='light')return;
      sampleContext.clearRect(0,0,128,128);sampleContext.drawImage(image,0,0,128,128);
      const pixels=sampleContext.getImageData(0,0,128,128).data;
      for(let y=0;y<128;y++){let hits=0;for(let x=0;x<128;x++){const i=(y*128+x)*4;if(pixels[i+3]>128 && Math.min(pixels[i],pixels[i+1],pixels[i+2])<235)hits++;}
        if(hits>=2){first=Math.min(first,y);last=Math.max(last,y);}}
    };
    const files=[];let writer=null,store=null;
    try {
      store=await GaussianPlayback.Media.ExportFrameStore.open();job.store=store;
      if(references)await references.prepare(count,(done,total)=>{if(valid())report(`准备 RGB / 事件 ${done}/${total}`);},valid);check();
      // Analyze the complete motion across all modes before fixing video size.
      // This common envelope prevents frame/canvas resizing and clipped motion.
      for(let modeIndex=0;modeIndex<settings.length;modeIndex++) {
        const name=videoModes[modeIndex][1];report(`准备 ${name} · 完整序列`);
        const entry=await prepareSequence(settings[modeIndex],valid,(done,total)=>report(`准备 ${name} ${done}/${total}`));
        await store.save(videoModes[modeIndex][0],entry);
        for(let frame=0;frame<count;frame++) {
          check();const bitmap=await createImageBitmap(entry.frames.get(frame).blob);
          try{measure(bitmap);}finally{bitmap.close();}
          if(modeIndex===0 && references){const ref=await references.frame(frame);try{measure(ref);}finally{ref.close();}}
          if(frame%40===0){report(`分析完整运动范围 · ${modeIndex+1}/5 · ${frame+1}/${count}`);await new Promise(resolve=>setTimeout(resolve,0));}
        }
      }
      const bounds=last<first?{top:0,bottom:1}:{top:Math.max(0,(first-8)/128),bottom:Math.min(1,(last+9)/128)};
      for(const r of regions){bounds.top=Math.max(0,Math.min(bounds.top,r.y-.035));bounds.bottom=Math.min(1,Math.max(bounds.bottom,r.y+r.h+.02));}
      const width=2560,columns=references?4:2,panel=width/columns,unit=panel/$('leftRenderCell').getBoundingClientRect().width;
      const header=Math.round(28*unit),imageHeight=Math.round(panel*(bounds.bottom-bounds.top));
      const detailWidth=regions.length?width/(regions.length*2):0;
      const pad=Math.min(12,Math.max(4,detailWidth*.025));
      const detailHeight=regions.length?Math.ceil(Math.max(2*pad+24,Math.min(Math.max(...regions.map(r=>r.h/r.w))*(detailWidth-2*pad)+2*pad,detailWidth*1.6,imageHeight*1.15))):0;
      const height=Math.ceil((header+imageHeight+(regions.length?1+header+detailHeight:0))/2)*2;
      const output=document.createElement('canvas');output.width=width;output.height=height;const ctx=output.getContext('2d',{alpha:false});
      const sources=Array.from({length:columns},()=>{const c=document.createElement('canvas');c.width=1024;c.height=1024;return c;});
      const start=scene.times_seconds[0],timestamps=scene.times_seconds.map(t=>Math.round((t-start)*1e6));
      const durations=timestamps.map((t,i)=>i+1<count?timestamps[i+1]-t:Math.round(1e6/fps));
      const duration=timestamps[count-1]+durations[count-1];
      for(let modeIndex=0;modeIndex<settings.length;modeIndex++) {
        check();const [mode,name]=videoModes[modeIndex];
        writer=await GaussianPlayback.Media.encoder(width,height,fps);
        for(let frame=0;frame<count;frame++) {
          check();const record=await store.get(mode,frame),bitmap=await decodeRecord(record);
          try {
            if(!record.meta.comparison)throw new Error(`${name} 未提供 V1/V3 成对画面`);
            const image=bitmap.image||bitmap,pw=record.meta.panel_width,ph=record.meta.height;
            sources[0].getContext('2d').drawImage(image,0,0,pw,ph,0,0,1024,1024);
            sources[1].getContext('2d').drawImage(image,pw,0,pw,ph,0,0,1024,1024);
            if(references){const rw=scene.reference.width,rh=scene.reference.height;
              sources[2].getContext('2d').drawImage(bitmap.reference,0,0,rw,rh,0,0,1024,1024);
              sources[3].getContext('2d').drawImage(bitmap.reference,rw,0,rw,rh,0,0,1024,1024);}
          }finally{bitmap.close();}
          ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);
          const titles=[`V1 · ${name.replace('V1/V3 ','')}`,`V3 · ${name.replace('V1/V3 ','')}`,'RGB 视频','事件相机视频 · 对齐 RGB'];
          for(const [i,source] of sources.entries()){
            const x=i*panel;ctx.fillStyle='#263543';ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(titles[i],x+10*unit,18*unit,panel-95*unit);
            ctx.save();ctx.fillStyle='#677987';ctx.font=`${9*unit}px sans-serif`;ctx.textAlign='right';ctx.fillText(`${scene.times_seconds[frame].toFixed(3)} s`,x+panel-10*unit,18*unit);ctx.restore();
            ctx.drawImage(source,0,bounds.top*1024,1024,(bounds.bottom-bounds.top)*1024,x,header,panel,imageHeight);
            if(i<2 && regions.length){ctx.save();ctx.beginPath();ctx.rect(x,header,panel,imageHeight);ctx.clip();ctx.translate(x,header);
              paintRegions(ctx,{x:0,y:-bounds.top*panel,width:panel,height:panel},unit,false);ctx.restore();}
            if(i){ctx.fillStyle='#dfe4e8';ctx.fillRect(x,0,1,header+imageHeight);}
          }
          if(regions.length){const y=header+imageHeight+1;ctx.fillStyle='#dfe4e8';ctx.fillRect(0,y-1,width,1);
            for(const [i,r] of regions.entries())for(let j=0;j<2;j++){
              const x=Math.round((i*2+j)*detailWidth),end=Math.round((i*2+j+1)*detailWidth),w=end-x;
              ctx.save();ctx.translate(x,y);ctx.fillStyle=r.color;ctx.fillRect(0,0,w,2);ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(`${r.label} · ${j===0?'V1':'V3'}`,10*unit,18*unit,w-20*unit);
              ctx.translate(0,header);paintCrop(ctx,sources[j],r,w,detailHeight);ctx.restore();
              if(i*2+j){ctx.fillStyle='#dfe4e8';ctx.fillRect(x,y,1,header+detailHeight);}
            }
          }
          await writer.add(output,timestamps[frame],durations[frame]);
          if(frame%5===0)report(`编码 ${modeIndex+1}/5 · ${name} · ${frame+1}/${count}`);
        }
        check();const blob=await writer.finish(duration);writer.close();writer=null;
        files.push({name:`${modeIndex+1}_${mode}.webm`,blob});
      }
      check();files.push({name:'manifest.json',blob:new Blob([JSON.stringify({scene:scene.name,frame_count:count,fps,duration_seconds:duration/1e6,
        width,height,overview:bounds,regions,camera:settings[0],modes:videoModes.map(([id,label])=>({id,label})),times_seconds:scene.times_seconds},null,2)],{type:'application/json'})});
      report('正在打包五种完整视频…');const archive=await GaussianPlayback.Media.zip(files);check();
      if(exportURL)URL.revokeObjectURL(exportURL);exportURL=URL.createObjectURL(archive);
      const link=$('exportDownload');link.href=exportURL;link.download=`scgs-${$('objectSelect').value||'scene'}-all-modes.zip`;link.hidden=false;link.click();
      report(`完成：5 个完整视频 · 每个 ${count} 帧 · ${(duration/1e6).toFixed(2)} 秒 · ${(archive.size/1024**2).toFixed(1)} MiB`);
    }catch(error){report(job.cancelled?'已取消导出；已缓存的帧可继续复用。':`导出失败：${error.message}`);console.error('Video export',error);}
    finally{writer?.close();store?.close();controls.forEach(([el,disabled])=>{el.disabled=disabled;});$('cancelExport').hidden=true;state.exporting=false;exportJob=null;state.dirty=true;}
  };

  window.addEventListener("beforeunload", () => {exportJob?.store?.close();state.socket?.close();});
  reset(); poll(); requestAnimationFrame(tick);
})();
