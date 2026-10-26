"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  // Avoid replacing identical live text on every animation frame.
  const text = (id, value) => {if ($(id).textContent !== value) $(id).textContent = value;};
  const canvas = $("canvas"), stage = $("stage"), context = canvas.getContext("2d", {alpha: false});
  const rightCanvas = $("comparisonCanvas"), rightContext = rightCanvas.getContext("2d", {alpha: false});
  const middleCanvas = $("middleCanvas");
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
  const detailByView=new Map();
  function selectDetailView(key) {
    if(detail.scope===key)return;
    if(detail.scope)detailByView.set(detail.scope,{regions:detail.regions,selected:detail.selected,next:detail.next});
    const saved=detailByView.get(key)||{regions:[],selected:null,next:0};
    Object.assign(detail,saved,{scope:key,drag:null});
    rebuildDetail();
  }
  const allRenderCanvases = [canvas,middleCanvas,rightCanvas];
  function versionLabels(mode=state.renderMode) {
    return state.scene?.mode_version_labels?.[mode] || state.scene?.version_labels || [state.scene?.left_label,state.scene?.right_label];
  }
  function renderCanvases(mode=state.renderMode) {return versionLabels(mode).length===3 ? allRenderCanvases : [canvas,rightCanvas];}
  function detailPairs() {return renderCanvases().map(source=>[source,$(source===canvas?'detailLeft':source===middleCanvas?'detailMiddle':'detailRight')]);}
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
    for(const source of [...renderCanvases(),rgbCanvas,eventCanvas]) {
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
      for(const source of [...renderCanvases(),rgbCanvas,eventCanvas]) {source.style.height="";source.style.top="";}
      return;
    }
    const {top,bottom}=overviewBounds(),width=$("leftRenderCell").getBoundingClientRect().width;
    if(!width || !canvas.width || !canvas.height) return;
    const fullHeight=width*canvas.height/canvas.width;
    const header=$("leftCompareLabel").getBoundingClientRect().height || 28;
    stage.style.setProperty("--overview-height",`${header+fullHeight*(bottom-top)}px`);
    for(const source of [...renderCanvases(),rgbCanvas,eventCanvas]) {
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
      for(const [i,source] of renderCanvases().entries()) {
        const cell=document.createElement("section"),title=document.createElement("div"),area=document.createElement("div"),view=document.createElement("canvas");
        cell.className="detail-cell";cell.style.setProperty("--roi-color",r.color);cell.dataset.region=r.label;
        title.className="detail-title";title.textContent=`${r.label} · ${comparisonStages()[i]}`;
        area.className="detail-image";view.setAttribute("aria-label",`${r.label} 选区 ${comparisonStages()[i]} 动态细节`);
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
    for(const [source,layer] of detailPairs()) {
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
      ? `选区：拖动框内移动 · 拖动边框 / 角点调整大小 · 下排同步对比 ${comparisonName()}`
      : comparisonView() && state.scene?.reference?.available
        ? `${comparisonName()}：拖动旋转 · Shift 平移 · 滚轮缩放；RGB / 事件：固定相机视角`
        : "拖动旋转 · Shift 平移 · 滚轮缩放";
  }
  $("detailToggle").onclick=()=>{
    detail.enabled=!detail.enabled;detail.drag=null;
    for(const source of allRenderCanvases)source.style.cursor="";
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
  function requestQuality(fallback) {
    const value=Number(launchOptions.get('jpeg_quality'));
    return Number.isInteger(value) && value>=60 && value<=95 ? value : fallback;
  }
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
  const presetViews=new GaussianPlayback.Media.PresetViews();
  const presetButtons=[...document.querySelectorAll('.preset')];
  let viewPreparation=null,preparedViewGroup=null,failedViewGroup=null;
  const frameRequests=new Map();
  let exportJob=null;
  function sequenceSettings(mode=state.renderMode) {
    const {yaw,pitch,zoom,pan_x,pan_y,background}=state;
    return {yaw,pitch,zoom,pan_x,pan_y,background,render_mode:mode,hybrid_quality:state.hybridQuality,
      clay_style:state.clayStyle,fovy:state.scene.fovy,
      quality:requestQuality(state.scene.default_stream_quality??95),...dimensions()};
  }
  function sequenceKey(settings){return JSON.stringify(settings);}
  let experimentPlan=null, experimentBusy=false;
  const experimentButtons=['experimentAll','experimentFrame','experimentPlanSave','experimentSequence','experimentBenchmark'];
  function experimentControls() {
    for(const id of experimentButtons) $(id).disabled=experimentBusy || !state.scene ||
      (['experimentSequence','experimentBenchmark'].includes(id) && !experimentPlan);
  }
  function experimentProgress(job) {
    const finished=job.status==='complete';
    const stagePercent=job.total>0 ? Math.min(100,job.completed/job.total*100) : 0;
    const percent=finished ? 100 : Math.max(0,Math.min(99.9,Number.isFinite(job.percent) ? job.percent : stagePercent));
    $('experimentProgress').hidden=false;
    $('experimentProgress').dataset.state=job.status;
    $('experimentProgressBar').value=percent;
    text('experimentProgressLabel',Number.isFinite(job.percent) || finished ? '整体流程进度' : '当前阶段进度');
    text('experimentPercent',`${Number(percent.toFixed(1))}%`);
    text('experimentStage',`${job.stage_index && !finished ? `阶段 ${job.stage_index}/${job.stage_count} · ` : ''}${job.message}`);
    const bytes=value=>{
      const units=['B','KiB','MiB','GiB','TiB'];let index=0;
      while(value>=1024 && index<units.length-1){value/=1024;index++;}
      return `${value.toFixed(index ? 1 : 0)} ${units[index]}`;
    };
    const unit={frames:'帧',renders:'次',versions:'个版本',files:'个文件',items:'项'}[job.unit] || '项';
    const count=job.unit==='bytes' ? `${bytes(job.completed)} / ${bytes(job.total)}` : `${job.completed} / ${job.total} ${unit}`;
    text('experimentDetail',finished ? '文件包已生成，浏览器下载进度请查看下载列表。' :
      job.stage==='queued' ? '等待任务开始…' : `${count}${job.detail ? ` · ${job.detail}` : ''}`);
    $('experimentProgressBar').setAttribute('aria-valuetext',`${percent}% · ${job.message}`);
  }
  function experimentSamples(current=false) {
    if(!state.scene) throw new Error('场景尚未就绪');
    const first=current ? (state.displayedFrame>=0 ? state.displayedFrame : state.frame) : Number($('experimentStart').value);
    const last=current ? first : ($('experimentEnd').value==='' ? state.scene.frame_count-1 : Number($('experimentEnd').value));
    const step=current ? 1 : Number($('experimentStep').value);
    if(![first,last,step].every(Number.isInteger) || first<0 || last<first || last>=state.scene.frame_count || step<1)
      throw new Error('请输入有效的起止帧和采样间隔');
    const settings=sequenceSettings(), samples=[];
    for(let frame=first;frame<=last;frame+=step) samples.push({...settings,frame,time_seconds:state.scene.times_seconds[frame]});
    return samples;
  }
  function adoptExperimentPlan(plan) {
    if(plan.format!=='scgs_experiment_plan_v1' || plan.scene!==$('objectSelect').value ||
       !Array.isArray(plan.samples) || !plan.samples.length || plan.samples.length>10000)
      throw new Error('采样表格式或物件不匹配');
    for(const sample of plan.samples) {
      if(!Number.isInteger(sample.frame) || sample.frame<0 || sample.frame>=state.scene.frame_count ||
         !Number.isFinite(sample.time_seconds) || Math.abs(sample.time_seconds-state.scene.times_seconds[sample.frame])>1e-8)
        throw new Error('采样表时间与模型不匹配');
    }
    experimentPlan=plan;
    text('experimentPlanStatus',`已固定 ${plan.samples.length} 个采样点及对应相机；修改视角不会改变该表。`);
    experimentControls();
  }
  $('experimentPlanSave').onclick=async()=>{
    try {
      const response=await fetch(`/api/experiments/plan${sceneQuery}`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({operation:'sequence',samples:experimentSamples()})});
      if(!response.ok)throw new Error(await response.text());
      const plan=await response.json();
      adoptExperimentPlan(plan);
      const url=URL.createObjectURL(new Blob([JSON.stringify(plan,null,2)],{type:'application/json'}));
      const link=document.createElement('a');link.href=url;link.download=`${plan.scene}-experiment-plan.json`;link.click();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    } catch(error) {text('experimentStatus',error.message);}
  };
  $('experimentPlanFile').onchange=async()=>{
    try {
      const file=$('experimentPlanFile').files[0];if(!file)return;
      if(file.size>8*1024*1024)throw new Error('采样表过大');
      adoptExperimentPlan(JSON.parse(await file.text()));
    } catch(error) {text('experimentStatus',error.message);}
  };
  async function startExperiment(operation) {
    if(experimentBusy)return;
    try {
      if(viewPreparation || exportJob)throw new Error('请等待当前预渲染或视频导出完成');
      if(!state.scene)throw new Error('场景尚未就绪');
      setPlaying(false);
      const complete=operation==='complete';
      if(complete && (!Number.isInteger(Number($('experimentStep').value)) || Number($('experimentStep').value)<1 || Number($('experimentStep').value)>10000))
        throw new Error('请输入有效采样间隔（1–10000）');
      const samples=complete ? null : operation==='frame' ? experimentSamples(true) : experimentPlan?.samples;
      if(!complete && !samples)throw new Error('请先保存或载入采样表');
      experimentBusy=true;experimentControls();$('experimentDownload').hidden=true;
      $('experimentLocation').hidden=true;
      experimentProgress({status:'running',stage:'queued',message:'正在提交导出任务…',percent:0,completed:0,total:1});
      text('experimentStatus','正在提交导出任务…');
      const selection=complete ? {sample_step:Number($('experimentStep').value || state.scene.export_sample_step || 4),settings:{...sequenceSettings(),frame:state.displayedFrame>=0 ? state.displayedFrame : state.frame}} : {samples};
      const measurement=operation==='benchmark' ? {warmup:Number($('experimentWarmup').value || 20),repeats:Number($('experimentRepeats').value || 5)} : {};
      const response=await fetch(`/api/experiments${sceneQuery}`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({operation,...selection,...measurement})});
      if(!response.ok)throw new Error(await response.text());
      let job=await response.json();
      while(job.status==='running') {
        experimentProgress(job);
        text('experimentStatus',`${$('experimentStage').textContent} · ${$('experimentDetail').textContent}`);
        await new Promise(resolve=>setTimeout(resolve,1000));
        const status=await fetch(`/api/experiments/${job.id}${sceneQuery}`);
        if(!status.ok)throw new Error(await status.text());
        job=await status.json();
      }
      experimentProgress(job);
      if(job.status!=='complete')throw new Error(job.message);
      $('experimentDownload').href=`/api/experiments/${job.id}/download${sceneQuery}`;
      $('experimentDownload').hidden=false;
      if(job.output_archive || job.output_directory) {
        text('experimentLocation',job.output_archive ? `服务器仅保留 ZIP：${job.output_archive}（${(job.output_bytes/1024**2).toFixed(1)} MiB）` : `渲染服务器保存目录：${job.output_directory}`);
        $('experimentLocation').hidden=false;
      }
      text('experimentStatus',complete ? `关键数据已保存${job.sampled_frames ? `：每 ${job.sample_step || 4} 帧采样，共 ${job.sampled_frames} 帧。` : '。'}${job.missing_data?.length ? '缺失或未对齐的数据已列入 data_availability.json。' : ''}解压后运行 python evaluate.py 进行独立评测。` : '导出完成。下载并解压 ZIP 后，运行独立评测脚本。');
      if(complete) $('experimentDownload').click();
    } catch(error) {
      text('experimentStatus',error.message);
      if(experimentBusy) {
        if($('experimentProgress').dataset.state!=='failed')text('experimentStage',error.message);
        $('experimentProgress').dataset.state='failed';
      }
    }
    finally {
      experimentBusy=false;experimentControls();state.dirty=true;
      if(state.socket?.readyState===3)scheduleRetry();
    }
  }
  $('experimentAll').onclick=()=>startExperiment('complete');
  $('experimentFrame').onclick=()=>startExperiment('frame');
  $('experimentSequence').onclick=()=>startExperiment('sequence');
  $('experimentBenchmark').onclick=()=>startExperiment('benchmark');
  function readySequence(){if(!state.scene)return null;const key=sequenceKey(sequenceSettings());const e=presetViews.get(key)||sequences.get(key);return e?.complete && (!references || references.prepared)?e:null;}
  async function readRecord(entry,frame){return entry.read?entry.read(frame):entry.frames.get(frame);}
  function viewGroup() {
    const base={...sequenceSettings(),yaw:0,pitch:0,zoom:1,pan_x:0,pan_y:0};
    return {key:sequenceKey(base),views:presetButtons.map(button=>({name:button.textContent,
      settings:{...base,yaw:+button.dataset.yaw,pitch:+button.dataset.pitch}}))};
  }
  function preparationUI(busy) {
    stage.classList.toggle('preparing-views',busy);
    $('viewerPanel').classList.toggle('waiting-view',busy);
    $('detailRow').style.visibility=busy?'hidden':'';
    stage.setAttribute('aria-busy',String(busy));
  }
  async function prepareAllViews(group) {
    const job={key:group.key,resume:state.playing||autoStartPending};viewPreparation=job;
    presetViews.activate(group.views.map(view=>sequenceKey(view.settings)));
    autoStartPending=false;setPlaying(false);state.dirty=false;
    state.minimumId=++state.id;state.pending.clear();
    const locked=[...document.querySelectorAll('button,input,select')]
      .filter(el=>el!==$('objectSelect')).map(el=>[el,el.disabled]);
    locked.forEach(([el])=>{el.disabled=true;});preparationUI(true);
    $('retryViews').hidden=true;
    const valid=()=>viewPreparation===job && group.key===viewGroup().key;
    const count=state.scene.frame_count,total=count*(group.views.length+(references?1:0));
    let finished=0;
    const viewStates=group.views.map(view=>presetViews.get(sequenceKey(view.settings))?'ready':'waiting');
    const renderViewStates=()=>group.views.forEach((view,i)=>{
      const el=$(`loadView${i}`),labels={waiting:'等待',rendering:'渲染中',saving:'缓存中',ready:'就绪'};
      el.dataset.state=viewStates[i];el.textContent=`${view.name} · ${labels[viewStates[i]]}`;
    });
    renderViewStates();
    const report=(name,done=0)=>{
      if(!valid())return;
      const progress=Math.min(99.9,(finished+done)/total*100);
      overlay('渲染中',`${name} · ${Math.floor(progress)}% · 全部六个视角完成后展示`);
      loadingProgress(progress);
      text('streamStatus',`渲染中 · ${Math.floor(progress)}%`);
    };
    report('准备完整动画');
    try {
      if(references){await references.prepare(count,(done)=>report('RGB / 事件视频',done),valid);finished+=count;}
      for(const [index,view] of group.views.entries()) {
        if(!valid())throw new Error('画面设置已变化');
        const key=sequenceKey(view.settings);
        if(!presetViews.get(key)) {
          viewStates[index]='rendering';renderViewStates();
          const entry=await prepareSequence(view.settings,valid,done=>report(`${view.name} ${index+1}/6`,done));
          viewStates[index]='saving';renderViewStates();
          report(`${view.name} · 保存本地缓存`,count);
          await presetViews.save(entry);
          sequences.remove(key);
        }
        viewStates[index]='ready';renderViewStates();
        finished+=count;report(`${view.name}已就绪`);
      }
      if(!valid())throw new Error('画面设置已变化');
      preparedViewGroup=group.key;failedViewGroup=null;
      loadingProgress(100);
      text('loadMessage','六个视角已就绪 · 100% · 正在显示画面');
      text('streamStatus','六个视角已就绪 · 本地播放');
      state.dirty=true;
    }catch(error){
      failedViewGroup=group.key;
      overlay('渲染未完成',`${error.message}。已完成的视角会保留，请重试。`,true);
      $('retryViews').hidden=false;state.dirty=false;
    }finally{
      viewPreparation=null;locked.forEach(([el,disabled])=>{el.disabled=disabled;});
      if(preparedViewGroup===group.key){
        controls(true);
        if(job.resume){$('loop').checked ||= launchOptions.get('loop')==='1';setPlaying(true);}
      }else controls(false);
    }
  }
  $('retryViews').onclick=()=>{failedViewGroup=null;state.dirty=true;};
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
    const saved=presetViews.get(entry.key);
    if(!saved)await ensureConnection(valid);
    let next=0,failure=null;
    await Promise.all(Array.from({length:4},async()=>{
      while(!failure && valid() && next<entry.count) {
        const index=next++;if(entry.frames.has(index))continue;
        try {const record=saved?await saved.read(index):await fetchFrame(settings,index);if(!valid())break;sequences.put(entry,index,record);if(valid())progress(entry.frames.size,entry.count);}
        catch(error){failure=error;}
      }
    }));
    if(failure)throw failure;if(!valid())throw new Error('已取消');
    if(!entry.complete)throw new Error('整段缓存未完成，请重试');return entry;
  }
  async function decodeRecord(record) {
    const decoded=await Promise.allSettled([createImageBitmap(GaussianPlayback.Media.recordBlob(record)),...(references?[references.frame(record.meta.frame)]:[])]);
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
        const record=await readRecord(stream.entry,frame);if(!record)throw new Error('本地缓存帧不完整');
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

  function comparisonStages(mode=state.renderMode) {
    return versionLabels(mode).map((label,i)=>String(label || "").match(/^V\d+/)?.[0] || (i===0?"V1":"V3"));
  }
  function comparisonName() {return comparisonStages().join(" / ");}
  function comparisonView() {
    return !!state.scene?.comparison || (state.scene?.comparison_modes || []).includes(state.renderMode);
  }
  function modeResolution(mode = state.renderMode) {
    const specified=Number(launchOptions.get('resolution'));
    if([512,640,768,1024,1280].includes(specified))return specified;
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
    $("clayOption").textContent = `${comparisonStages('clay').join(' / ')} 白模 · 清晰柔光`;
    $("pseudocolorOption").textContent = `${comparisonStages('pseudocolor').join(' / ')} 伪彩色 · 柔光`;
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
    const triplet = paired && versionLabels().length===3;
    stage.classList.toggle('triplet',triplet);
    middleCanvas.hidden = $('middleRenderCell').hidden = $('middleCompareLabel').hidden = !triplet;
    if(!triplet)$('detailMiddle').hidden=true;
    $('middleCompareLabel').textContent=versionLabels()[1] || 'V2';
    middleCanvas.setAttribute('aria-label','V2 动态渲染，与 V1、V3 同相机同时间');
    const detailLayout=renderCanvases().length;
    if(detail.layout!==detailLayout){detail.layout=detailLayout;rebuildDetail();}
    $("detailTools").hidden = !paired;
    refreshDetail();
    $("expandView").textContent = paired ? "放大对比" : "放大画面";
    $("expandView").title = paired ? "全屏显示同步版本视窗" : "全屏显示当前物件";
    rightCanvas.hidden = $("rightRenderCell").hidden = !paired;
    const quad = !!scene.reference?.available && paired;
    stage.classList.toggle("quad", quad);
    stage.style.setProperty('--overview-columns',renderCanvases().length+(quad?2:0));
    $("rgbCell").hidden = $("eventCell").hidden = !quad;
    if (quad) {
      $("referenceAlignment").textContent = "对齐 RGB · 同步";
      $("referenceAlignment").title = "与 RGB 使用相同像素坐标；事件区间结束时刻对应当前 RGB 帧，首帧无前序事件。";
      $("viewHint").textContent = `${comparisonName()}：拖动旋转 · Shift 平移 · 滚轮缩放；RGB / 事件：固定相机视角`;
      $("expandView").textContent = triplet ? "放大五列对比" : "放大四列对比";
      $("expandView").title = triplet ? "全屏显示五个同步视窗" : "全屏显示四个同步视窗";
    }
    updateDetailHint();
    $("leftCompareLabel").hidden = $("rightCompareLabel").hidden = !paired;
    $("sceneKind").textContent = color ? "PSEUDOCOLOR" : clay ? "CLAY SURFACE" : hybrid ? "MESH + GAUSSIAN" : mesh ? "MESH SURFACE" : "GAUSSIAN SCENE";
    $("sceneName").textContent = color ? `${comparisonName()} 伪彩色对比` : clay ? `${comparisonName()} 白模对比` : hybrid ? `${comparisonName()} 混合动态对比` : mesh ? `${comparisonName()} Mesh 对比` : scene.name;
    $("leftCompareLabel").textContent = clay || color || mesh || hybrid ? (modeInfo.left_label || "V1 Mesh · 原始结果") : scene.left_label;
    $("rightCompareLabel").textContent = clay || color || mesh || hybrid ? (modeInfo.right_label || "V3 Mesh · 最新优化结果") : scene.right_label;
    if(triplet && !/^V1\b/.test($('leftCompareLabel').textContent || ''))
      $('leftCompareLabel').textContent=['V1',$('leftCompareLabel').textContent].filter(Boolean).join(' · ');
    $("sequenceBadge").textContent = color ? "伪彩色 · 同帧同视角" : clay ? "白模 · 同帧同视角" : hybrid ? "Mesh＋Gaussian · 同帧同视角" : mesh ? `${comparisonName()} Mesh · 同帧同视角` : paired ?
      (triplet ? "V1 / V2 / V3 · 同帧同视角" : "原始 / 优化 · 同帧同视角") : scene.event_validation ? "原始事件校正 · 候选预览" : "逐帧高斯 · 实时渲染";
    $("gaussianCount").textContent = (mesh ? modeInfo.triangle_count : scene.gaussian_count).toLocaleString("en-US");
    $("primitiveLabel").textContent = mesh ? "TRIANGLES / 三角面数量" : "GAUSSIANS / 高斯数量";
    $("sceneDescription").textContent = clay || color || mesh || hybrid ? modeDescription :
      (scene.description || "真实高斯序列，随时间更新位置与朝向。改变视角时重新进行 3D 渲染。");
    $("snapshot").textContent = color ? `保存 ${comparisonName()} 伪彩色对比图 ↓` : clay ? `保存 ${comparisonName()} 白模对比图 ↓` : hybrid ? "保存 Mesh＋Gaussian 动态对比图 ↓" : mesh ? `保存 ${comparisonName()} Mesh 对比图 ↓` : paired ? "保存原始 / 优化对比图 ↓" : "保存当前画面 ↓";
    if (quad) $("snapshot").textContent = `保存 1×${renderCanvases().length+2} 对比图 ↓`;
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
  function loadingProgress(value) {
    const percent=clamp(Number(value)||0,0,100);
    $('loadProgress').value=percent;text('loadPercent',`${Math.floor(percent)}%`);
  }
  function overlay(title, message, error = false) {
    $('loadViews').hidden=!viewPreparation;
    $("overlay").hidden = false;
    $("overlay").classList.toggle("error", error);
    $("loadTitle").textContent = title;
    $("loadMessage").textContent = message;
  }
  function controls(enabled) {
    ["play", "previous", "next", "restart", "timeline", "snapshot"].forEach(id => $(id).disabled = !enabled);
    experimentControls();
    if(!enabled) for(const id of experimentButtons) $(id).disabled=true;
  }
  function setPlaying(value) {
    if(viewPreparation && value)return;
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
    stopStream();
    selectDetailView(`preset:${yaw}:${pitch}`);
    Object.assign(state, {yaw, pitch, zoom: 1, pan_x: 0, pan_y: 0, dirty: true});
    document.querySelectorAll(".preset").forEach(button => button.classList.toggle("active",
      +button.dataset.yaw === yaw && +button.dataset.pitch === pitch));
    const viewName = yaw === 0 && pitch === 0 ? "正视" : yaw === 180 && pitch === 0 ? "背面" :
      yaw === 0 && pitch === 35 ? "俯视" : yaw === 0 && pitch === -35 ? "仰视" : yaw === 35 && pitch === 20 ? "斜视" : yaw === 80 && pitch === 0 ? "侧视" : "自由视角";
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
    if(experimentBusy)return;
    if(viewPreparation)return;
    if(state.scene && !state.exporting && $('smoothPlayback').checked) {
      const group=viewGroup();
      if(preparedViewGroup!==group.key) {
        if(failedViewGroup!==group.key)prepareAllViews(group);
        return;
      }
    }
    if(state.exporting || state.preparing)return;
    if(!state.playing && state.dirty && $("smoothPlayback").checked && readySequence() && !state.localStill) {
      const entry=readySequence(),id=++state.id,frame=state.frame;state.localStill=true;state.dirty=false;
      readRecord(entry,frame).then(async record=>({record,bitmap:await decodeRecord(record)}))
        .then(({record,bitmap})=>{if(!state.playing && !state.dirty && sequenceKey(sequenceSettings())===entry.key)present({meta:{...record.meta,id,client_cache_hit:true},bitmap});else bitmap.close();})
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
      quality: requestQuality(state.interacting ? 72 : (state.scene.default_still_quality ?? 85)), ...dimensions()}));
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
      fovy: state.scene.fovy, quality: requestQuality(state.scene.default_stream_quality ?? 78), ...dimensions()}));
    $("streamStatus").textContent = "预缓冲…";
  }
  function present(item, close = true) {
    if(viewPreparation){if(close)item.bitmap.close();return;}
    const {meta} = item, bitmap = item.bitmap.image || item.bitmap, now = performance.now();
    $("snapshot").disabled = false;
    const panelWidth = meta.comparison ? meta.panel_width : meta.width;
    if (canvas.width !== panelWidth || canvas.height !== meta.height) {
      canvas.width = panelWidth; canvas.height = meta.height;
    }
    if (meta.comparison) {
      const targets=renderCanvases(meta.render_mode || state.renderMode);
      if((meta.panel_count || 2)!==targets.length)throw new Error('版本数量与渲染帧不一致');
      for(const [i,target] of targets.entries()){
        if(target.width!==panelWidth || target.height!==meta.height){target.width=panelWidth;target.height=meta.height;}
        target.getContext('2d').drawImage(bitmap,i*panelWidth,0,panelWidth,meta.height,0,0,panelWidth,meta.height);
        target.dataset.frame=String(meta.panel_frames?.[i] ?? meta.frame);
      }
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
    if (!$("overlay").hidden && !viewPreparation) {$("overlay").hidden = true; preparationUI(false);status("已连接", "ready");}
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
      controls(!state.exporting && !viewPreparation); requestFrame();
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
          else request.resolve({meta,bytes:data.slice(4+length),type:"image/jpeg"});
          return;
        }
        // Late replies to expired preparation/export requests must never be
        // mistaken for a visible still, especially while a failed group is hidden.
        if(!meta.stream && !state.pending.has(meta.id))return;
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
      if(experimentBusy){state.pending.clear();status('导出进行中 · 预览稍后恢复');return;}
      if(viewPreparation){status('连接中断 · 可重试渲染','error');return;}
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
      loadingProgress(data.progress || 0);
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
        $('experimentStep').value=String(scene.export_sample_step || 4);
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
    } else if (!state.playing && !state.preparing && !state.exporting && !viewPreparation && !failedViewGroup) {
      text("streamStatus", "逐帧查看");
    } else if (state.interacting) {
      text("streamStatus", "视角交互 · 松开恢复播放");
    }
    if (now - state.lastReceived > 1500) $("displayFps").textContent = "0.0";
    requestFrame(); requestAnimationFrame(tick);
  }
  $("smoothPlayback").onchange=()=>{setPlaying(false);failedViewGroup=null;controls(true);state.dirty=true;};
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
    if(state.exporting || viewPreparation || failedViewGroup)return;
    if (event.target.closest("button") || event.target.closest(".overlay") || event.target.closest(".reference-cell") || event.target.closest(".compare-label")) return;
    if (detail.enabled && comparisonView()) {
      if (renderCanvases().includes(event.target)) startDetail(event,event.target);
      return;
    }
    stage.focus({preventScroll: true}); stage.setPointerCapture(event.pointerId);
    state.interacting = true; stopStream();
    pointer = {x: event.clientX, y: event.clientY, id: event.pointerId, pan: event.button === 2 || event.shiftKey};
  };
  stage.onpointermove = event => {
    if (detail.enabled && comparisonView()) {
      if (detail.drag) moveDetail(event);
      else if (renderCanvases().includes(event.target)) {
        const p=detailPoint(event,event.target),hit=p && hitDetail(p);
        const h=hit?.handle;
        event.target.style.cursor=!h?"crosshair":h==="move"?"move":`${h==="ne"||h==="sw"?"nesw":h==="nw"||h==="se"?"nwse":h==="n"||h==="s"?"ns":"ew"}-resize`;
      }
      return;
    }
    if (!pointer || pointer.id !== event.pointerId) return;
    const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
    if(dx || dy)selectDetailView("free");
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
  stage.ondblclick = event => {if (!state.exporting && !viewPreparation && !failedViewGroup && !(detail.enabled && comparisonView()) && !event.target.closest(".reference-cell")) reset();};
  stage.addEventListener("wheel", event => {
    if(state.exporting || viewPreparation || failedViewGroup)return;
    if (event.target.closest(".reference-cell") || event.target.closest(".compare-label")) return;
    if (detail.enabled && comparisonView()) return;
    state.interacting = true; stopStream(); clearTimeout(wheelTimer);
    wheelTimer = setTimeout(() => {state.interacting = false; state.dirty = true;}, 180);
    event.preventDefault(); state.zoom = Math.max(.2, Math.min(8, state.zoom * Math.exp(event.deltaY * .001)));
    state.dirty = true;
  }, {passive: false});
  document.addEventListener("keydown", event => {
    if(state.exporting || viewPreparation || failedViewGroup)return;
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
      const renders=renderCanvases(),sources=state.scene.reference?.available?[...renders,rgbCanvas,eventCanvas]:renders;
      const bounds=state.scene.reference?.available?overviewBounds():{top:0,bottom:1};
      const width=canvas.width,unit=width/$("leftRenderCell").getBoundingClientRect().width;
      const header=Math.round(28*unit),imageHeight=Math.round(canvas.height*(bounds.bottom-bounds.top));
      const topHeight=imageHeight+header,withDetails=detail.enabled && detail.regions.length>0,count=detail.regions.length*renders.length;
      const totalWidth=width*sources.length,cropHeight=withDetails?detailImageHeight(totalWidth/count,imageHeight):0;
      snapshot=document.createElement("canvas");snapshot.width=totalWidth;snapshot.height=topHeight+(withDetails?1+header+cropHeight:0);
      const ctx=snapshot.getContext("2d");ctx.fillStyle="#fff";ctx.fillRect(0,0,snapshot.width,snapshot.height);
      const names=[...renders.map(c=>$(c===canvas?'leftCompareLabel':c===middleCanvas?'middleCompareLabel':'rightCompareLabel').textContent),"RGB 视频 · 固定相机","事件相机视频 · 对齐 RGB"];
      for(const [i,source] of sources.entries()) {
        const x=i*width;ctx.fillStyle="#263543";ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(names[i],x+10*unit,18*unit,width-95*unit);
        ctx.save();ctx.fillStyle="#677987";ctx.font=`${9*unit}px sans-serif`;ctx.textAlign="right";ctx.fillText(`${state.scene.times_seconds[state.displayedFrame].toFixed(3)} s`,x+width-10*unit,18*unit);ctx.restore();
        ctx.drawImage(source,0,bounds.top*source.height,source.width,(bounds.bottom-bounds.top)*source.height,x,header,width,imageHeight);
        if(detail.enabled && i<renders.length){ctx.save();ctx.beginPath();ctx.rect(x,header,width,imageHeight);ctx.clip();ctx.translate(x,header);
          paintRegions(ctx,{x:0,y:-bounds.top*canvas.height,width,height:canvas.height},width/detailGeometry(source).width,false);ctx.restore();}
        if(i){ctx.fillStyle="#dfe4e8";ctx.fillRect(x,0,1,topHeight);}
      }
      if(withDetails) {
        ctx.fillStyle="#dfe4e8";ctx.fillRect(0,topHeight,totalWidth,1);
        for(const [i,r] of detail.regions.entries()) for(const [j,source] of renders.entries()) {
          const index=i*renders.length+j,x=Math.round(index*totalWidth/count),end=Math.round((index+1)*totalWidth/count),w=end-x;
          ctx.save();ctx.translate(x,topHeight+1);ctx.fillStyle=r.color;ctx.fillRect(0,0,w,3);
          ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(`${r.label} · ${comparisonStages()[j]}`,10*unit,18*unit,w-20*unit);
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
  const allVideoModes=[['gaussian','Gaussian 高斯'],['mesh','V1/V3 纯 Mesh'],['clay','V1/V3 白模 · 清晰柔光'],['pseudocolor','V1/V3 伪彩色 · 柔光'],['hybrid','Mesh＋GS']];
  let exportURL=null;
  $('cancelExport').onclick=()=>{if(exportJob){exportJob.cancelled=true;text('exportStatus','正在取消；已缓存的帧会保留。');}};
  $('exportVideo').onclick=async()=>{
    if(exportJob || !state.scene)return;
    if(!window.VideoEncoder || !window.VideoFrame){text('exportStatus','请使用新版 Chrome 或 Edge 导出完整视频。');return;}
    const videoModes=allVideoModes.filter(([mode])=>(mode==='gaussian' || state.scene[mode]?.available)
      && (state.scene.comparison || (state.scene.comparison_modes || []).includes(mode))).map(([mode,name])=>[mode,name.replace('V1/V3',comparisonStages(mode).join('/'))]);
    if(!videoModes.length){text('exportStatus','当前物件尚未提供成对画面。');return;}
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
          check();const bitmap=await createImageBitmap(GaussianPlayback.Media.recordBlob(entry.frames.get(frame)));
          try{measure(bitmap);}finally{bitmap.close();}
          if(modeIndex===0 && references){const ref=await references.frame(frame);try{measure(ref);}finally{ref.close();}}
          if(frame%40===0){report(`分析完整运动范围 · ${modeIndex+1}/${videoModes.length} · ${frame+1}/${count}`);await new Promise(resolve=>setTimeout(resolve,0));}
        }
      }
      const bounds=last<first?{top:0,bottom:1}:{top:Math.max(0,(first-8)/128),bottom:Math.min(1,(last+9)/128)};
      for(const r of regions){bounds.top=Math.max(0,Math.min(bounds.top,r.y-.035));bounds.bottom=Math.min(1,Math.max(bounds.bottom,r.y+r.h+.02));}
      const width=2560,layouts=[];
      const output=document.createElement('canvas'),ctx=output.getContext('2d',{alpha:false});
      const start=scene.times_seconds[0],timestamps=scene.times_seconds.map(t=>Math.round((t-start)*1e6));
      const durations=timestamps.map((t,i)=>i+1<count?timestamps[i+1]-t:Math.round(1e6/fps));
      const duration=timestamps[count-1]+durations[count-1];
      for(let modeIndex=0;modeIndex<settings.length;modeIndex++) {
        check();const [mode,name]=videoModes[modeIndex];
        const stages=comparisonStages(mode),renderCount=stages.length,columns=renderCount+(references?2:0);
        const panel=width/columns,unit=panel/$('leftRenderCell').getBoundingClientRect().width;
        const header=Math.round(28*unit),imageHeight=Math.round(panel*(bounds.bottom-bounds.top));
        const detailWidth=regions.length?width/(regions.length*renderCount):0,pad=Math.min(12,Math.max(4,detailWidth*.025));
        const detailHeight=regions.length?Math.ceil(Math.max(2*pad+24,Math.min(Math.max(...regions.map(r=>r.h/r.w))*(detailWidth-2*pad)+2*pad,detailWidth*1.6,imageHeight*1.15))):0;
        const height=Math.ceil((header+imageHeight+(regions.length?1+header+detailHeight:0))/2)*2;
        output.width=width;output.height=height;layouts.push({mode,width,height,version_labels:stages});
        const sources=Array.from({length:columns},()=>{const c=document.createElement('canvas');c.width=1024;c.height=1024;return c;});
        writer=await GaussianPlayback.Media.encoder(width,height,fps);
        for(let frame=0;frame<count;frame++) {
          check();const record=await store.get(mode,frame),bitmap=await decodeRecord(record);
          try {
            if(!record.meta.comparison)throw new Error(`${name} 未提供成对画面`);
            const image=bitmap.image||bitmap,pw=record.meta.panel_width,ph=record.meta.height;
            if((record.meta.panel_count || 2)!==renderCount)throw new Error('视频版本数量不一致');
            for(let i=0;i<renderCount;i++)sources[i].getContext('2d').drawImage(image,i*pw,0,pw,ph,0,0,1024,1024);
            if(references){const rw=scene.reference.width,rh=scene.reference.height;
              sources[renderCount].getContext('2d').drawImage(bitmap.reference,0,0,rw,rh,0,0,1024,1024);
              sources[renderCount+1].getContext('2d').drawImage(bitmap.reference,rw,0,rw,rh,0,0,1024,1024);}
          }finally{bitmap.close();}
          ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);
          const titles=[...stages.map(stage=>`${stage} · ${name.replace(/^V\d+(?:\/V\d+)+ /,'')}`),'RGB 视频','事件相机视频 · 对齐 RGB'];
          if(scene.detail_finish?.side==='right')titles[renderCount-1]+=' · 细节增强';
          for(const [i,source] of sources.entries()){
            const x=i*panel;ctx.fillStyle='#263543';ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(titles[i],x+10*unit,18*unit,panel-95*unit);
            ctx.save();ctx.fillStyle='#677987';ctx.font=`${9*unit}px sans-serif`;ctx.textAlign='right';ctx.fillText(`${scene.times_seconds[frame].toFixed(3)} s`,x+panel-10*unit,18*unit);ctx.restore();
            ctx.drawImage(source,0,bounds.top*1024,1024,(bounds.bottom-bounds.top)*1024,x,header,panel,imageHeight);
            if(i<renderCount && regions.length){ctx.save();ctx.beginPath();ctx.rect(x,header,panel,imageHeight);ctx.clip();ctx.translate(x,header);
              paintRegions(ctx,{x:0,y:-bounds.top*panel,width:panel,height:panel},unit,false);ctx.restore();}
            if(i){ctx.fillStyle='#dfe4e8';ctx.fillRect(x,0,1,header+imageHeight);}
          }
          if(regions.length){const y=header+imageHeight+1;ctx.fillStyle='#dfe4e8';ctx.fillRect(0,y-1,width,1);
            for(const [i,r] of regions.entries())for(let j=0;j<renderCount;j++){
              const x=Math.round((i*renderCount+j)*detailWidth),end=Math.round((i*renderCount+j+1)*detailWidth),w=end-x;
              ctx.save();ctx.translate(x,y);ctx.fillStyle=r.color;ctx.fillRect(0,0,w,2);ctx.font=`600 ${11*unit}px sans-serif`;ctx.fillText(`${r.label} · ${stages[j]}`,10*unit,18*unit,w-20*unit);
              ctx.translate(0,header);paintCrop(ctx,sources[j],r,w,detailHeight);ctx.restore();
              if(i*renderCount+j){ctx.fillStyle='#dfe4e8';ctx.fillRect(x,y,1,header+detailHeight);}
            }
          }
          await writer.add(output,timestamps[frame],durations[frame]);
          if(frame%5===0)report(`编码 ${modeIndex+1}/${videoModes.length} · ${name} · ${frame+1}/${count}`);
        }
        check();const blob=await writer.finish(duration);writer.close();writer=null;
        files.push({name:`${modeIndex+1}_${mode}.webm`,blob});
      }
      check();files.push({name:'manifest.json',blob:new Blob([JSON.stringify({scene:scene.name,frame_count:count,fps,duration_seconds:duration/1e6,
        width,height:layouts.every(l=>l.height===layouts[0].height)?layouts[0].height:null,layouts,overview:bounds,regions,camera:settings[0],modes:videoModes.map(([id,label])=>({id,label})),
        display_finish:scene.detail_finish||undefined,times_seconds:scene.times_seconds},null,2)],{type:'application/json'})});
      report(`正在打包 ${videoModes.length} 种完整视频…`);const archive=await GaussianPlayback.Media.zip(files);check();
      if(exportURL)URL.revokeObjectURL(exportURL);exportURL=URL.createObjectURL(archive);
      const link=$('exportDownload');link.href=exportURL;link.download=`scgs-${$('objectSelect').value||'scene'}-all-modes.zip`;link.hidden=false;link.click();
      report(`完成：${videoModes.length} 个完整视频 · 每个 ${count} 帧 · ${(duration/1e6).toFixed(2)} 秒 · ${(archive.size/1024**2).toFixed(1)} MiB`);
    }catch(error){report(job.cancelled?'已取消导出；已缓存的帧可继续复用。':`导出失败：${error.message}`);console.error('Video export',error);}
    finally{writer?.close();store?.close();controls.forEach(([el,disabled])=>{el.disabled=disabled;});$('cancelExport').hidden=true;state.exporting=false;exportJob=null;state.dirty=true;}
  };

  window.addEventListener("beforeunload", () => {presetViews.close();exportJob?.store?.close();state.socket?.close();});
  reset(); poll(); requestAnimationFrame(tick);
})();
