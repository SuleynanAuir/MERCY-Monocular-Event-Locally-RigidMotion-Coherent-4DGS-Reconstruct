'use strict';
(() => {
  const $=id=>document.getElementById(id), query=location.search;
  let data, timer, working=false, polling=false, awaitingLoad=false, selectedResult=null, upload=null, stopped=false, notice=null;
  function modelLabel(model) {
    const date=new Date(model.trained_at || NaN);
    return `${model.scene || '模型'} · ${Number.isNaN(date.getTime()) ? '时间未知' : date.toLocaleString('zh-CN',{hour12:false})}`;
  }
  function deleteState() {
    const selected=data?.models.find(x=>x.id===$('modelSelect').value);
    $('deleteModel').disabled=working||data?.loading||data?.job?.state==='running'||!selected||selected.active;
  }
  function progress(message,value) {
    notice=message; $('modelTask').textContent=message; $('modelProgress').hidden=false;
    if(value===null) $('modelProgress').removeAttribute('value');
    else $('modelProgress').value=value;
  }
  async function api(path,body) {
    const response=await fetch('/fvv/api/'+path+query,body===undefined ? {cache:'no-store'} :
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!response.ok) throw new Error((await response.text()).slice(0,400));
    return response.json();
  }
  function options(element,items,label) {
    const previous=element.value, signature=JSON.stringify(items);
    if(element.dataset.signature===signature) return;
    element.dataset.signature=signature; element.replaceChildren();
    for(const item of items) {const option=document.createElement('option');option.value=item.id;option.textContent=label(item);element.append(option);}
    if(items.some(x=>x.id===previous)) element.value=previous;
    else if(items.some(x=>x.active)) element.value=items.find(x=>x.active).id;
  }
  async function refresh() {
    if(polling||stopped) return;
    polling=true;
    try {
      data=await api('models');
      options($('modelSelect'),data.models,x=>`${x.active?'当前 · ':''}${modelLabel(x)} · ${(x.bytes/1048576).toFixed(1)} MiB`);
      if(selectedResult && !data.models.some(x=>x.id===selectedResult)) selectedResult=null;
      deleteState();
      options($('trainingProfile'),data.profiles,x=>`${x.label} · ${x.iterations} 次`);
      const running=data.job?.state==='running';
      $('loadModel').disabled=working||data.loading||!data.models.length;
      $('startTraining').disabled=working||running||data.loading||!data.profiles.length;
      $('importModel').disabled=working||running||data.loading||!$('modelFile').files.length;
      if(!working) {
        if(data.loading) progress('校验模型并加载到 GPU…',null);
        else if(awaitingLoad) {
          awaitingLoad=false;
          if(data.load_error) {progress('加载失败，原模型保留：'+data.load_error,0);window.FVV.connect();}
          else {progress('模型已就绪，可以自由浏览。',100);window.FVV.connect();}
        } else if(running) progress(`${data.job.stage} · ${data.job.iteration}/${data.job.iterations} 次 · ${data.job.progress.toFixed(1)}%`,data.job.iteration?data.job.progress:null);
        else if(data.job?.state==='failed') progress('训练失败，原模型保留：'+data.job.error,0);
        else if(data.job?.state==='complete') {selectedResult=data.job.id;progress(`训练完成（${data.job.iterations} 次），可加载新模型；当前浏览模型保持不变。`,100);}
        else $('modelTask').textContent=notice || (data.models.length?'选择已有模型即可浏览，也可独立训练或导入模型包。':'该场景尚无模型，请先训练或导入。');
      }
      $('loadTrained').hidden=!selectedResult;
      $('loadTrained').disabled=working||data.loading;
    } catch(error) {if(!working) $('modelTask').textContent='模型管理服务未连接：'+error.message;}
    finally {polling=false;if(!stopped) timer=setTimeout(refresh,1500);}
  }
  async function load(id) {
    if(working||!id) return;
    working=true;window.FVV.disconnect();progress('准备加载模型…',null);
    try {await api('models/load',{id});awaitingLoad=true;}
    catch(error) {progress('加载失败：'+error.message,0);window.FVV.connect();}
    finally {working=false;}
  }
  $('loadModel').onclick=()=>load($('modelSelect').value);
  $('modelSelect').onchange=deleteState;
  $('deleteModel').onclick=async()=>{
    const selected=data?.models.find(x=>x.id===$('modelSelect').value);
    if(working||data.loading||selected?.active||!selected) return;
    if(!window.confirm(`删除 ${modelLabel(selected)}？模型文件将永久移除。`)) return;
    working=true;deleteState();
    try {await api('models/delete',{id:selected.id,sha256:selected.sha256});
      if(selectedResult===selected.id) selectedResult=null;
      progress('模型已删除，存储空间已释放。',100);
    } catch(error) {progress('删除失败：'+error.message,0);}
    finally {working=false;clearTimeout(timer);await refresh();}
  };
  $('loadTrained').onclick=()=>load(selectedResult);
  $('startTraining').onclick=async()=>{
    if(working) return; working=true;$('startTraining').disabled=true;selectedResult=null;
    try {await api('training',{profile:$('trainingProfile').value});progress('训练任务已提交，正在准备真实数据…',null);}
    catch(error) {progress('无法启动训练：'+error.message,0);}
    finally {working=false;}
  };
  $('modelFile').onchange=()=>{$('importModel').disabled=working||!$('modelFile').files.length;};
  $('importModel').onclick=()=>{
    const file=$('modelFile').files[0];if(!file||working) return;
    if(file.size>512*1048576) {progress('模型包不能超过 512 MiB。',0);return;}
    working=true;$('importModel').disabled=true;progress('正在上传模型包…',0);
    upload=new XMLHttpRequest();upload.open('POST','/fvv/api/models/import'+query);upload.timeout=900000;
    upload.setRequestHeader('Content-Type','application/zip');
    upload.upload.onprogress=e=>{if(e.lengthComputable) progress(`上传模型包 ${Math.round(100*e.loaded/e.total)}%`,100*e.loaded/e.total);};
    upload.upload.onload=()=>progress('上传完成，正在校验文件与场景…',null);
    upload.onload=async()=>{
      working=false;
      if(upload.status<200||upload.status>=300) {progress('导入失败：'+upload.responseText.slice(0,400),0);return;}
      try {const result=JSON.parse(upload.responseText);await load(result.id);}
      catch(error) {progress('导入失败：'+error.message,0);}
    };
    upload.onerror=upload.ontimeout=()=>{working=false;progress('上传连接中断，请检查隧道后重试。',0);};
    upload.send(file);
  };
  window.addEventListener('pagehide',()=>{stopped=true;clearTimeout(timer);if(upload&&upload.readyState!==4)upload.abort();});
  refresh();
})();
