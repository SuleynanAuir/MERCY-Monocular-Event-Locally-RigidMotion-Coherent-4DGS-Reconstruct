'use strict';
(() => {
  const $=id=>document.getElementById(id), query=location.search;
  let stopped=false, timer, submitting=false, signature='';
  if((new URLSearchParams(query).get('scene')||'lego')!=='lego'){
    $('orbitPath').value='full';$('legoOrbitPath').disabled=true;$('legoOverviewPath').disabled=true;
  }
  function routeControls(){const full=$('orbitPath').value==='full';for(const id of ['orbitRadius','orbitElevation'])$(id).disabled=!full;}
  $('orbitPath').onchange=routeControls;routeControls();
  async function api(path,body) {
    const r=await fetch('/fvv/api/'+path+query,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok)throw Error((await r.text()).slice(0,240));return r.json();
  }
  async function refresh() {
    try {
      const data=await api('orbit-export'),job=data.job, running=job?.state==='running';
      $('saveOrbit').disabled=submitting||running;$('cancelOrbit').hidden=!running;
      $('orbitProgress').hidden=!job;$('orbitProgress').value=job?.progress||0;
      if(job)$('orbitStatus').textContent=running?`正在导出 ${job.frame}/${job.frames} 帧 · ${Math.floor(job.progress)}%`:job.state==='complete'?'导出完成，请保存视频和参数 JSON。':'导出未完成：'+job.error;
      const next=JSON.stringify(data.exports);
      if(signature!==next){signature=next;$('orbitDownloads').replaceChildren();
        for(const item of data.exports){
          const row=document.createElement('p'),label=document.createElement('span');row.className='orbit-export-item';label.textContent=new Date(item.created_at).toLocaleString()+' ';row.append(label);
          for(const [kind,text] of [['video','下载 MP4'],['parameters','下载参数 JSON']]){
            const a=document.createElement('a');a.textContent=text+' ';a.href='/fvv/api/orbit-export/'+item.id+'/'+kind+query;a.download='';row.append(a);
          }
          const button=document.createElement('button');button.textContent='删除副本';button.onclick=async()=>{
            if(!window.confirm('删除服务器上的这份视频与参数？已下载的文件不受影响。'))return;
            try{await api('orbit-export/'+item.id+'/delete',{});signature='';clearTimeout(timer);await refresh();}catch(e){$('orbitStatus').textContent=e.message;}
          };row.append(button);$('orbitDownloads').append(row);
        }
      }
    }catch(e){$('orbitStatus').textContent='视频导出服务未连接：'+e.message;}
    finally{if(!stopped){clearTimeout(timer);timer=setTimeout(refresh,1500);}}
  }
  $('saveOrbit').onclick=async()=>{
    if(submitting)return;
    const pose=window.FVV.exportPose();if(!pose){$('orbitStatus').textContent='请先加载动态高斯模型。';return;}
    submitting=true;$('saveOrbit').disabled=true;
    try{await api('orbit-export',{...pose,path:$('orbitPath').value,seconds:Number($('orbitSeconds').value),fps:Number($('orbitFps').value),width:Number($('orbitWidth').value),radius:Number($('orbitRadius').value),elevation:Number($('orbitElevation').value)});}
    catch(e){$('orbitStatus').textContent='无法导出：'+e.message;return;}
    finally{submitting=false;$('saveOrbit').disabled=false;}
    clearTimeout(timer);await refresh();
  };
  $('cancelOrbit').onclick=async()=>{try{await api('orbit-export/cancel',{});clearTimeout(timer);await refresh();}catch(e){$('orbitStatus').textContent=e.message;}};
  window.addEventListener('pagehide',()=>{stopped=true;clearTimeout(timer);});
  refresh();
})();
