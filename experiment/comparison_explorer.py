"""Shared eight-panel metric explorer for the portable and frozen reports."""
HTML = '''<div class="metric-atlas"><div class="atlas-toolbar"><label>共同时间 <input id="sample" type="range" min="0" value="0"></label><output id="values" aria-live="polite"></output><div id="version-toggles"></div></div><p class="note">同量纲指标共用坐标：PSNR、SSIM / IoU、两类缺失率分别合并。颜色表示版本，线型表示指标；拖动时间或悬停曲线，八个面板同步定位。其余指标保留独立量纲与刻度。</p><div id="metric-grid" class="metric-grid"></div></div>'''
CSS = '''
.metric-atlas{margin-top:20px}.atlas-toolbar{position:sticky;top:0;z-index:2;background:#f3f8f8eF;border:1px solid #d9e5e8;border-radius:10px;padding:12px 16px;backdrop-filter:blur(8px)}.atlas-toolbar input[type=range]{width:min(430px,60vw);vertical-align:middle}.atlas-toolbar output{display:inline-block;margin:0 12px;font-variant-numeric:tabular-nums}.atlas-toolbar label{margin-right:18px}#version-toggles{display:inline-flex;gap:12px;flex-wrap:wrap}#version-toggles label{margin:0}.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.metric-card{min-width:0;border:1px solid #dce7e9;border-radius:12px;padding:16px;background:linear-gradient(150deg,#fff,#f7fafb)}.metric-card h3{font-size:17px;margin:0 0 8px}.metric-card svg{display:block;width:100%;background:transparent;touch-action:pan-y}.metric-lines{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#526a74}.metric-line{display:inline-block;width:24px;border-top:2px solid #536b76;margin-right:5px;vertical-align:middle}.metric-line[data-line="1"]{border-top-style:dashed}.metric-line[data-line="2"]{border-top-style:dotted}.version-0{color:#507ead}.version-1{color:#d77a38}.version-2{color:#309b8a}.version-3{color:#ad729b}.version-4{color:#879e53}.version-5{color:#cc6269}.metric-reading{font-size:11px;line-height:1.7;font-variant-numeric:tabular-nums;min-height:42px}.metric-reading span{display:inline-block;margin:2px 8px 2px 0}.metric-reading b{font-weight:600}.atlas-empty{padding:30px;text-align:center;color:#758991}@media(max-width:800px){.metric-grid{grid-template-columns:1fr}.atlas-toolbar{position:static}.metric-card{padding:10px}}
'''
JS = r'''
(()=>{
const D=JSON.parse(document.getElementById('series').textContent);
const C=['#507ead','#d77a38','#309b8a','#ad729b','#879e53','#cc6269'],dash=['','7 4','2 3'];
const groups=[['重建质量 · PSNR','dB',['foreground_psnr','psnr']],['结构相似性与轮廓','0–1',['ssim','foreground_ssim','iou']],['前景 / 部件缺失率','%',['missing','part_missing']],['感知误差 · LPIPS','原始值',['lpips']],['事件相对 L1','原始值',['event_l1']],['事件余弦一致性','原始值',['event_cosine']],['极性一致性','%',['polarity']],['无事件区域伪变化','log-change',['silent']]];
const slider=document.getElementById('sample'),grid=document.getElementById('metric-grid'),toggle=document.getElementById('version-toggles');
D.names=D.names||['V1','V2','V3'];slider.max=Math.max(0,D.times.length-1);
D.names.forEach((name,i)=>{const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.v=i;label.append(input,document.createTextNode(' '+name));label.className='version-'+i;toggle.append(label);input.addEventListener('change',draw)});
function escape(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const cards=groups.map(([title,unit,keys])=>{const card=document.createElement('div');card.className='metric-card';card.innerHTML=`<h3>${title} <small>(${unit})</small></h3><div class="metric-lines"></div><svg viewBox="0 0 560 260" role="img" aria-label="${title}时间曲线"></svg><div class="metric-reading"></div>`;
card.querySelector('.metric-lines').innerHTML=keys.map((key,j)=>`<span><i class="metric-line" data-line="${j}"></i>${escape(D.metrics[key].label)}</span>`).join('');grid.append(card);return{card,keys,svg:card.querySelector('svg'),readout:card.querySelector('.metric-reading')}});
function draw(){const enabled=[...toggle.querySelectorAll('input:checked')].map(c=>+c.dataset.v),selected=+slider.value,current=D.times[selected]||0;
document.getElementById('values').textContent=`t = ${current.toFixed(4)} s · ${selected+1} / ${D.times.length}`;
for(const {keys,svg,readout}of cards){
const available=keys.some(key=>D.metrics[key].values.some(row=>row.some(x=>x!==null)));
svg.style.display=available?'':'none';
if(!available){readout.innerHTML='<div class="atlas-empty">'+keys.map(key=>'<b>'+escape(D.metrics[key].label)+'</b><br>'+escape(D.metrics[key].unavailable_reason||'缺少有效参考数据')).join('<br>')+'<br>补齐参考数据后，请重新导出并计算报告。</div>';continue}
let all=[];for(const key of keys)for(const v of enabled)all.push(...D.metrics[key].values[v].filter(x=>x!==null));
let lo=all.length?Math.min(...all):0,hi=all.length?Math.max(...all):1,pad=(hi-lo||1)*.08;lo-=pad;hi+=pad;
const start=D.times[0]||0,end=D.times.at(-1)||1,X=t=>64+(t-start)/(end-start||1)*472,Y=v=>210-(v-lo)/(hi-lo)*178;
let h='';for(let j=0;j<=4;j++){let val=lo+(hi-lo)*j/4,t=start+(end-start)*j/4;h+=`<path d="M64 ${Y(val)}H536" stroke="#dfe8eb"/><text x="57" y="${Y(val)+4}" text-anchor="end" font-size="10" fill="#526a74">${Math.abs(val)<.01&&val!==0?val.toExponential(1):Number(val.toPrecision(3))}</text><text x="${X(t)}" y="236" text-anchor="middle" font-size="10" fill="#526a74">${t.toFixed(2)} s</text>`}
let readings=[];
keys.forEach((key,k)=>{const m=D.metrics[key],event=m.values[0].length!==D.times.length,t=event?D.event_times:D.times;
let nearest=t.reduce((best,value,j)=>Math.abs(value-current)<Math.abs(t[best]-current)?j:best,0);
const matched=t.length&&Math.abs(t[nearest]-current)<1e-6;
for(const v of enabled){let path='',open=false;m.values[v].forEach((value,j)=>{if(value===null){open=false;return}path+=(open?'L':'M')+X(t[j])+','+Y(value)+' ';open=true});h+=`<path d="${path}" fill="none" stroke="${C[v]}" stroke-width="1.6" stroke-dasharray="${dash[k]}"/>`;
if(matched&&m.values[v][nearest]!==null)h+=`<circle cx="${X(current)}" cy="${Y(m.values[v][nearest])}" r="3" fill="${C[v]}" stroke="white"/>`}
readings.push(`<div><b>${escape(m.label)}</b> `+enabled.map(v=>`<span class="version-${v}">${escape(D.names[v])}: ${!matched?'无对应窗口':m.values[v][nearest]===null?'不可用 / 排除':Number(m.values[v][nearest].toPrecision(5))}</span>`).join('')+'</div>');
});
h+=`<path d="M${X(current)} 25V210" stroke="#8399a4" stroke-dasharray="4 4"/>`;if(!all.length)h+='<text x="285" y="125" text-anchor="middle" fill="#657d85">请勾选至少一个有有效数据的版本</text>';
svg.innerHTML=h;readout.innerHTML=readings.join('');
}
}
slider.addEventListener('input',draw);
for(const {svg}of cards)svg.addEventListener('pointermove',e=>{if(e.pointerType==='touch'&&!e.buttons)return;const b=svg.getBoundingClientRect(),fraction=Math.max(0,Math.min(1,((e.clientX-b.left)/b.width*560-64)/472)),time=D.times[0]+fraction*(D.times.at(-1)-D.times[0]);let index=D.times.reduce((best,t,j)=>Math.abs(t-time)<Math.abs(D.times[best]-time)?j:best,0);if(index!==+slider.value){slider.value=index;draw()}});
draw();
})();
'''
