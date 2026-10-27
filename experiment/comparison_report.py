"""Portable comparison report; PNG figures and compact numerical artifacts only."""
import html
import json
import re
from pathlib import Path
import numpy as np
from experiment.export_report_analysis import METRICS
from utils.comparison_outputs import identity, figure_name
from experiment.comparison_explorer import HTML as EXPLORER, CSS as EXPLORER_CSS, JS as EXPLORER_JS


def write(output, analysis, figures, warnings):
    output=Path(output);esc=html.escape;names=[r['version'] for r in analysis['summary']]
    def fmt(v):return '不可用' if v is None else f'{v:.5g}'
    rows=''.join('<tr><th>'+esc(label)+(' ↑' if direction>0 else ' ↓')+(' ('+unit+')' if unit else '')+'</th>'+''.join('<td>'+('∞' if r.get(key+'_infinite') else fmt(r.get(key)))+'</td>' for r in analysis['summary'])+'</tr>' for key,(label,unit,direction) in METRICS.items())
    titles={'01_quality':'质量与事件一致性','02_temporal':'动作周期与逐时刻变化','04_qualitative':'固定时刻原始画面对照','05_error_maps':'空间误差与 V3−V2 像素差值','06_parts':'公共部件覆盖','07_efficiency':'效率与质量代价','08_motion':'运动与采样有效性'}
    perf=analysis.get('performance',{})
    with __import__('gzip').open(output/'data/input_manifest.json.gz','rt') as stream:source=json.load(stream)
    environment=source.get('environment',{})
    resources=[('原始测量设备',environment.get('gpu','未记录')),('原始分辨率',str(analysis['protocol']['resolutions'])),('预热 / 测量轮数',str(perf.get('warmup','未记录'))+' / '+str(perf.get('repeats','未记录'))),('比较组测量次数',perf.get('samples','未记录')),('全组平均 / p95 时延 (ms)',fmt(perf.get('mean_ms'))+' / '+fmt(perf.get('p95_ms'))),('进程峰值显存 (MiB)',fmt(perf.get('memory_peak')/1024**2 if perf.get('memory_peak') else None)),('进程加载耗时 (s)',fmt(perf.get('load_seconds'))),('计时包含轨迹状态采集',str(perf.get('includes_state_capture','未记录'))),('原始计时范围',str(perf.get('scope','未记录')))]
    resource_rows=''.join('<tr><th>'+esc(k)+'</th><td>'+esc(str(v))+'</td></tr>' for k,v in resources)
    images=''.join(f'<section id="{esc(name)}"><div class="eyebrow">{esc(name[:2])} / COMPARISON</div><h2>{esc(titles.get(name,name))}</h2><a href="figures/{esc(figure_name(output,name))}" target="_blank" rel="noopener"><img loading="lazy" src="figures/{esc(figure_name(output,name))}" alt="{esc(titles.get(name,name))}"></a><a download href="figures/{esc(figure_name(output,name))}">下载 PNG ↗</a></section>' for name in sorted(figures))
    with np.load(output/'data/series.npz',allow_pickle=False) as z:
        payload=dict(times=z['times'].tolist(),event_times=z['event_times'].tolist(),names=names,metrics={k:dict(label=v[0],unit=v[1],values=[[None if not np.isfinite(x) else float(x) for x in row] for row in z[k]]) for k,v in METRICS.items()})
    tablelinks=''.join(f'<a download href="tables/{esc(p.name)}">{esc(p.stem)} CSV</a>' for p in sorted((output/'tables').glob('*.csv')))
    checks=''.join('<tr><th>'+esc(r['version'])+'</th><td>'+str(r['aligned_rgb'])+' / '+str(r['total_frames'])+'</td><td>'+str(r['metrics']['lpips'])+'</td><td>'+str(r['metrics']['iou'])+'</td><td>'+str(r['metrics']['part_missing'])+'</td><td>'+str(r['metrics']['event_l1'])+'</td></tr>' for r in analysis['availability'])
    notes=['雷达图仍按原始值映射半径；绿色 ↑ / + 表示越高越好，红色 ↑ / + 表示越高越差。下方分面圆点呈现逐指标改善百分比，保留负值；并非显著性检验。','部件热图按固定 GT 面积分组，显示对总可见面积缺失的贡献；不是部件内部缺失率。热图保留每个采样时刻，嵌入总缺失曲线与共同周期；面积加权与部件等权结果同时保留，不强制单调改善。','周期由公共源物体位移的重复性确定，不从预测质量峰值或 jerk 峰值反推动作周期。','事件主结果只使用 response_validation 窗口；原始逐窗表保留标定重叠及其他窗口。','公共部件掩码评估可见覆盖，不代表语义身份准确率或遮挡背面的完整性。','置信区间仅描述当前序列，不能代替独立场景、独立训练种子；性能轮次不是训练种子。','轨迹点只在各版本内固定，未验证跨版本点对应；原始 3D jerk 不用于证明平滑度提升。','显存为进程级记录；缺失的单版本显存、独立运动真值、训练成本等不会推算补齐。','PSNR 完全一致的帧标记为 ∞；有限配对区间排除非有限值，见逐帧标志。']+warnings
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>对照试验 · 可复现报告</title><link rel="icon" href="data:,"><style>
*{box-sizing:border-box}body{margin:0;background:#edf3f4;color:#213e4a;font:15px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}header{background:#183c49;color:white;padding:38px max(24px,calc((100vw - 1180px)/2))}h1{font-size:34px;margin:8px 0}main{max-width:1240px;margin:auto;padding:24px}section{padding:30px;background:white;border:1px solid #dce5e9;border-radius:16px;margin-bottom:22px;scroll-margin-top:15px}h2{margin:5px 0 20px}.eyebrow{font-size:12px;letter-spacing:2px;color:#438f8a}img{width:100%;display:block}a{color:#187d70}nav,.links{display:flex;gap:16px;flex-wrap:wrap}header a{color:#b4e9dc}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px;border-bottom:1px solid #dee7ea}thead{background:#edf3f5}.table{overflow:auto}li{margin-bottom:7px}.note{color:#657d85}select,input{padding:10px;max-width:100%}svg{width:100%;background:#f9fbfc}output{display:block}code{overflow-wrap:anywhere}.links a{font-size:13px}@media(max-width:650px){main{padding:10px}section{padding:18px}h1{font-size:27px}th,td{padding:6px;font-size:12px}}
</style></head><body><header><div>SC-GS / COMPARISON STUDY</div><h1>对照试验 · 全维度评测</h1><div>__COUNT__ 个采样时刻 · __VERSIONS__ · 按导出中的相机、时间与原始分辨率计算</div><nav><a href="#summary">指标总表</a><a href="#explorer">时间浏览</a><a href="#availability">数据完整性</a><a href="#files">结果文件</a></nav></header><main>
<section id="summary"><div class="eyebrow">MEASURED RESULTS</div><h2>原始指标与量纲</h2><div class="table"><table><thead><tr><th>指标</th>__HEAD__</tr></thead><tbody>__ROWS__</tbody></table></div><p class="note">↑ 越大越好，↓ 越小越好。不可用项保留原因与有效样本数，不补零、不删除劣势结果。</p></section>
<section id="explorer"><div class="eyebrow">TEMPORAL EXPLORER</div><h2>逐时刻比较</h2>__EXPLORER__</section>
__IMAGES__
<section><div class="eyebrow">MEASUREMENT CONTEXT</div><h2>性能测量与资源记录</h2><div class="table"><table>__RESOURCE_ROWS__</table></div><p class="note">总组时延与单版本时延分开报告；不从同进程记录推算单版本显存或训练时间。</p></section>
<section id="availability"><div class="eyebrow">EVIDENCE COVERAGE</div><h2>数据完整性与解释范围</h2><div class="table"><table><thead><tr><th>版本</th><th>对齐 RGB / 总帧数</th><th>LPIPS 帧</th><th>IoU 帧</th><th>部件帧</th><th>事件主窗口</th></tr></thead><tbody>__CHECKS__</tbody></table></div><ul>__NOTES__</ul><a href="data/version_mapping.json">版本与来源映射</a> · <a href="data/availability.json">完整性记录</a> · <a href="data/protocol.json">评测协议</a> · <a href="data/visualization_protocol.json">选帧、部件筛选与图轴规则</a><p class="note">本页只包含对照试验；不包含消融结果。文件哈希：<code>__SHA__</code></p></section>
<section id="files"><h2>紧凑结果文件</h2><p>保留 PNG、逐帧／逐窗／部件 CSV、协议、分数、来源清单与代码快照，不复制模型、视频或全部渲染帧。</p><div class="links">__TABLELINKS__<a href="artifact_manifest.json">文件哈希清单</a></div></section></main><script id="series" type="application/json">__PAYLOAD__</script><script>
__EXPLORER_JS__
</script></body></html>'''
    meta=identity(output)
    if meta:doc=doc.replace('<h1>对照试验 · 全维度评测</h1>',f'<h1>{esc(meta["object_name"])} · 对照试验</h1><div>{esc(meta["created_at"])} · 北京时间</div>')
    replacements={'EXPLORER':EXPLORER,'EXPLORER_JS':EXPLORER_JS,'COUNT':str(analysis['protocol']['sample_count']),'VERSIONS':esc(' / '.join(names)),'HEAD':''.join('<th>'+esc(n)+'</th>' for n in names),'ROWS':rows,'IMAGES':images,'RESOURCE_ROWS':resource_rows,'CHECKS':checks,'NOTES':''.join('<li>'+esc(t)+'</li>' for t in notes),'SHA':analysis['protocol']['manifest_sha256'],'TABLELINKS':tablelinks,'PAYLOAD':json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')}
    for key,value in replacements.items():doc=doc.replace('__'+key+'__',value)
    css=re.search(r'<style>(.*?)</style>',doc,re.S).group(1)
    doc=doc.replace(css,css+EXPLORER_CSS)
    css+=EXPLORER_CSS
    script=re.search(r'<script>(.*?)</script>',doc,re.S).group(1)
    (output/'report.css').write_text(css,encoding='utf-8')
    (output/'report.js').write_text(script,encoding='utf-8')
    doc=doc.replace('<style>'+css+'</style>','<link rel="stylesheet" href="report.css">').replace('<script>'+script+'</script>','<script src="report.js"></script>')
    (output/'report.html').write_text(doc,encoding='utf-8')
