"""Efficiency and paired temporal statistics from the exact UI capture records.

These are conditional on one frozen training seed, not independent repeated
training trials. Frames are temporally blocked rather than treated as IID.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .protocol import save_json


def paired_blocks(delta, seed=17, repetitions=2000):
    delta=np.asarray(delta,dtype=float)
    n=len(delta);length=max(2,int(np.ceil(np.sqrt(n))))
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(repetitions,int(np.ceil(n/length))))
    ids=(starts[:,:,None]+np.arange(length))%n
    samples=delta[ids.reshape(repetitions,-1)[:,:n]].mean(1)
    chunks=[delta[i:i+length] for i in range(0,n,length)]
    sums=np.array([x.sum() for x in chunks]);b=len(chunks)
    if b>15:raise ValueError('Exact sign test would be too large')
    signs=np.array(list(itertools.product((-1,1),repeat=b)))
    null=signs@sums/n;observed=float(delta.mean())
    return dict(mean=observed,ci95=np.quantile(samples,[.025,.975]).tolist(),
        p=float(np.mean(np.abs(null)>=abs(observed)-1e-14)),frames=n,
        block_length=length,sign_blocks=b,permutations=len(null),bootstrap_repetitions=repetitions)


def generate(root,output):
    from .visualize import style,save
    import matplotlib.pyplot as plt
    inventory=json.loads((root/'input-inventory.json').read_text())
    inventory['scenes']=[s for s in inventory['scenes'] if all(
        (root/'ui-evaluation/full'/f"{s['scene']}-v1-{v}"/'clay/results.json').exists() for v in ('v2','v3'))]
    if not inventory['scenes']:raise ValueError('No complete UI scene captures')
    efficiency=[];statistics=[]
    for scene in inventory['scenes']:
        name=scene['scene'];test=set(scene['split']['frames']['test'])
        captures=[]
        for version in ('v2','v3'):
            cap=json.loads((root/'ui-evaluation/full'/f'{name}-v1-{version}'/'clay/results.json').read_text())
            captures.append(cap)
            rows=[r['metadata'] for r in cap['frames'][1:] if not r['metadata'].get('cache_hit')]
            efficiency.append(dict(scene=name,pair='V1/'+version.upper(),samples=len(rows),
                render_median_ms=float(np.median([r['render_ms'] for r in rows])),
                render_p95_ms=float(np.quantile([r['render_ms'] for r in rows],.95)),
                encode_median_ms=float(np.median([r['encode_ms'] for r in rows])),
                request=cap['request'],scope='Actual UI pair server timing; first frame and cache hits excluded; no network/browser time'))
        byversion={v:{r['frame']:r['metrics'][side] for r in cap['frames'] if r['frame'] in test}
            for v,cap,side in [('V1',captures[0],'left'),('V2',captures[0],'right'),('V3',captures[1],'right')]}
        for baseline in ('V1','V2'):
            for metric,direction in [('rgb_l1',-1),('psnr',1),('ssim',1)]:
                ids=sorted(test)
                delta=[direction*(byversion['V3'][i][metric]-byversion[baseline][i][metric]) for i in ids]
                statistics.append(dict(scene=name,comparison='V3-'+baseline,metric=metric,
                    positive_means='V3 improvement in the named display metric',**paired_blocks(delta)))
    ordered=sorted(range(len(statistics)),key=lambda i:statistics[i]['p'])
    last=0.
    for rank,i in enumerate(ordered):
        last=max(last,min(1.,statistics[i]['p']*(len(ordered)-rank)))
        statistics[i]['p_holm']=last
        statistics[i]['reject_at_005']=last<.05
    eff=output/'04_efficiency';sig=output/'05_significance'
    from .metrics import paired_effect, holm
    scene_tests=[]
    for comparison in ('V3-V1','V3-V2'):
        for key in ('rgb_l1','psnr','ssim'):
            selected=[r for r in statistics if r['comparison']==comparison and r['metric']==key]
            effects=[r['mean'] for r in selected]
            scene_tests.append(dict(comparison=comparison,metric=key,scenes=[r['scene'] for r in selected],
                **paired_effect(np.zeros(len(effects)),effects,1)))
    measured_tests=[r for r in scene_tests if r['status']=='measured']
    for row,p in zip(measured_tests,holm([r['p_raw'] for r in measured_tests])):
        row['p_holm']=p;row['reject_at_005']=p<.05
    save_json(eff/'results.json',dict(scope='Measured UI server pair render/encode timing, not browser FPS or network throughput',rows=efficiency))
    save_json(sig/'results.json',dict(scope='Primary unit is one scene, one frozen training seed; no training-seed uncertainty',
        completed_scenes=len(inventory['scenes']),expected_scenes=5,
        primary_scene_tests=scene_tests,
        conditional_temporal_protocol=f'Appendix only: circular paired moving-block bootstrap CI; exact paired block sign-flip two-sided p; Holm over {len(statistics)} available comparisons',
        conditional_temporal_rows=statistics))
    lines=['# 实际 UI 渲染效率','',
        '实际 WebSocket 响应的服务器计时。排除首帧和缓存命中；V1/V2 与 V1/V3 分别为整对渲染耗时。分辨率每格 640×640、JPEG 95；V3 保留 UI 原有 2 倍超采样。不同模型点数与显示增强均计入。不是独立单模型速度，也不等于客户端播放 FPS。','',
        '| 场景 | 配对 | 样本帧 | 渲染中位数 ms | P95 ms | 编码中位数 ms |','|---|---|---:|---:|---:|---:|']
    for r in efficiency:lines.append(f"| {r['scene']} | {r['pair']} | {r['samples']} | {r['render_median_ms']:.2f} | {r['render_p95_ms']:.2f} | {r['encode_median_ms']:.2f} |")
    (eff/'results.md').write_text('\n'.join(lines)+'\n')
    lines=['# 五场景配对统计','',
        f"当前完成 {len(inventory['scenes'])}/5 个场景。主检验的独立单位是场景，每个场景先汇总预先固定测试段。至少两个场景才计算场景区间；完整五场景时对 6 项主检验使用 Holm 校正。五场景双侧精确检验最小 p 为 0.0625，不能宣称在 0.05 水平显著。",'',
        '| 对照 | 指标 | 场景数 | 平均改善 | 95% 场景区间 | Holm p |','|---|---|---:|---:|---|---:|']
    for r in scene_tests:
        if r['status']!='measured':
            lines.append(f"| {r['comparison']} | {r['metric']} | {r['n']} | N/A | 独立场景不足 | N/A |")
        else:lines.append(f"| {r['comparison']} | {r['metric']} | {r['n']} | {r['effect']:.6f} | [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}] | {r['p_holm']:.4f} |")
    lines+=['','## 附录：条件于单个场景的时间变化','',
        f'只对本轮冻结 UI 的 RGB L1、PSNR、SSIM 做同帧配对；正值表示 V3 在该项指标改善。使用循环移动块 bootstrap（2000 次）估计区间，精确块符号翻转检验，并对当前可用的 {len(statistics)} 项做 Holm 校正（五场景完整时为 30 项）。短片测试段只有少量独立时间块，统计分辨率有限。',
        '**只有一个训练种子；这些区间描述同段视频的时间变化，不代表跨训练种子、跨数据集的显著性。白模对 RGB 指标含显示材质差异。**','',
        '| 场景 | 对照 | 指标 | 平均改善 | 95% 块区间 | Holm p |','|---|---|---|---:|---|---:|']
    for r in statistics:lines.append(f"| {r['scene']} | {r['comparison']} | {r['metric']} | {r['mean']:.6f} | [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}] | {r['p_holm']:.4f} |")
    (sig/'results.md').write_text('\n'.join(lines)+'\n')
    style();names=[s['scene'] for s in inventory['scenes']];x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for j,pair in enumerate(('V1/V2','V1/V3')):
        values=[next(r for r in efficiency if r['scene']==s and r['pair']==pair) for s in names]
        heights=[r['render_median_ms'] for r in values]
        ax.bar(x+(j-.5)*.32,heights,.30,label=pair,color=('#879BB0','#159E96')[j])
    ax.set_xticks(x,[s.title() for s in names]);ax.set_ylabel('Server pair rendering · median ms');ax.legend(frameon=False)
    ax.set_title('Actual UI rendering cost · uncached frames',fontweight='bold')
    save(fig,eff/'visualization/ui_render_cost')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    for ax,key,title in zip(axes,('rgb_l1','psnr','ssim'),('RGB L1 reduction','PSNR increase (dB)','SSIM increase')):
        vals=[next(r for r in statistics if r['scene']==s and r['comparison']=='V3-V1' and r['metric']==key) for s in names]
        means=np.array([r['mean'] for r in vals]);ci=np.array([r['ci95'] for r in vals])
        for i,r in enumerate(vals):
            ax.plot(ci[i],[i,i],color='#159E96',linewidth=2)
            ax.plot(means[i],i,'o',color='#159E96')
        ax.axvline(0,color='#94A3B8',ls='--',lw=1);ax.set_yticks(np.arange(len(names)),[s.title() for s in names]);ax.invert_yaxis();ax.set_title(title)
    fig.suptitle('V3 versus V1 · paired temporal block 95% intervals · one frozen seed',fontweight='bold')
    save(fig,sig/'visualization/paired_temporal_intervals')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();generate(args.root,args.output)


if __name__=='__main__':main()
