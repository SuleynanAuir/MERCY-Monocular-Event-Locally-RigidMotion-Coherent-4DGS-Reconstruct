"""Measured result tables and standalone publication-style visual artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
from .protocol import METRICS,save_json
from .metrics import paired_effect,holm
from .visualize import style,save,COLORS,inventory_plots

SCENES=['lego','fan','lamp','metronome','microphone']
ENGLISH=dict(rgb_l1='RGB L1',psnr='PSNR (dB)',ssim='SSIM',event_relative_l1='Held-out event relative L1',
    event_pass='Event pass fraction',visible_missing='Visible coverage deficit',back_missing='Rear missing',
    joint_gap_max='Source-joint gap (scene units)',angular_jerk='Angular jerk RMS (rad/s³)',position_jerk='3D jerk RMS (scene units/s³)')


def read_runs(folder):
    rows=[]
    planfile=folder/'queue_protocol.json'
    plan=json.loads(planfile.read_text()) if planfile.exists() else None
    registered={f'{s}_{m}_s{seed}_{variant}':(s,m,seed,variant)
                for s,m,seed,variant in plan['jobs']} if plan else {}
    for path in sorted(folder.glob('runs/*/evaluation.json')):
        row=json.loads(path.read_text());config=json.loads((path.parent/'config.json').read_text())
        identity=(row['scene'],config['method'],config['seed'],config['variant'])
        if plan and (registered.get(path.parent.name)!=identity or
                     config['steps']!=(0 if config['method']=='C1' else plan['steps'])):
            raise ValueError('Unregistered run or unequal budget: '+str(path.parent))
        row.update(seed=config['seed'],variant=config['variant'],run_path=str(path.parent),
            inputs=config['inputs'],best_step=json.loads((path.parent/'runtime.json').read_text())['best_step'])
        rows.append(row)
    return rows


def aggregate(rows,ablation=False):
    result=[]
    for scene in SCENES:
        names=['full','no_mlp','no_event','no_joint','no_jerk'] if ablation else ['C1','C2','C3']
        for name in names:
            subset=[r for r in rows if r['scene']==scene and (r['variant'] if ablation else r['method'])==name]
            if not subset:continue
            if len({json.dumps(r['inputs'],sort_keys=True) for r in subset})!=1:
                raise ValueError('Cannot aggregate runs with mismatched frozen inputs')
            metrics={}
            for key,_,_ in METRICS:
                finite=[r['metrics'][key]['value'] for r in subset if r['metrics'][key]['value'] is not None]
                if len(finite)==len(subset):
                    metrics[key]=dict(value=float(np.mean(finite)),sd=float(np.std(finite,ddof=1)) if len(finite)>1 else None,n=len(finite))
                    if key=='event_pass':metrics[key].update(passed=[r['metrics'][key]['passed'] for r in subset],total=subset[0]['metrics'][key]['total'])
                else:metrics[key]=dict(value=None,reason=next(r['metrics'][key]['reason'] for r in subset if r['metrics'][key]['value'] is None))
            result.append(dict(scene=scene,method=name,metrics=metrics,seeds=[r['seed'] for r in subset],
                observation_group=subset[0]['observation_group'],selected_steps=[r['best_step'] for r in subset]))
    return result


def formatted(metric,key):
    if metric['value'] is None:return 'N/A'
    if key=='event_pass':
        values=metric['passed'];label=str(values[0]) if len(set(values))==1 else f'{np.mean(values):.2f} ± {np.std(values,ddof=1):.2f}'
        return label+'/'+str(metric['total'])
    value=metric['value'];sd=metric['sd']
    scale=100 if key in ('visible_missing','back_missing') else 1
    suffix='%' if scale==100 else ''
    return (f'{value*scale:.6g}'+(f' ± {sd*scale:.3g}' if sd is not None else '')+suffix)


def tables(rows,folder,title):
    folder.mkdir(parents=True,exist_ok=True)
    queue=folder/'queue_protocol.json'
    expected=len(json.loads(queue.read_text())['jobs']) if queue.exists() else None
    finished=len(list(folder.glob('runs/*/evaluation.json')))
    status='complete' if expected is not None and finished==expected else 'partial'
    progress=f'已完成 {finished}/{expected} 组正式运行。' if status=='complete' else f'进行中：已完成 {finished}/{expected} 组；以下是阶段性结果。'
    lines=['# '+title,'','本轮为统一运动适配版；范围和指标定义见 [实验协议](../README.md)。',
        progress,
        'C2/C3 主表为三个种子的均值 ± 样本标准差；固定 C1 不重复计算标准差。消融为种子 17。',
        'N/A 表示证据不可用；可见缺失为图像前景覆盖代理，背面缺失不以先验充当真值。','']
    for scene in SCENES:
        lines+=['## '+scene.title(),'','| 实验编号／方法 | '+' | '.join(x[1] for x in METRICS)+' |',
                '| '+' | '.join(['---']*(1+len(METRICS)))+' |']
        for row in rows:
            if row['scene']!=scene:continue
            name={'C1':'C1 · Bind-GS','C2':'C2 · Align-GS','C3':'C3 / A0 · MERCY','full':'A0 · MERCY',
                  'no_mlp':'A1 · 去局部 MLP','no_event':'A2 · 去事件','no_joint':'A3 · 去连接约束','no_jerk':'A4 · 去轨迹约束'}[row['method']]
            lines+=['| '+name+' | '+' | '.join(formatted(row['metrics'][key],key) for key,_,_ in METRICS)+' |']
        selected=[r for r in rows if r['scene']==scene]
        steps='；'.join(r['method']+' '+('/'.join(map(str,r['selected_steps']))) for r in selected)
        lines+=['','验证选中的更新步数：'+steps+'。0 表示保留零修正基线。','']
    (folder/'results.md').write_text('\n'.join(lines)+'\n')
    save_json(folder/'summary.json',dict(status=status,completed_runs=finished,expected_runs=expected,rows=rows))
    with (folder/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scene','method','seeds']+[item for key,_,_ in METRICS for item in (key,key+'_sd')])
        for r in rows:writer.writerow([r['scene'],r['method'],','.join(map(str,r['seeds']))]+[value for k,_,_ in METRICS for value in (r['metrics'][k]['value'],r['metrics'][k].get('sd'))])


def comparison_plot(rows,folder,keys,stem,title,methods=None):
    import matplotlib.pyplot as plt
    methods=methods or ['C1','C2','C3'];style()
    fig,axes=plt.subplots(5,len(keys),figsize=(4.5*len(keys),13),squeeze=False,layout='constrained')
    lookup={(r['scene'],r['method']):r for r in rows}
    for i,scene in enumerate(SCENES):
        for j,key in enumerate(keys):
            labels=[{'full':'Full','no_mlp':'−MLP','no_event':'−Events','no_joint':'−Joints','no_jerk':'−Trajectory'}.get(m,m) for m in methods]
            ax=axes[i,j];ax.set_xticks(range(len(methods)),labels,fontsize=8,rotation=15 if len(methods)>3 else 0)
            if i==0:ax.set_title(ENGLISH[key],fontsize=12,fontweight='bold',pad=14)
            if j==0:ax.set_ylabel(scene.title(),fontweight='bold',fontsize=12,labelpad=14)
            available=False
            for x,method in enumerate(methods):
                r=lookup.get((scene,method));metric=r['metrics'][key] if r else None
                if metric is None or metric['value'] is None:continue
                available=True
                ax.errorbar(x,metric['value'],yerr=metric.get('sd') or 0,fmt='o',ms=7,
                    color=COLORS[x%len(COLORS)],capsize=4,lw=1.7)
                ax.annotate(f"{metric['value']:.4g}",(x,metric['value']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xlim(-.4,len(methods)-.6);ax.margins(y=.28)
            if not available:
                exists=any((scene,m) in lookup for m in methods)
                message='Unavailable reference' if exists else 'Evaluation pending'
                ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,color='#8A96A3',fontsize=9)
                ax.set_yticks([]);ax.grid(False)
            ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',visible=False)
    fig.suptitle(title,fontsize=17,fontweight='bold')
    note='error bars: training-seed SD' if len(methods)==3 else 'seed 17 · each ablation retrained'
    fig.supxlabel('Native 640 × 640 GS · fixed common G0 · held-out time block · '+note,fontsize=10)
    save(fig,folder/'visualization'/stem)


def qualitative(rows,folder):
    import matplotlib.pyplot as plt
    from PIL import Image
    style();fig,axes=plt.subplots(5,4,figsize=(12,15),layout='constrained')
    for i,scene in enumerate(SCENES):
        runs={r['method']:Path(r['run_path']) for r in rows if r['scene']==scene and r['seed']==17}
        for j,name in enumerate(['Reference','C1','C2','C3']):
            ax=axes[i,j];ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
            if i==0:ax.set_title(name,fontweight='bold')
            if j==0:ax.set_ylabel(scene.title(),fontweight='bold')
            run=runs.get('C1' if name=='Reference' else name)
            files=sorted((run/'renders').glob(('reference' if name=='Reference' else 'frame')+'_*.png')) if run else []
            if files:ax.imshow(Image.open(files[len(files)//2]))
            else:ax.text(.5,.5,'Unavailable',ha='center',va='center')
    fig.suptitle('Same-frame, same-camera held-out comparison',fontweight='bold',fontsize=17)
    fig.supxlabel('Seed 17 · middle of the three preselected test snapshots · no display filtering',fontsize=10)
    save(fig,folder/'visualization/qualitative_comparison')


def learning_curves(rows,folder):
    import matplotlib.pyplot as plt
    style();fig,axes=plt.subplots(1,5,figsize=(17,3.7),layout='constrained')
    for ax,scene in zip(axes,SCENES):
        shown=set()
        for row in rows:
            if row['scene']!=scene or row['method']=='C1':continue
            curve=json.loads((Path(row['run_path'])/'training.json').read_text())
            data=[r for r in curve if 'validation' in r]
            method=row['method'];color=COLORS[1 if method=='C2' else 2]
            ax.plot([r['step'] for r in data],[r['validation'] for r in data],color=color,alpha=.7,
                    lw=1.5,label=method if method not in shown else None)
            chosen=next(r for r in data if r['step']==row['best_step'])
            ax.plot(chosen['step'],chosen['validation'],'o',ms=4,color=color)
            shown.add(method)
        ax.set(title=scene.title(),xlabel='Optimizer updates');ax.tick_params(axis='y',labelsize=8)
    axes[0].set_ylabel('Validation selection objective');axes[-1].legend(frameon=False)
    fig.suptitle('Validation trajectories · dots mark selected checkpoints',fontsize=16,fontweight='bold')
    save(fig,folder/'visualization/validation_curves')


def event_gate_diagnostics(rows,folder):
    """Failure reasons overlap; show separate rates, never stack their counts."""
    import matplotlib.pyplot as plt
    reasons=['insufficient_event_evidence','cosine','polarity_agreement',
             'unsupported_change_fraction','relative_l1']
    labels=['Weak evidence','Cosine','Polarity','Unsupported','Relative L1']
    scenes=[s for s in SCENES if any(r['scene']==s and r['event_metrics'] for r in rows)]
    records=[]
    for scene in scenes:
        for method in ('C1','C2','C3'):
            selected=[r for r in rows if r['scene']==scene and r['method']==method]
            samples=[]
            for run in selected:
                intervals=run['event_metrics']
                counts={reason:sum(reason in r['gate']['reasons'] for r in intervals) for reason in reasons}
                samples.append(dict(seed=run['seed'],total=len(intervals),counts=counts))
            records.append(dict(scene=scene,method=method,samples=samples,
                mean_rejected_fraction={reason:float(np.mean([r['counts'][reason]/r['total'] for r in samples])) for reason in reasons}))
    save_json(folder/'event_gate_diagnostics.json',dict(records=records,
        scope='Held-out intervals; failure reasons can overlap. Mean fraction across registered seeds; not additive.',
        unavailable={'fan':'Truncated raw event input'}))
    style();fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,scene in zip(axes.flat,scenes):
        for j,method in enumerate(('C1','C2','C3')):
            row=next(r for r in records if r['scene']==scene and r['method']==method)
            values=[100*row['mean_rejected_fraction'][reason] for reason in reasons]
            ax.bar(np.arange(len(reasons))+(j-1)*.24,values,width=.23,color=COLORS[j],label=method)
        ax.set(title=scene.title(),xticks=np.arange(len(reasons)),xticklabels=labels,
               ylim=(0,105),ylabel='Rejected intervals (%)')
        ax.tick_params(axis='x',labelsize=8)
        if not any(v for r in records if r['scene']==scene for v in r['mean_rejected_fraction'].values()):
            ax.text(.5,.45,'All held-out intervals pass\nfor all three methods',transform=ax.transAxes,
                    ha='center',va='center',color=COLORS[0])
    axes[0,0].legend(frameon=False,ncol=3,fontsize=8)
    for ax in list(axes.flat)[len(scenes):]:ax.axis('off')
    fig.suptitle('Event acceptance audit · which criteria reject each interval?',fontsize=15,fontweight='bold')
    fig.supxlabel('Failure reasons may overlap · all methods share thresholds · Fan has no complete raw event input',fontsize=9)
    save(fig,folder/'visualization/event_gate_failures')


def foreground_report(raw,folder):
    """Expose foreground quality beside background-dominated full-frame scores."""
    rows=[];keys=('rgb_l1','psnr','ssim')
    for scene in SCENES:
        for method in ('C1','C2','C3'):
            selected=[r for r in raw if r['scene']==scene and r['method']==method]
            if not selected:continue
            metrics={}
            for key in keys:
                samples=[[f['foreground'][key] for f in r['frame_metrics']] for r in selected]
                if any(v is None for sample in samples for v in sample):
                    metrics[key]=dict(value=None,sd=None,reason='Exact-match infinite PSNR; see frame records')
                    continue
                values=[float(np.mean(sample)) for sample in samples]
                metrics[key]=dict(value=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None,n=len(values))
            rows.append(dict(scene=scene,method=method,metrics=metrics))
    save_json(folder/'foreground_summary.json',dict(scope='Secondary image metrics on fixed reference foreground masks; primary ten-column table remains full-frame.',rows=rows))
    comparison_plot(rows,folder,list(keys),'foreground_quality','Foreground fidelity · fixed reference-image masks')


def significance(rows,root,ablation=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    lookup={(r['scene'],r['method']):r for r in rows};records=[]
    eligible=[s for s in SCENES if (s,'C3') in lookup and lookup[s,'C3']['observation_group']=='RGB+raw_events']
    comparisons=[('main',lookup,first,second,first+' → '+second)
                 for first,second in [('C1','C2'),('C1','C3'),('C2','C3')]]
    if ablation:
        alookup={(r['scene'],r['method']):r for r in ablation}
        comparisons.extend(('ablation',alookup,'full',method,'A0 → A'+str(i))
            for i,method in enumerate(('no_mlp','no_event','no_joint','no_jerk'),1))
    for study,table,first,second,pair in comparisons:
        for key,_,direction in METRICS:
            a=[];b=[];scenes=[]
            for scene in eligible:
                values=[table.get((scene,m),{}).get('metrics',{}).get(key,{}).get('value') for m in (first,second)]
                if any(v is None for v in values):continue
                if key in ('joint_gap_max','position_jerk'):
                    diagonal=json.loads((root/'01_main_comparison/prepared'/scene/'prior.json').read_text())['diagonal']
                    values=[v/diagonal for v in values]
                a.append(values[0]);b.append(values[1]);scenes.append(scene)
            records.append(dict(study=study,pair=pair,metric=key,scenes=scenes,
                coordinate_normalization='divide by fixed source diagonal' if key in ('joint_gap_max','position_jerk') else 'native metric',
                **paired_effect(a,b,direction)))
    tested=[r for r in records if r['status']=='measured']
    for r,adjusted in zip(tested,holm([r['p_raw'] for r in tested])):r['p_holm']=adjusted
    folder=root/'05_significance'
    minimum_p=2/(2**len(eligible)) if eligible else None
    save_json(folder/'results.json',dict(records=records,unit='independent scene; training seeds averaged within scene',
        excluded_groups={'fan':'RGB-only input protocol; one scene, descriptive only'},
        eligible_scenes=eligible,minimum_two_sided_exact_p=minimum_p,
        caveat='Small scene sample; sign exchangeability assumed. Bootstrap intervals are exploratory.',
        multiplicity='Holm across all measured main and ablation metric × method-pair tests; ablations use matched seed 17'))
    lines=['# 配对统计显著性','','统计单位为场景；先在每个场景内平均训练种子。Fan 为 RGB-only，与 RGB＋事件场景分开报告。',
        '正向效应表示符合指标期望方向。关节间隙和三维 jerk 先除以固定场景对角线；bootstrap 区间仅作探索描述。','',
        '| 研究 | 比较 | 指标 | 场景数 | 正向效应 | 探索性 95% 区间 | 原始 p | Holm p |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    chinese={key:label for key,label,_ in METRICS}
    for r in records:
        if r['status']!='measured':continue
        lines.append(f"| {r['study']} | {r['pair']} | {chinese[r['metric']]} | {r['n']} | {r['effect']:.6g} | [{r['ci95'][0]:.6g}, {r['ci95'][1]:.6g}] | {r['p_raw']:.6g} | {r['p_holm']:.6g} |")
    lines+=['',f'当前最多 {len(eligible)} 个可比较场景，精确双侧符号翻转检验最小 p 值为 {minimum_p}；不能通过增加相邻帧数规避这一限制。']
    (folder/'results.md').write_text('\n'.join(lines)+'\n')
    style();selected=[r for r in tested if r['pair']=='C2 → C3'];fig,axes=plt.subplots(3,3,figsize=(13,8),layout='constrained')
    for ax,r in zip(axes.flat,selected):
        effect=r['effect'];lo,hi=r['ci95'];ax.axvline(0,color='#ABB6C0',lw=1)
        ax.plot([lo,hi],[0,0],color=COLORS[0],lw=3);ax.plot(effect,0,'o',color=COLORS[0],ms=7)
        label={'joint_gap_max':'Source-joint gap / scene diagonal',
               'position_jerk':'3D jerk / scene diagonal (s⁻³)'}.get(r['metric'],ENGLISH[r['metric']])
        ax.set_yticks([]);ax.set_ylim(-1,1);ax.set_title(label,fontsize=10,fontweight='bold')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useOffset=False)
        ax.tick_params(axis='x',labelsize=8);ax.xaxis.get_offset_text().set_fontsize(8)
        ax.text(.03,.08,f"n={r['n']} scenes · Holm p={r['p_holm']:.3g}",transform=ax.transAxes,fontsize=9)
    for ax in list(axes.flat)[len(selected):]:ax.axis('off')
    fig.suptitle('C3 versus C2 · paired scene effects and exploratory 95% intervals',fontweight='bold',fontsize=15)
    fig.supxlabel('Positive = desired improvement · four RGB+event scenes · seeds averaged within scene',fontsize=10)
    save(fig,folder/'visualization/paired_effects')


def supplementary(root):
    import matplotlib.pyplot as plt
    sensitivity=[]
    for path in sorted((root/'03_sensitivity/runs').glob('*/results.json')):sensitivity.append(json.loads(path.read_text()))
    if sensitivity:
        style();fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for j,key in enumerate(('rgb_l1','event_relative_l1','position_jerk')):
            for i,data in enumerate(sensitivity):
                vals=[r['metrics'][key]['value'] for r in data['rows']]
                if any(v is None for v in vals) or abs(vals[0])<1e-12:continue
                delta=(np.asarray(vals)/vals[0]-1)*100
                axes[j].plot([r['gain'] for r in data['rows']],delta,'o-',ms=4,color=COLORS[i],label=data['scene'].title())
            axes[j].axhline(0,color='#9BA8B4',lw=.7)
            axes[j].set(title=ENGLISH[key],xlabel='MLP inference residual strength',ylabel='Change from zero gain (%)')
        axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle('Validation-only sensitivity · fixed trained weights · no test-based selection',fontsize=14,fontweight='bold')
        save(fig,root/'03_sensitivity/visualization/residual_strength')
        save_json(root/'03_sensitivity/results.json',dict(scenes=sensitivity))
        lines=['# 推理残差强度敏感度','','固定种子 17 已训练权重，仅在验证段评估；未重新训练，也不据此选择测试成绩。','',
            '| 场景 | MLP 强度 | RGB L1 ↓ | 事件相对 L1 ↓ | 三维 jerk ↓ |','| --- | --- | --- | --- | --- |']
        for data in sensitivity:
            for row in data['rows']:
                vals=[row['metrics'][k]['value'] for k in ('rgb_l1','event_relative_l1','position_jerk')]
                cells=['N/A' if v is None else f'{v:.6g}' for v in vals]
                lines.append('| '+data['scene']+f" | {row['gain']:.2f} | "+' | '.join(cells)+' |')
        (root/'03_sensitivity/results.md').write_text('\n'.join(lines)+'\n')
    efficiency=root/'04_efficiency/results.json'
    if efficiency.exists():
        data=json.loads(efficiency.read_text())['runs'];style();fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
        for ax,key,label,scale in zip(axes,('median_gpu_ms','mean_training_wall_seconds','maximum_training_peak_gpu_bytes'),
                ('Warmed rendering (ms)','Mean training + validation (s)','Max peak allocation (GiB)'),(1,1,1/2**30)):
            for i,method in enumerate(('C1','C2','C3')):
                values=[next((r[key]*scale for r in data if r['scene']==s and r['method']==method),np.nan) for s in SCENES]
                ax.bar(np.arange(5)+(i-1)*.24,values,width=.23,label=method,color=COLORS[i])
            ax.set(xticks=np.arange(5),xticklabels=[s.title() for s in SCENES],ylabel=label);ax.tick_params(axis='x',rotation=30)
        axes[0].legend(frameon=False);fig.suptitle('Measured efficiency · RTX 2080 Ti · 640 × 640',fontsize=16,fontweight='bold')
        fig.supxlabel('Inference: seed 17, 20 warm-up frames, 5 repeats · fixed C1: validation only · C2/C3: 200 updates',fontsize=9)
        save(fig,root/'04_efficiency/visualization/efficiency')
        lines=['# 效率实验','','原生 640×640 GS。推理计时为种子 17，二十帧预热后五轮；C2/C3 各训练 200 次更新，训练耗时为已登记种子的平均值，包含验证，排除共同预处理和最终测试。',
            '推理指标是包围运动＋GS 渲染调用的 CUDA-event 耗时，含调用内搬运与提交间隙；不等同于纯 kernel 计时或网页端到端帧率。',
            '峰值显存为各训练种子中最大的 PyTorch 已分配峰值，非整卡占用。C1 没有优化更新，其训练栏只有初始验证耗时。','',
            '| 场景 | 方法 | 高斯数 | 可训练参数 | 平均训练＋验证 s | GPU 中位数 ms | GPU P95 ms | 最大峰值 GiB |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in data:
            sd=row['sd_training_wall_seconds'];duration=f"{row['mean_training_wall_seconds']:.2f}"+(f' ± {sd:.2f}' if sd is not None else '')
            lines.append(f"| {row['scene']} | {row['method']} | {row['gaussians']:,} | {row['trainable_parameters']} | {duration} | {row['median_gpu_ms']:.3f} | {row['p95_gpu_ms']:.3f} | {row['maximum_training_peak_gpu_bytes']/2**30:.3f} |")
        (root/'04_efficiency/results.md').write_text('\n'.join(lines)+'\n')


def generate(root):
    from .audit_results import audit
    root=Path(root);audit(root)
    main=read_runs(root/'01_main_comparison');rows=aggregate(main)
    if not rows:raise ValueError('No completed measured runs')
    tables(rows,root/'01_main_comparison','五场景主实验／对照')
    for keys,stem,title in [(['rgb_l1','psnr','ssim'],'image_quality','Image fidelity · held-out temporal block'),
        (['event_relative_l1','event_pass','visible_missing'],'event_coverage','Event consistency and visible coverage'),
        (['joint_gap_max','angular_jerk','position_jerk'],'motion_constraints','Motion and source-joint diagnostics')]:
        comparison_plot(rows,root/'01_main_comparison',keys,stem,title)
    qualitative(main,root/'01_main_comparison')
    learning_curves(main,root/'01_main_comparison')
    event_gate_diagnostics(main,root/'01_main_comparison')
    foreground_report(main,root/'01_main_comparison')
    ablation=read_runs(root/'02_ablation');aggregated=[]
    if ablation:
        ablation += [r for r in main if r['method']=='C3' and r['seed']==17]
        aggregated=aggregate(ablation,True);tables(aggregated,root/'02_ablation','五场景消融 · 种子 17')
        comparison_plot(aggregated,root/'02_ablation',['rgb_l1','event_relative_l1','position_jerk'],'ablation',
            'Controlled retraining ablations · matched 200-step budget',methods=['full','no_mlp','no_event','no_joint','no_jerk'])
        comparison_plot(aggregated,root/'02_ablation',['joint_gap_max','angular_jerk','visible_missing'],'ablation_constraints',
            'Constraint ablations · source joints, angular motion and coverage',methods=['full','no_mlp','no_event','no_joint','no_jerk'])
    significance(rows,root,aggregated);supplementary(root)
    if (root/'01_main_comparison/runs/fan_C1_s17_full/evaluation.json').exists():
        from .precision_control import main as precision
        precision(root)
    inventory=root/'01_main_comparison/server_inventory.json'
    if inventory.exists():inventory_plots(json.loads(inventory.read_text()),root)
    from .diagnostics import fan_training_fit
    fan_training_fit(root)
    print('Wrote measured reports and visualizations:',root,flush=True)
