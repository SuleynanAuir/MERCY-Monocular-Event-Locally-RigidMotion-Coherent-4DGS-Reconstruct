"""Separate UI display enhancement from geometry using identical frozen assets.

The backend is the production UI renderer. Every tested full-quality frame must
first match its actual WebSocket JPEG byte-for-byte. Controls change only the
normal finish/supersampling or JPEG quality; native model audits stay unchanged.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .protocol import save_json, METRICS
from .ui_capture import decode_packet,score


def run(root,scene):
    from utils.gaussian_pair_renderer import GaussianPairRenderer
    from utils.viewer_cache import EncodedFrameCache
    from utils.viewer_state import validate_request
    capture=root/'ui-evaluation/full'/f'{scene}-v1-v3'/'clay'
    record=json.loads((capture/'results.json').read_text())
    inventory=json.loads((root/'input-inventory.json').read_text())
    entry=next(s for s in inventory['scenes'] if s['scene']==scene)
    frames=entry['split']['frames']['test'];size=record['request']['width']
    renderer=GaussianPairRenderer(root/scene/'pair-v1-v3.json','cpu');renderer.load()
    if renderer.info['asset_identity']!=record['status']['scene']['asset_identity']:
        raise ValueError('Display control renderer/asset differs from actual UI capture')
    output=root/'ui-evaluation/display-controls'/scene;output.mkdir(parents=True,exist_ok=False)
    rows=[];normals=renderer.detail_normals;supersamples=list(renderer.supersamples)
    for frame in frames:
        requested=validate_request(dict(record['request'],id=frame,frame=frame),record['status']['scene']['frame_count'])
        _,jpeg,pair=decode_packet(renderer.render(requested),frame,size)
        frozen=next(r for r in record['frames'] if r['frame']==frame)
        if hashlib.sha256(jpeg).hexdigest()!=frozen['jpeg_sha256']:
            raise ValueError(f'Full backend output differs from actual UI WebSocket at {scene}/{frame}')
        reference=np.asarray(Image.open(capture/'references'/f'{frame:04d}.image').convert('RGB'))[:,:size]
        base_score=score(reference,pair[:,size:])
        for name,quality,enhanced in [('full',95,True),('no_display_enhancement',95,False),('jpeg_85',85,True),('jpeg_78',78,True)]:
            renderer.detail_normals=normals if enhanced else None
            renderer.supersamples=supersamples if enhanced else [1,1]
            # Explicit cache reset prevents returning enhanced pixels for the
            # unenhanced condition, whose other request fields are identical.
            renderer.frames=EncodedFrameCache(max_bytes=32*1024**2,max_entries=32)
            request=dict(requested,quality=quality)
            meta,data,pixels=decode_packet(renderer.render(request),frame,size)
            if name=='full' and data!=jpeg:raise ValueError('Fresh frame does not match frozen pixels')
            values=base_score if name=='full' else score(reference,pixels[:,size:])
            rows.append(dict(frame=frame,condition=name,metrics=values,jpeg_sha256=hashlib.sha256(data).hexdigest(),metadata=meta))
            dest=output/name;dest.mkdir(exist_ok=True)
            (dest/f'{frame:04d}.jpg').write_bytes(data)
        renderer.detail_normals=normals;renderer.supersamples=supersamples
        renderer.frames=EncodedFrameCache(max_bytes=32*1024**2,max_entries=32)
    save_json(output/'results.json',dict(scene=scene,asset_identity=renderer.info['asset_identity'],
        all_full_frames_match_websocket=True,scope='Display-only controls, frozen geometry and motion; not a retrained method ablation',
        normal_finish_available=normals is not None,original_supersamples=supersamples,frames=frames,rows=rows))


def report(root,output):
    from .visualize import style,save
    import matplotlib.pyplot as plt
    scenes=tuple(s for s in ('lego','fan','lamp','metronome','microphone')
                 if (root/'ui-evaluation/display-controls'/s/'results.json').exists())
    if not scenes:raise ValueError('No completed display controls')
    records=[json.loads((root/'ui-evaluation/display-controls'/s/'results.json').read_text()) for s in scenes]
    ablation=[];sensitivity=[]
    for rec in records:
        for condition in ('full','no_display_enhancement','jpeg_85','jpeg_78'):
            rows=[r for r in rec['rows'] if r['condition']==condition]
            result=dict(scene=rec['scene'],condition=condition,n=len(rows),
                normal_finish_available=rec['normal_finish_available'],
                metrics={k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('rgb_l1','psnr','ssim')})
            native=json.loads((root/'native-audit'/rec['scene']/'v3/evaluation.json').read_text())
            result['native_metrics_unchanged']={k:v for k,v in native['metrics'].items() if k not in ('rgb_l1','psnr','ssim')}
            if condition in ('full','no_display_enhancement'):ablation.append(result)
            if condition!='no_display_enhancement':sensitivity.append(result)
    for folder,title,rows,scope in [('02_ablation','显示增强消融',ablation,'冻结 V3 几何与运动，仅同时移除法线细节增强及 2× 超采样。仍使用同一白模基础材质；不是重新训练的 MLP／事件／关节消融。'),
        ('03_sensitivity','JPEG 展示参数敏感度',sensitivity,'冻结 V3 资产、640×640 相机和白模参数，仅比较 JPEG 质量 78、85、95。不是 MLP 权重、训练参数或采样密度敏感度。')]:
        dest=output/folder
        save_json(dest/'results.json',dict(scope=scope,completed_scenes=len(scenes),expected_scenes=5,
            full_output_matches_websocket=True,rows=rows))
        lines=['# '+title,'',scope,'',
            f'当前完成 {len(scenes)}/5 个场景。没有法线增强资产的场景只移除超采样，不虚构法线增强的贡献。','',
            '每个测试帧先验证原版渲染与实际 UI WebSocket JPEG 完全一致，再运行显示对照。前三项为当前条件画面指标；原生事件／三维指标来自相同冻结 V3 资产，保持不变。','',
            '| 场景 | 条件 | '+' | '.join(title for _,title,_ in METRICS)+' |',
            '|---|---|'+'---:|'*len(METRICS)]
        for r in rows:
            values=[]
            for key,_,_ in METRICS:
                value=r['metrics'].get(key) if key in r['metrics'] else r['native_metrics_unchanged'][key]['value']
                if key=='event_pass' and value is not None:
                    event=r['native_metrics_unchanged'][key];values.append(f"{event['passed']}/{event['total']} ({100*value:.2f}%)")
                else:values.append('N/A' if value is None else f'{value:.6f}')
            lines.append(f"| {r['scene']} | {r['condition']} | "+' | '.join(values)+' |')
        (dest/'results.md').write_text('\n'.join(lines)+'\n')
        style();fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
        conditions=list(dict.fromkeys(r['condition'] for r in rows));x=np.arange(len(scenes))
        for ax,key,title2 in zip(axes,('rgb_l1','psnr','ssim'),('RGB L1 ↓','PSNR ↑','SSIM ↑')):
            for j,condition in enumerate(conditions):
                ys=[next(r['metrics'][key] for r in rows if r['scene']==s and r['condition']==condition) for s in scenes]
                ax.plot(x,ys,'o-',label=condition,linewidth=1.4,markersize=4)
            ax.set_xticks(x,[s.title() for s in scenes],rotation=25);ax.set_title(title2);ax.legend(fontsize=8,frameon=False)
        fig.suptitle('Frozen V3 · '+('display enhancement control' if folder=='02_ablation' else 'JPEG quality sensitivity'),fontweight='bold')
        save(fig,dest/'visualization/display_controls')
    fig,axes=plt.subplots(len(records),2,figsize=(8,max(4,2.6*len(records))),squeeze=False)
    for i,rec in enumerate(records):
        f=rec['frames'][len(rec['frames'])//2]
        for j,condition in enumerate(('no_display_enhancement','full')):
            img=np.asarray(Image.open(root/'ui-evaluation/display-controls'/rec['scene']/condition/f'{f:04d}.jpg'))[:,640:]
            axes[i,j].imshow(img);axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);axes[i,j].grid(False)
            if i==0:axes[i,j].set_title(('Base clay','UI full soft lighting')[j])
            if j==0:axes[i,j].set_ylabel(rec['scene'].title())
    fig.subplots_adjust(top=.94,bottom=.02,left=.08,right=.99,hspace=.06,wspace=.04)
    fig.suptitle('Same V3 geometry and motion · display-only ablation',fontweight='bold')
    save(fig,output/'02_ablation/visualization/same_model_display_ablation')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--scene',choices=('lego','fan','lamp','metronome','microphone'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.scene:run(args.root,args.scene)
    else:report(args.root,args.output or args.root/'report')


if __name__=='__main__':main()
