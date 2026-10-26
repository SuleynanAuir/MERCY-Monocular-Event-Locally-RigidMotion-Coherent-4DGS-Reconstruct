"""Report exact UI captures without re-rendering or selectively improving images."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .protocol import save_json, METRICS


def generate(root, output):
    from .visualize import style, save
    import matplotlib.pyplot as plt
    inventory=json.loads((root/'input-inventory.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; images=[]; verification=[]
    for scene in inventory['scenes']:
        name=scene['scene']; paths=[root/'ui-evaluation/full'/f'{name}-v1-{v}'/'clay' for v in ('v2','v3')]
        if not all((p/'results.json').exists() for p in paths):continue
        captures=[json.loads((p/'results.json').read_text()) for p in paths]
        if any(c['scope']!='full_sequence' for c in captures):raise ValueError('Incomplete UI sequence')
        if [r['frame'] for r in captures[0]['frames']]!=[r['frame'] for r in captures[1]['frames']]:
            raise ValueError('Pair timelines differ')
        width=captures[0]['request']['width']
        max_difference=0
        for frame in captures[0]['frames']:
            i=frame['frame']
            panels=[np.asarray(Image.open(p/'frames'/f'{i:04d}.jpg').convert('RGB')) for p in paths]
            max_difference=max(max_difference,int(np.abs(panels[0][:,:width].astype(int)-panels[1][:,:width]).max()))
        if max_difference:raise ValueError(f'{name}: V1 changed between V2/V3 pairings')
        verification.append(dict(scene=name,v1_pixel_identical_all_frames=True,frame_count=len(captures[0]['frames'])))
        test=set(scene['split']['frames']['test'])
        for version,cap,side in [('V1',captures[0],'left'),('V2',captures[0],'right'),('V3',captures[1],'right')]:
            measured=[row['metrics'][side] for row in cap['frames'] if row['frame'] in test]
            mean={key:float(np.mean([r[key] for r in measured])) for key in ('rgb_l1','psnr','ssim')}
            native_path=root/'native-audit'/name/version.lower()/'evaluation.json'
            native=json.loads(native_path.read_text()) if native_path.exists() else None
            for key,_,_ in METRICS[3:]:
                mean[key]=native['metrics'][key]['value'] if native else None
            rows.append(dict(scene=name,version=version,metrics=mean,frames=len(measured),
                full_sequence={key:float(np.mean([r['metrics'][side][key] for r in cap['frames']])) for key in ('rgb_l1','psnr','ssim')},
                native_audit_identity=native.get('ui_asset_identity') if native else None,
                native_metric_details=native.get('metrics') if native else None,
                artifact_identity=cap['status']['scene'].get('asset_identity')))
        i=scene['video']['frame_count']//2
        ref=np.asarray(Image.open(paths[0]/'references'/f'{i:04d}.image').convert('RGB'))[:,:width]
        pairs=[np.asarray(Image.open(p/'frames'/f'{i:04d}.jpg').convert('RGB')) for p in paths]
        images.append((name,i,[ref,pairs[0][:,:width],pairs[0][:,width:],pairs[1][:,width:]]))
    if not rows:raise ValueError('No complete actual-UI captures')
    save_json(output/'results.json',dict(scope='Actual UI clay pixels; fixed temporal test subset and full-sequence supplement',
        completed_scenes=len(images),expected_scenes=5,rows=rows,verification=verification))
    lines=['# 实际 UI 白模／清晰柔光对照','',
        f'当前完成 **{len(images)}/5 个场景**。V1/C1＝原始高斯与基础绑定；V2/C2＝观测校正；V3/C3＝结构／表面修复及适用的局部补偿。未完成场景不填入旧数值。','',
        '图像直接来自本轮 UI WebSocket；参考图来自同一网页的 RGB 面板。主表为预先固定的时间测试段，完整序列分项保存在 JSON。',
        '**这些 RGB 指标衡量白模展示与 RGB 参考的画面差异，包含材质／光照差异，不能单独解释为几何重建精度。**','',
        'RGB 三列采用实际网页白模；其余指标是**同一冻结 UI 高斯资产**的原生强度／三维运动审计。原生事件渲染使用原传感器背景、640 px，不将白模展示光照当作传感器观测。缺少独立背面真值时保留 N/A。','',
        '| 场景 | 版本 | '+' | '.join(title for _,title,_ in METRICS)+' |',
        '|---|---|'+'---:|'*len(METRICS)]
    for row in rows:
        m=row['metrics'];values=['N/A' if m.get(key) is None else f'{m[key]:.6f}' for key,_,_ in METRICS]
        details=row.get('native_metric_details') or {}
        event=details.get('event_pass',{})
        if event.get('status') == 'measured':
            values[4]=f"{event['passed']}/{event['total']} ({100*event['value']:.2f}%)"
        lines.append(f"| {row['scene'].title()} | {row['version']} | "+' | '.join(values)+' |')
    lines+=['','缺失率以 0–1 比例表示；可见缺失率为原始 RGB 前景中 alpha < 0.5 的像素比例，仅是覆盖代理。关节间隙和三维 jerk 使用各场景源坐标单位，角 jerk 使用 rad/s³；不能跨场景直接平均这些绝对量。无完整原始事件的 Fan 不给出事件成绩。',
            '', '固定选取每段中间帧制作下图，不依据评分选帧。每个场景的全部帧均验证：V1 在 V1/V2 与 V1/V3 两种配对中逐像素相同。',
            '', '![Actual UI comparison](visualization/actual_ui_comparison.png)', '',
            '事件、三维轨迹和关节指标需使用相应冻结版本的原生模型审计，不能由白模截图或彩色事件预览推算。旧运动适配实验数据不进入本表。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    style()
    fig,axes=plt.subplots(len(images),4,figsize=(12,2.9*len(images)),squeeze=False)
    for row,(name,frame,panels) in enumerate(images):
        # A single crop from the union of all four panels reduces whitespace;
        # exactly the same crop is applied to every version and reference.
        foreground=np.any(np.stack(panels)<242,axis=(0,3))
        yy,xx=np.nonzero(foreground)
        if len(xx):
            x0,x1=max(0,xx.min()-20),min(width,xx.max()+21)
            y0,y1=max(0,yy.min()-20),min(width,yy.max()+21)
        else:x0,y0,x1,y1=0,0,width,width
        for col,pixels in enumerate(panels):
            ax=axes[row,col];ax.imshow(pixels[y0:y1,x0:x1]);ax.grid(False);ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(['UI RGB reference','V1 · baseline','V2 · observation correction','V3 · full rebuilt stage'][col],fontsize=11,fontweight='bold')
            if col==0:ax.set_ylabel(name.title()+f'\nframe {frame}',fontsize=10,fontweight='bold')
    fig.suptitle('Actual UI output · white clay / soft lighting',fontsize=17,fontweight='bold',y=.995)
    fig.text(.5,.012,'Same frame and camera · WebSocket JPEG pixels · shared crop for layout only · scores use uncropped frames',ha='center',fontsize=9)
    fig.subplots_adjust(left=.06,right=.995,bottom=.06,top=1-.85/(2.9*len(images)),wspace=.03,hspace=.08)
    save(fig,output/'visualization/actual_ui_comparison')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    names=[x[0] for x in images]; colors=['#8697AC','#DDA45D','#169C92']
    for ax,key,title in zip(axes,('rgb_l1','psnr','ssim'),('RGB L1 ↓','PSNR (dB) ↑','SSIM ↑')):
        x=np.arange(len(names))
        for j,version in enumerate(('V1','V2','V3')):
            values=[next(r['metrics'][key] for r in rows if r['scene']==s and r['version']==version) for s in names]
            ax.bar(x+(j-1)*.24,values,.23,label=version,color=colors[j])
        ax.set_xticks(x, [s.title() for s in names],rotation=25);ax.set_title(title);ax.legend(frameon=False)
    fig.suptitle('UI display discrepancy against RGB · fixed test frames',fontweight='bold')
    save(fig,output/'visualization/ui_display_metrics')
    # Raw signed changes remain visible even when V3 is worse. Color scale is
    # clipped only for legibility, while numeric annotations retain full values.
    change=np.full((len(names),len(METRICS)),np.nan)
    for i,name in enumerate(names):
        v1=next(r['metrics'] for r in rows if r['scene']==name and r['version']=='V1')
        v3=next(r['metrics'] for r in rows if r['scene']==name and r['version']=='V3')
        for j,(key,_,direction) in enumerate(METRICS):
            a,b=v1[key],v3[key]
            if a is not None and b is not None:
                if key=='event_pass':change[i,j]=100*(b-a)
                elif abs(a)>1e-12:change[i,j]=100*direction*(b-a)/abs(a)
                elif abs(b)<1e-12:change[i,j]=0.
    fig,ax=plt.subplots(figsize=(14,4.6),layout='constrained')
    cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('#EDF1F5')
    heat=ax.imshow(np.ma.masked_invalid(change),cmap=cmap,vmin=-20,vmax=20,aspect='auto')
    ax.grid(False)
    ax.set_xticks(range(len(METRICS)),['RGB L1','PSNR','SSIM','Event L1','Event pass*','Visible missing','Rear missing','Joint gap','Angular jerk','3D jerk'],rotation=30,ha='right')
    ax.set_yticks(range(len(names)),[s.title() for s in names])
    for i in range(len(names)):
        for j in range(len(METRICS)):
            value=change[i,j]
            label='N/A' if np.isnan(value) else (f'{value:+.1e}' if abs(value)>=1000 else f'{value:+.2f}')
            ax.text(j,i,label,ha='center',va='center',fontsize=9,color='#17212B')
    ax.set_title('V3 versus V1 · signed improvement (%) · positive is better',fontweight='bold',pad=15)
    ax.set_xlabel('* Event pass: percentage-point difference. RGB: actual clay UI; other columns: native asset audit. Color clipped at ±20.',labelpad=12,fontsize=9)
    fig.colorbar(heat,ax=ax,shrink=.75,label='Improvement (%)')
    save(fig,output/'visualization/v3_signed_changes')
    if 'lego' in names:
        # Fixed source-image ROI, identical across versions and time; never
        # search for a frame/crop that maximizes the apparent improvement.
        roi=(.10,.20,.65,.65)
        source=root/'ui-evaluation/full'
        pairs=[source/f'lego-v1-{v}'/'clay' for v in ('v2','v3')]
        capture=json.loads((pairs[0]/'results.json').read_text())
        n=len(capture['frames']);w=capture['request']['width']
        frames=[0,n//2,n-1];x0,y0,x1,y1=[round(x*w) for x in roi]
        fig,axes=plt.subplots(3,3,figsize=(11,8),squeeze=False)
        for i,frame in enumerate(frames):
            raw=[np.asarray(Image.open(p/'frames'/f'{frame:04d}.jpg')) for p in pairs]
            panels=[raw[0][:,:w],raw[0][:,w:],raw[1][:,w:]]
            for j,pixels in enumerate(panels):
                axes[i,j].imshow(pixels[y0:y1,x0:x1],interpolation='none')
                axes[i,j].grid(False);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                if i==0:axes[i,j].set_title(('V1','V2','V3')[j],fontweight='bold')
                if j==0:axes[i,j].set_ylabel(f"Frame {frame}\n{capture['frames'][frame]['time_seconds']:.3f} s")
        fig.suptitle('Lego moving-arm detail · actual UI pixels',fontweight='bold',fontsize=16)
        fig.text(.5,.015,'Fixed ROI and first/middle/last frames · no sharpening or version-specific crop',ha='center',fontsize=10)
        fig.subplots_adjust(top=.90,bottom=.06,left=.075,right=.99,wspace=.025,hspace=.06)
        save(fig,output/'visualization/lego_motion_detail')
        save_json(output/'visualization/lego_motion_detail_protocol.json',dict(
            source='Actual UI JPEG panels',roi_normalized=roi,frames=frames,
            use='Visualization only; main scores use the original uncropped images'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();generate(args.root,args.output)


if __name__=='__main__':main()
