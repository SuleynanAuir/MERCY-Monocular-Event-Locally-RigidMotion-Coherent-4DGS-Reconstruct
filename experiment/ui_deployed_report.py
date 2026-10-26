"""Report original deployed and archived UI evidence with separate provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .protocol import METRICS, save_json


def generate(root):
    from .visualize import style, save
    import matplotlib.pyplot as plt
    rows=[];images={mode:[] for mode in ('gaussian','clay')}
    archive=root/'historical_export_review'
    archived=json.loads((archive/'results.json').read_text())
    for r in archived['rows']:
        scene,mode=r['scene'],r['mode'];i=r['frame_count']//2
        raw=np.asarray(Image.open(archive/scene/f'{mode}-original-ui-{i:04d}.png'))
        w,h,y=[r['layout'][k] for k in ('panel_width','image_height','header')]
        panels=[raw[y:y+h,k*w:(k+1)*w] for k in (2,0,1)]
        images[mode].append((scene,'archived UI',i,panels))
        for version in ('V1','V3'):
            rows.append(dict(scene=scene,mode=mode,version=version,source='archived_lossy_ui_export',
                             frames=r['frame_count'],metrics=r['means'][version],
                             metric_scope=r['scope'],source_sha256=r['source_sha256']))
    live=root/'original_live_review'
    verified=json.loads((live/'verification.json').read_text())
    if not verified['deployment_unchanged']:raise ValueError('Deployment identity failed')
    for scene in verified['completed_scenes']:
        for mode in ('gaussian','clay'):
            p=live/'captures'/scene/mode;r=json.loads((p/'results.json').read_text())
            i=len(r['frames'])//2;w=r['request']['width']
            raw=np.asarray(Image.open(p/'frames'/f'{i:04d}.jpg'))
            ref=np.asarray(Image.open(p/'references'/f'{i:04d}.image'))[:,:w]
            images[mode].append((scene,'original live UI',i,[ref,raw[:,:w],raw[:,w:]]))
            for version,side in (('V1','left'),('V3','right')):
                rows.append(dict(scene=scene,mode=mode,version=version,source='original_deployment_websocket',
                    frames=len(r['frames']),metrics=r['means'][side],metric_scope=r['metric_scope']))
    save_json(root/'pixel_review.json',dict(
        scope='Original UI evidence review; live and historical video metrics remain separate',
        rows=rows,full_five_scene_current_deployment_verified=False,
        current_deployment_verified_scenes=verified['completed_scenes'],
        archived_video_scenes=sorted({r['scene'] for r in rows if r['source']=='archived_lossy_ui_export'}),
        v2_available=False,native_metrics_complete=False,training_performed=False))
    for kind,folder,title in [('original_deployment_websocket',live,'原部署 UI 全段像素结果'),
                              ('archived_lossy_ui_export',archive,'原 UI 历史导出视频全段复核')]:
        selected=[r for r in rows if r['source']==kind]
        lines=['# '+title,'',
            '仅复核现成 UI 结果，没有重新训练或替换左右模型。GS 与白模分别报告。',
            '白模对 RGB 的 L1／PSNR／SSIM 含材质与照明差异；不保证视觉更美观时三项都提升。完整视频已参与展示，不能称为未曝光测试集。','']
        if kind=='archived_lossy_ui_export':
            lines+=['来源为原 UI 导出的有损 WebM；按原布局裁出同一前视画面，以同一固定掩码排除标题、方框边线、标签和压缩边缘。SSIM 排除与掩码相交的整个窗口。数值不能与无框、无裁剪的实时 JPEG 直接合并。最新部署资产身份尚未核实。','']
        else:
            lines+=['来源为原项目目录、原场景配置和原左右资产，通过 WebSocket 采集的完整 640×640 JPEG。采集前后所有代码、模型与参考文件哈希一致。','']
        lines+=['| 场景 | 模式 | 版本 | 帧数 | '+' | '.join(t for _,t,_ in METRICS)+' |',
                '|---|---|---|---:|'+'---:|'*len(METRICS)]
        for r in selected:
            cells=[f"{r['metrics'][key]:.6f}" if key in r['metrics'] else 'N/A' for key,_,_ in METRICS]
            lines.append(f"| {r['scene']} | {r['mode']} | {r['version']} | {r['frames']} | "+' | '.join(cells)+' |')
        lines+=['','N/A：本次像素复核不能从视频或 JPEG 计算原生事件、三维轨迹、关节或独立背面覆盖；未套用替代模型的审计值。原部署中未单列 V2，未虚构 V2 结果。']
        (folder/'results.md').write_text('\n'.join(lines)+'\n')
    style()
    for mode,items in images.items():
        fig,axes=plt.subplots(len(items),3,figsize=(11,2.8*len(items)),squeeze=False)
        for row,(scene,source,frame,panels) in enumerate(items):
            foreground=np.any(np.stack(panels)<242,axis=(0,3));yy,xx=np.nonzero(foreground)
            h,w=panels[0].shape[:2]
            x0,x1=max(0,int(xx.min())-14),min(w,int(xx.max())+15)
            y0,y1=max(0,int(yy.min())-14),min(h,int(yy.max())+15)
            for j,pixels in enumerate(panels):
                ax=axes[row,j];ax.imshow(pixels[y0:y1,x0:x1]);ax.grid(False)
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:ax.set_title(('UI RGB','V1 in original UI','V3 in original UI')[j],fontweight='bold')
                if j==0:ax.set_ylabel(scene.title()+f'\n{source}\nframe {frame}',fontsize=9)
        fig.suptitle('Original UI evidence · '+('Gaussian' if mode=='gaussian' else 'white clay / soft lighting'),fontweight='bold',fontsize=16)
        fig.subplots_adjust(top=.95,bottom=.04,left=.12,right=.99,wspace=.025,hspace=.12)
        fig.text(.5,.012,'Fixed middle frames · one shared layout crop per scene · original overlays retained · historical/live sources labeled',ha='center',fontsize=8)
        save(fig,root/'visualization'/('original_ui_'+mode))
    # Within-scene changes only: never average mixed archival/live protocols.
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    names=['lego','fan','lamp','metronome','microphone']
    for ax,key,title in zip(axes,('rgb_l1','psnr','ssim'),('RGB L1 reduction (%)','PSNR increase (dB)','SSIM increase')):
        for j,mode in enumerate(('gaussian','clay')):
            changes=[]
            for s in names:
                pair={r['version']:r['metrics'][key] for r in rows if r['scene']==s and r['mode']==mode}
                changes.append(100*(pair['V1']-pair['V3'])/pair['V1'] if key=='rgb_l1' else pair['V3']-pair['V1'])
            ax.bar(np.arange(5)+(j-.5)*.32,changes,.30,label=mode,color=('#278F87','#9C8BC1')[j])
        ax.axhline(0,color='#718096',lw=.8);ax.set_xticks(np.arange(5),[n.title() for n in names],rotation=35,ha='right')
        ax.set_title(title);ax.legend(frameon=False)
    fig.suptitle('V3 vs V1 within each original UI source · positive is better',fontweight='bold')
    save(fig,root/'visualization/within_source_changes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();generate(args.root)
