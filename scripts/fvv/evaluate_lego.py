"""Evaluate trained Hybrid on fixed-camera Lego development splits."""
import argparse
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

import hybrid
from common import digest


def evaluate(args):
    torch.set_num_threads(4)
    renderer=hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=args.model.resolve()))
    baseline = hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=args.compare_model.resolve())) if args.compare_model else None
    from utils.loss_utils import ssim
    dataset=args.dataset.resolve();meta=json.loads((dataset/'provenance.json').read_text())
    entries=json.loads((dataset/('transforms_test.json' if args.split == 'validation' else 'transforms_holdout.json')).read_text())['frames']
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    pose=dict(position=[0.,0.,0.],yaw=0.,pitch=0.,fov=meta['fov_degrees'],width=meta['width'],height=meta['height'])
    rows=[]; panel=Image.new('RGB',(meta['width']*(3 if baseline else 2),meta['height']*3+90))
    selected=[0,len(entries)//2,len(entries)-1]
    for i,item in enumerate(entries):
        frame={**pose,'time':float(item['time'])}
        image=renderer.frame(frame,raw=True).clamp(0,1)
        with Image.open(dataset/(item['file_path']+'.png')) as f:reference=np.asarray(f.convert('RGB')).copy()
        target=torch.from_numpy(reference).permute(2,0,1).float()/255
        mask=target.mean(0)>8/255
        mse=(image-target).square().mean().item();fg_mse=(image[:,mask]-target[:,mask]).square().mean().item()
        error=image-target
        dx=error[:,:,1:]-error[:,:,:-1];dy=error[:,1:,:]-error[:,:-1,:]
        edge_mae=.5*(dx[:,mask[:,1:]|mask[:,:-1]].abs().mean()+dy[:,mask[1:,:]|mask[:-1,:]].abs().mean())
        rows.append(dict(time=item['time'],psnr=-10*math.log10(max(mse,1e-12)),
                    foreground_psnr=-10*math.log10(max(fg_mse,1e-12)),ssim=float(ssim(image,target)),
                    mae=float((image-target).abs().mean()),edge_mae=float(edge_mae)))
        if i in selected:
            index=selected.index(i); top=index*(meta['height']+30)
            draw=ImageDraw.Draw(panel)
            label='Original RGB | baseline | candidate' if baseline else 'Original RGB | trained Hybrid'
            draw.text((10,top+5),f'{label}: t={item["time"]:.3f}s',fill='white')
            panel.paste(Image.fromarray(reference),(0,top+30))
            if baseline:
                previous=baseline.frame(frame,raw=True).clamp(0,1)
                panel.paste(Image.fromarray((previous.permute(1,2,0).numpy()*255).round().astype('uint8')),(meta['width'],top+30))
            panel.paste(Image.fromarray((image.permute(1,2,0).numpy()*255).round().astype('uint8')),(meta['width']*(2 if baseline else 1),top+30))
    panel.save(root/'holdout-comparison.jpg',quality=95)
    timings=[]
    for i in range(125):
        frame={**pose,'time':meta['time_duration'][1]*(i%100)/99,'yaw':15*math.sin(i*.1)}
        start=time.perf_counter();renderer.frame(frame); elapsed=(time.perf_counter()-start)*1000
        if i>=5:timings.append(elapsed)
    for i,yaw in enumerate([-20,0,20]):
        (root/f'novel-view-{i}.jpg').write_bytes(renderer.frame({**pose,'time':3.,'yaw':yaw}))
    first=renderer.frame({**pose,'time':meta['time_duration'][0]},raw=True)
    last=renderer.frame({**pose,'time':meta['time_duration'][1]},raw=True)
    shifted=renderer.frame({**pose,'time':meta['time_duration'][0],'position':[.1,0.,0.]},raw=True)
    record=dict(asset_kind='trained real Lego observation dataset',supervision=meta['supervision'],
        model_sha256=digest(args.model/'model.pth'),
        comparison_model_sha256=digest(args.compare_model/'model.pth') if baseline else None,
        dataset_provenance_sha256=digest(dataset/'provenance.json'),
        evaluation_scope=meta['split_scope'],split=args.split,frame_count=len(rows),
        edge_metric='mean absolute forward-gradient error on reference foreground plus boundary; lower is better',
        means={k:float(np.mean([r[k] for r in rows])) for k in ['psnr','foreground_psnr','ssim','mae','edge_mae']},
        rows=rows,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        gaussian_count=int(len(renderer.model.get_xyz)+len(renderer.model.get_static_xyz)),
        dynamic_count=int(len(renderer.model.get_xyz)),static_count=int(len(renderer.model.get_static_xyz)),
        time_change_mean_abs=float((first-last).abs().mean()),
        camera_translation_change_mean_abs=float((first-shifted).abs().mean()),
        benchmark=dict(width=pose['width'],height=pose['height'],samples_ms=timings,
                       mean_ms=float(np.mean(timings)),p95_ms=float(np.percentile(timings,95)),
                       fps=1000/float(np.mean(timings)),scope='render+CPU copy+JPEG; no network/browser'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (root/'evaluation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k not in ['rows','benchmark']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','model','dataset','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compare-model',type=Path)
    p.add_argument('--split',choices=['validation','holdout'],default='holdout')
    evaluate(p.parse_args())
