"""Compare temporal culling on real Lego; fixed-camera ROI and image fidelity."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
import torch
import hybrid
import temporal_kernel


def run(args):
    torch.set_num_threads(4)
    r=hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=args.model.resolve()))
    p=dict(position=[0,0,0],yaw=0,pitch=0,fov=r.meta['fov'],width=640,height=640)
    rois={'bucket':(60,60,340,280),'arm':(210,180,430,400),'body':(350,300,600,440)}
    times=np.linspace(.2,6.4,63)
    dataset=args.dataset.resolve();frames=json.loads((dataset/'transforms_holdout.json').read_text())['frames']
    refs=[]
    for item in frames:
        with Image.open(dataset/(item['file_path']+'.png')) as f:refs.append(torch.from_numpy(np.array(f.convert('RGB'))).permute(2,0,1).float()/255)
    from utils.loss_utils import ssim
    results={}
    for name,enabled,sigma in [('baseline',False,.05),('alpha_bound',True,0.),('prefilter_240',True,1/240),('prefilter_120',True,1/120),('prefilter_60',True,1/60)]:
        temporal_kernel.select(args.repo,enabled)
        r.pipe.opa_threshold=sigma
        jerks={k:[] for k in rois};speeds=[]
        for t in times:
            triplet=[r.frame({**p,'time':float(t+d)},raw=True) for d in [-1/240,0,1/240]]
            delta=(triplet[2]-2*triplet[1]+triplet[0]).abs()
            for k,(x0,y0,x1,y1) in rois.items():jerks[k].append(float(delta[:,y0:y1,x0:x1].mean()))
        quality=[]
        for entry,target in zip(frames,refs):
            out=r.frame({**p,'time':entry['time']},raw=True).clamp(0,1)
            mask=target.mean(0)>8/255
            error=out-target
            dx=error[:,:,1:]-error[:,:,:-1];dy=error[:,1:,:]-error[:,:-1,:]
            edge=.5*(dx[:,mask[:,1:]|mask[:,:-1]].abs().mean()+dy[:,mask[1:,:]|mask[:-1,:]].abs().mean())
            quality.append(dict(ssim=float(ssim(out,target)),edge_mae=float(edge),psnr=float(-10*torch.log10((out-target).square().mean())),
                foreground_psnr=float(-10*torch.log10((out[:,mask]-target[:,mask]).square().mean()))))
        for i in range(125):
            started=time.perf_counter();r.frame({**p,'time':float(times[i%len(times)])});ms=(time.perf_counter()-started)*1000
            if i>=5:speeds.append(ms)
        results[name]=dict(temporal_jerk={k:dict(mean=float(np.mean(v)),p95=float(np.percentile(v,95)),values=v) for k,v in jerks.items()},
            quality={k:float(np.mean([row[k] for row in quality])) for k in quality[0]},
            render_jpeg_mean_ms=float(np.mean(speeds)))
    args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps({k:{**v,'temporal_jerk':{region:values['mean'] for region,values in v['temporal_jerk'].items()}} for k,v in results.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['repo','model','dataset','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
