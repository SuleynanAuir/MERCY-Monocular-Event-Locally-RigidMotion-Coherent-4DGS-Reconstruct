"""Compare ROI temporal residuals to real RGB, including unseen timestamps."""
import argparse,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid
from refine_lego import decode


def run(a):
    torch.set_num_threads(4)
    frames=decode(a.video);results={}
    rois={'bucket':(40,40,340,430),'arm':(210,120,470,470),'body':(350,380,610,560)}
    root=a.output;root.mkdir(exist_ok=False,parents=True)
    panel=Image.new('RGB',(8*320,3*320))
    for name,model,row in [('previous',a.previous,1),('refined',a.model,2)]:
        r=hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=model.resolve(),temporal_sigma=1/60))
        from utils.loss_utils import ssim
        p=dict(position=[0,0,0],yaw=0,pitch=0,fov=r.meta['fov'],width=640,height=640,
               temporal_samples=a.temporal_samples if name=='refined' else 1)
        accum={k:dict(temporal_residual=[],interior_temporal_residual=[],second_difference_residual=[],legacy_evaluation_mse=[],odd_frame_mse=[]) for k in rois}
        quality=[];prev=None;prev2=None;previous_target=None
        for i in range(399):
            out=r.frame({**p,'time':i/60},raw=True).clamp(0,1)
            gt=torch.from_numpy(frames[i].copy()).permute(2,0,1).float()/255
            error=out-gt
            interior=None
            if previous_target is not None:
                changed=(gt-previous_target).abs().amax(0,keepdim=True)>.04
                edge=torch.nn.functional.max_pool2d(changed.float().unsqueeze(0),5,1,2)[0,0]
                interior=(edge==0)&(gt.mean(0)>8/255)&(previous_target.mean(0)>8/255)
            for k,(x0,y0,x1,y1) in rois.items():
                crop=lambda x:x[:,y0:y1,x0:x1]
                if prev is not None:accum[k]['temporal_residual'].append(float(crop(error-prev).abs().mean()))
                if interior is not None:
                    mask=interior[y0:y1,x0:x1]
                    if mask.any():accum[k]['interior_temporal_residual'].append(float(crop(error-prev)[:,mask].abs().mean()))
                if prev2 is not None:accum[k]['second_difference_residual'].append(float(crop(error-2*prev+prev2).abs().mean()))
                if i%20==10:accum[k]['legacy_evaluation_mse'].append(float(crop(error).square().mean()))
                if i%20==11:accum[k]['odd_frame_mse'].append(float(crop(error).square().mean()))
            if i%20==10:
                mask=gt.mean(0)>8/255
                dx=error[:,:,1:]-error[:,:,:-1];dy=error[:,1:,:]-error[:,:-1,:]
                quality.append(dict(psnr=float(-10*torch.log10(error.square().mean())),ssim=float(ssim(out,gt)),
                    foreground_psnr=float(-10*torch.log10(error[:,mask].square().mean())),
                    edge_mae=float(.5*(dx[:,mask[:,1:]|mask[:,:-1]].abs().mean()+dy[:,mask[1:,:]|mask[:-1,:]].abs().mean()))))
            if i in range(144,160,2):
                col=(i-144)//2
                for image,rr in [(gt,0),(out,row)]:
                    im=Image.fromarray((image.permute(1,2,0).numpy()*255).astype('uint8'))
                    panel.paste(im.crop((40,50,440,450)).resize((320,320)),(col*320,rr*320))
            prev2,prev=prev,error;previous_target=gt
        offgrid={k:[] for k in rois}
        for timestamp in np.linspace(.2,6.4,63):
            triplet=[r.frame({**p,'time':float(timestamp+d)},raw=True) for d in [-1/240,0,1/240]]
            delta=(triplet[2]-2*triplet[1]+triplet[0]).abs()
            for k,(x0,y0,x1,y1) in rois.items():offgrid[k].append(float(delta[:,y0:y1,x0:x1].mean()))
        still=r.frame({**p,'time':2.5},raw=True)
        repeat=r.frame({**p,'time':2.5},raw=True)
        shifted=r.frame({**p,'time':2.5,'position':[.1,0,0],'yaw':-5},raw=True)
        moving=r.frame({**p,'time':3.5},raw=True)
        checks=dict(same_pose_time_max_difference=float((still-repeat).abs().max()),
            changed_view_mean_difference=float((still-shifted).abs().mean()),
            changed_time_mean_difference=float((still-moving).abs().mean()),
            model_bytes=(model/'model.pth').stat().st_size,
            dynamic_points=len(r.model.get_xyz),static_points=len(r.model.get_static_xyz))
        provenance=model/'refinement-provenance.json'
        scope=json.loads(provenance.read_text()).get('evaluation_scope') if provenance.exists() else 'previous development holdouts'
        results[name]=dict(temporal_samples=p['temporal_samples'],sha256=r.meta['sha256'],evaluation_scope=scope,checks=checks,
            offgrid_second_difference={k:float(np.mean(v)) for k,v in offgrid.items()},rois={k:{metric:float(np.mean(v)) for metric,v in data.items()} for k,data in accum.items()},
            legacy_evaluation_frame_quality={k:float(np.mean([v[k] for v in quality])) for k in quality[0]})
        del r;torch.cuda.empty_cache()
    for row,label in enumerate(['Original RGB','Previous FVV','ROI temporal refinement']):ImageDraw.Draw(panel).text((4,row*320+4),label,fill='red')
    panel.save(root/'roi-comparison.jpg',quality=94)
    (root/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['repo','previous','model','video','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--temporal-samples',type=int,choices=[1,3],default=1,help='Candidate rendering samples; previous stays at one')
    run(p.parse_args())
