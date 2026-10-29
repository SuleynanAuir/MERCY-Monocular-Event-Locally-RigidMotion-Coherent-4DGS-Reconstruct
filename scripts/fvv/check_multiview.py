"""Evaluate held-out source-scene pullback views and the original RGB view."""
import argparse,json,math,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image,ImageDraw
import hybrid
from refine_lego import view_camera


def run(a):
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False)
    views=json.loads((a.views/'views.json').read_text())['views']
    validation=[v for v in views if v['split']=='validation']
    selected=[validation[i] for i in [0,1,len(validation)//2,len(validation)//2+1]]
    panel=Image.new('RGB',(4*480,3*500));results={}
    for name,model,row in [('previous',a.previous,1),('candidate',a.model,2)]:
        r=hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=model.resolve(),temporal_sigma=1/60))
        from utils.loss_utils import ssim
        rows=[]
        with torch.no_grad():
            for v in validation:
                target=torch.from_numpy(np.array(Image.open(a.views/v['file']).convert('RGB'))).permute(2,0,1).float()/255
                image=r.render_fn(view_camera(v),r.model,r.pipe,r.bg)['render'].clamp(0,1).cpu()
                error=image-target;fg=target.mean(0)>8/255
                scale=1.5/(1.5+v['distance']);lo=int(320+(40-320)*scale)-12
                hi=int(320+(430-320)*scale)+12;right=int(320+(340-320)*scale)+12
                mask=torch.zeros((640,640),dtype=torch.bool);mask[lo:hi,lo:right]=True
                mask &= target.mean(0)>.2
                dx=error[:,:,1:]-error[:,:,:-1];dy=error[:,1:,:]-error[:,:-1,:]
                rows.append(dict(time=v['time'],distance=v['distance'],psnr=float(-10*torch.log10(error.square().mean())),
                    foreground_psnr=float(-10*torch.log10(error[:,fg].square().mean())),ssim=float(ssim(image,target)),
                    bucket_mse=float(error[:,mask].square().mean()),
                    bucket_dark_pixel_fraction=float(((image.mean(0)[mask]-target.mean(0)[mask])<-.15).float().mean()),
                    edge_mae=float(.5*(dx[:,fg[:,1:]|fg[:,:-1]].abs().mean()+dy[:,fg[1:,:]|fg[:-1,:]].abs().mean()))))
                if v in selected:
                    col=selected.index(v)
                    for im,rr in [(target,0),(image,row)]:
                        rgb=Image.fromarray((im.permute(1,2,0).numpy()*255).astype('uint8')).resize((480,480))
                        panel.paste(rgb,(col*480,rr*500+20))
            results[name]=dict(sha256=r.meta['sha256'],rows=rows,
                means={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ['time','distance']})
        del r;torch.cuda.empty_cache()
    for i,label in enumerate(['Source scene reference (not captured RGB)','Previous FVV','Multiview candidate']):ImageDraw.Draw(panel).text((5,i*500+3),label,fill='white')
    panel.save(a.output/'pullback-comparison.jpg',quality=94)
    (a.output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v['means'] for k,v in results.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['repo','previous','model','views','output']:p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
