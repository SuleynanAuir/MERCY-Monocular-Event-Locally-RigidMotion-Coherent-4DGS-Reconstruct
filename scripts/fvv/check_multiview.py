"""Evaluate source-scene RGB references and record each model's training scope."""
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
    if not validation:raise ValueError('No validation views')
    angles=sorted(set(v.get('angle',0) for v in validation))
    selected=[min((v for v in validation if v.get('angle',0)==angle),key=lambda v:abs(v.get('elevation',0))) for angle in angles]
    if len(selected)==1:selected=[validation[i] for i in sorted(set([0,len(validation)//3,2*len(validation)//3,len(validation)-1]))]
    selected=selected[:8];tile=320 if len(selected)>4 else 480
    panel=Image.new('RGB',(len(selected)*tile,3*(tile+24)));results={}
    for name,model,row in [('previous',a.previous,1),('candidate',a.model,2)]:
        r=hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=model.resolve(),temporal_sigma=1/60))
        provenance=model/'refinement-provenance.json'
        production_geometry=bool(r.meta.get('source_surface_initialization') or r.meta.get('rigid_bucket')) or (provenance.is_file() and 'production all-time' in str(json.loads(provenance.read_text()).get('ray_scope','')))
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
                rows.append(dict(time=v['time'],distance=v['distance'],angle=v.get('angle',0),elevation=v.get('elevation',0),psnr=float(-10*torch.log10(error.square().mean())),
                    foreground_psnr=float(-10*torch.log10(error[:,fg].square().mean())),ssim=float(ssim(image,target)),
                    bucket_mse=float(error[:,mask].square().mean()),
                    bucket_dark_pixel_fraction=float(((image.mean(0)[mask]-target.mean(0)[mask])<-.15).float().mean()),
                    edge_mae=float(.5*(dx[:,fg[:,1:]|fg[:,:-1]].abs().mean()+dy[:,fg[1:,:]|fg[:-1,:]].abs().mean()))))
                rows[-1]['background_mae']=float(image[:,~fg].abs().mean())
                if v in selected:
                    col=selected.index(v)
                    for im,rr in [(target,0),(image,row)]:
                        rgb=Image.fromarray((im.permute(1,2,0).numpy()*255).astype('uint8')).resize((tile,tile))
                        panel.paste(rgb,(col*tile,rr*(tile+24)+24))
            results[name]=dict(sha256=r.meta['sha256'],sidecar_sha256=r.meta.get('rigid_bucket',{}).get('sha256'),rows=rows,
                scope=('Supplied Blender source RGB references; candidate uses production all-time geometry, so these are not held-out geometry timestamps. Not captured novel camera views. Bucket mask is approximate for orbit angles.' if production_geometry else 'Supplied Blender source scene; held-out source RGB timestamps, not captured novel camera views. Bucket mask is approximate for orbit angles.'),
                means={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ['time','distance','angle','elevation']},
                by_angle={str(angle):{key:float(np.mean([x[key] for x in rows if x['angle']==angle])) for key in ['psnr','foreground_psnr','ssim','edge_mae','background_mae']} for angle in angles})
        del r;torch.cuda.empty_cache()
    for i,label in enumerate(['Source scene reference (not captured RGB)','Previous FVV','Multiview candidate']):ImageDraw.Draw(panel).text((5,i*(tile+24)+3),label,fill='white')
    panel.save(a.output/'pullback-comparison.jpg',quality=94)
    (a.output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v['means'] for k,v in results.items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['repo','previous','model','views','output']:p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
