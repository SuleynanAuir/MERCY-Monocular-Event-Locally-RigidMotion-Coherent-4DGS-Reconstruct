"""Read-only footprint sweep for a completed fixed-source body candidate."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
import hybrid
from refine_lego import view_camera


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','views','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model))
    if r.meta.get('rigid_bucket',{}).get('schema')!=3:raise ValueError('Requires a completed fixed body candidate')
    body=r.bucket.values['point_groups']==7;original=r.bucket.values['scales'].clone();views=[]
    for v in json.loads((a.views/'views.json').read_text())['views']:
        if v['split']!='train':views.append((v,view_camera(v),torch.from_numpy(np.array(Image.open(a.views/v['file']).convert('RGB')))))
    result=dict(core_sha256=r.meta['sha256'],sidecar_sha256=r.meta['rigid_bucket']['sha256'],opacity_quantiles=torch.quantile(r.bucket.values['opacity'][body].sigmoid(),torch.tensor([0.,.1,.5,.9,1.],device='cuda')).cpu().tolist(),candidates=[],scope='Source validation views used for footprint calibration; not independent captured views.')
    with torch.no_grad():
        for scale in [.0035,.0045,.0055,.007]:
            r.bucket.values['scales'][body,:2]=scale;rows=[];panel=Image.new('RGB',(1280,640))
            for index,(v,cam,pixels) in enumerate(views):
                gt=pixels.cuda().permute(2,0,1).float()/255;pred=r.render_fn(cam,r.model,r.pipe,r.bg)['render'].clamp(0,1);mask=gt.mean(0)>8/255;error=pred-gt
                rows.append(dict(file=v['file'],foreground_psnr=float(-10*torch.log10(error[:,mask].square().mean().clamp_min(1e-12))),background_mae=float(error[:,~mask].abs().mean())))
                if index<8:panel.paste(Image.fromarray((pred.permute(1,2,0)*255).byte().cpu().numpy()).resize((320,320)),((index%4)*320,(index//4)*320))
            result['candidates'].append(dict(scale=scale,foreground_psnr=float(np.mean([x['foreground_psnr'] for x in rows])),worst_psnr=min(x['foreground_psnr'] for x in rows),background_mae=float(np.mean([x['background_mae'] for x in rows])),rows=rows))
            panel.save(a.output/(str(scale)+'.jpg'),quality=94)
        r.bucket.values['scales'].copy_(original)
    (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({**result,'candidates':[{k:v for k,v in x.items() if k!='rows'} for x in result['candidates']]},indent=2))


if __name__=='__main__':main()
