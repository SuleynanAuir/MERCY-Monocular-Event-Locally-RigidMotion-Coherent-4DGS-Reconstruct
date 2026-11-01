"""Read-only opacity floor sweep; reject apparent solidity that damages detail."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
import hybrid
from refine_lego import view_camera
from rigid_bucket import bucket_only_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['repo','model','arm-views','bucket-views','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model));original=r.bucket.values['opacity'];masked=bucket_only_model(r.model)
    refs={}
    for name,folder in [('arm',a.arm_views),('bucket',a.bucket_views)]:
        refs[name]=[(view_camera(v),torch.from_numpy(np.array(Image.open(folder/v['file']).convert('RGB'))).cuda().permute(2,0,1).float()/255) for v in json.loads((folder/'views.json').read_text())['views'] if v['split']=='validation']
    result={}
    with torch.no_grad():
        for floor in [0.,.2,.4,.6]:
            r.bucket.values['opacity']=original if not floor else torch.logit(original.sigmoid().clamp_min(floor))
            row={}
            for name,component in [('arm',r.bucket.arm_only()),('bucket',r.bucket.head_only())]:
                render=component.wrap(r.base_render_fn);metrics=[]
                for cam,target in refs[name]:
                    image=render(cam,masked,r.pipe,r.bg)['render'].clamp(0,1);fg=target.mean(0)>8/255
                    metrics.append(dict(psnr=float(-10*torch.log10((image[:,fg]-target[:,fg]).square().mean())),background_mae=float(image[:,~fg].abs().mean())))
                row[name]=dict(psnr=float(np.mean([x['psnr'] for x in metrics])),worst_psnr=min(x['psnr'] for x in metrics),background_mae=float(np.mean([x['background_mae'] for x in metrics])))
            result[str(floor)]=row;print(json.dumps(dict(floor=floor,metrics=row)),flush=True)
    result['scope']='Read-only complete isolated component references; no weights written; solidity alone is not acceptance.'
    a.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
