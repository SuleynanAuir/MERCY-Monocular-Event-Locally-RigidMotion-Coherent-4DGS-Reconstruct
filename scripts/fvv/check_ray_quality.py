"""Per-piece source depth, coverage and dark-error diagnostics for Lego FVV."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import hybrid
from refine_lego import view_camera
from ray_supervision import with_inverse_depth


def run(a):
    torch.set_num_threads(4)
    manifest=json.loads((a.rays/'views.json').read_text())
    views=[v for v in manifest['views'] if v['split']=='validation']
    if not views:raise ValueError('No geometry comparison views')
    originals=json.loads((a.rgb_views/'views.json').read_text())['views']
    report={'scope':'Explicit source-scene geometry diagnostics; candidate training provenance determines timestamp independence; not captured depth or novel-camera measurements','models':{}}
    for tag,path in [('previous',a.previous),('candidate',a.model)]:
        renderer=hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=path.resolve(),temporal_sigma=1/60))
        render=with_inverse_depth(renderer.base_render_fn)
        if renderer.bucket is not None:render=renderer.bucket.wrap(render)
        parts_report={};rows=[]
        with torch.no_grad():
            for view in views:
                camera=view_camera(view);package=render(camera,renderer.model,renderer.pipe,renderer.bg)
                with np.load(a.rays/view['file'],allow_pickle=False) as data:
                    target=torch.tensor(data['inverse_depth'],device='cuda');parts=torch.tensor(data['parts'].astype(np.int64),device='cuda');moving=torch.tensor(data['moving'],device='cuda')
                shape=target.shape
                inv=F.adaptive_avg_pool2d(package['inverse_depth'][None],shape)[0,0]
                alpha=F.adaptive_avg_pool2d(package['alpha'][None],shape)[0,0]
                image=F.adaptive_avg_pool2d(package['render'][None],shape)[0].clamp(0,1)
                ref=next(v for v in originals if all(v[k]==view[k] for k in ['time','angle','elevation','distance']))
                rgb=torch.tensor(np.array(Image.open(a.rgb_views/ref['file']).convert('RGB')),device='cuda').permute(2,0,1).float()/255
                rgb=F.adaptive_avg_pool2d(rgb[None],shape)[0]
                fg=parts>=0;interior=(F.max_pool2d((~fg).float()[None,None],3,1,1)[0,0]==0)&fg
                error=(inv-target).abs();front=inv-target>.03;holes=alpha<.9;dark=image.mean(0)<rgb.mean(0)-.15
                rgb_error=image-rgb;rgb_mse=rgb_error.square().mean(0)
                # Both sides of a part boundary contribute to its edge audit.
                dx=F.pad((rgb_error[:,:,1:]-rgb_error[:,:,:-1]).abs().mean(0),(0,1))
                dy=F.pad((rgb_error[:,1:,:]-rgb_error[:,:-1,:]).abs().mean(0),(0,0,0,1))
                edge_residual=(dx+dy)*.5
                def summary(mask):
                    return {key:float(value[mask].float().mean()) if mask.any() else None for key,value in [('inverse_depth_mae',error),('front_occluder_fraction',front),('low_coverage_fraction',holes),('dark_fraction',dark),('rgb_mse',rgb_mse),('edge_residual',edge_residual)]}
                rows.append(dict(time=view['time'],angle=view['angle'],all=summary(interior),moving=summary(interior&moving),static=summary(interior&~moving)))
                for label in torch.unique(parts[interior]).tolist():
                    mask=interior&(parts==label);count=int(mask.sum())
                    if count<4:continue
                    entry=parts_report.setdefault(str(label),dict(name=manifest['objects'][label],moving=label in manifest['moving_parts'],pixels=0,views=0,sums={key:0. for key in ['inverse_depth_mae','front_occluder_fraction','low_coverage_fraction','dark_fraction','rgb_mse','edge_residual']}))
                    entry['pixels']+=count;entry['views']+=1
                    for key,value in summary(mask).items():entry['sums'][key]+=value*count
        for entry in parts_report.values():entry['means']={k:v/entry['pixels'] for k,v in entry.pop('sums').items()}
        means={group:{key:float(np.mean([row[group][key] for row in rows if row[group][key] is not None])) for key in rows[0]['all']} for group in ['all','moving','static']}
        report['models'][tag]=dict(sha256=renderer.meta['sha256'],means=means,rows=rows,visible_parts=parts_report)
        del renderer;torch.cuda.empty_cache()
    with a.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({tag:value['means'] for tag,value in report['models'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','previous','model','rays','rgb-views','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
