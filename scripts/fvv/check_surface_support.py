"""Bounded geometric diagnostics against the explicitly supplied mesh prior.

This is a training-prior fit diagnostic, not held-out reconstruction accuracy.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import hybrid


def run(args):
    torch.set_num_threads(4)
    results={}
    with np.load(args.geometry,allow_pickle=False) as data:
        points=data['points'];normals=data['normals'];times=data['times']
    if points.shape!=normals.shape or points.shape[1:]!=(8192,3) or not 1<=len(times)<=64:
        raise ValueError('Invalid bounded surface prior')
    for name,path in [('previous',args.previous),('candidate',args.model)]:
        renderer=hybrid.Renderer(SimpleNamespace(repo=args.repo.resolve(),model=path.resolve(),temporal_sigma=1/60))
        offsets=[];thicknesses=[];distances=[]
        torch.manual_seed(6666)
        with torch.no_grad():
            for index,timestamp in enumerate(times):
                covariance,offset=renderer.model.get_current_covariance_and_mean_offset(1.,float(timestamp))
                active=(renderer.model.get_marginal_t(float(timestamp)).flatten()>.1)&(renderer.model.get_opacity.flatten()>.05)
                selected=torch.where(active)[0]
                selected=selected[torch.randperm(len(selected),device='cuda')[:4096]]
                centers=(renderer.model.get_xyz+offset)[selected]
                source=torch.tensor(points[index],device='cuda')
                normal=torch.tensor(normals[index],device='cuda')
                for start in range(0,len(centers),512):
                    center=centers[start:start+512]
                    distance,nearest=torch.cdist(center,source).min(1)
                    n=normal[nearest];x,y,z=n.unbind(-1)
                    c=covariance[selected[start:start+512]]
                    variance=c[:,0]*x*x+2*c[:,1]*x*y+2*c[:,2]*x*z+c[:,3]*y*y+2*c[:,4]*y*z+c[:,5]*z*z
                    offsets.extend(((center-source[nearest])*n).sum(-1).abs().cpu().tolist())
                    thicknesses.extend(variance.clamp_min(0).sqrt().cpu().tolist())
                    distances.extend(distance.cpu().tolist())
        def stats(values):
            return dict(mean=float(np.mean(values)),p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)))
        results[name]=dict(sha256=renderer.meta['sha256'],samples=len(offsets),normal_offset=stats(offsets),
                           normal_sigma=stats(thicknesses),sampled_surface_distance=stats(distances))
        del renderer;torch.cuda.empty_cache()
    results['scope']='Training mesh prior fit; sampled points, dynamic Gaussians only; not held-out accuracy'
    with args.output.open('x') as stream:json.dump(results,stream,indent=2)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','previous','model','geometry','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
