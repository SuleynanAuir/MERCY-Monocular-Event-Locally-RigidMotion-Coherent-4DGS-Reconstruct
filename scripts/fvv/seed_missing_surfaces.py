"""Bounded source-mesh moving-surface gap seeding, always into a new candidate.

Existing rows are preserved exactly. New low-opacity support follows source
surface normals and measured source-animation velocities; it is not a trained
result and must pass RGB, motion and orbit validation after fitting.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import hybrid
from common import digest,usage
from seed_covariance import factor_surface_motion


def run(a):
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.seeds,allow_pickle=False) as data:
        points=data['points'];normals=data['normals'];velocity=data['velocities'];times=data['times'];parts=data['parts']
    if not 1<=len(points)<=65536 or points.shape!=(len(points),3) or normals.shape!=points.shape or velocity.shape!=points.shape or times.shape!=(len(points),1) or parts.shape!=(len(points),):
        raise ValueError('Invalid bounded seed source')
    if any(not np.isfinite(v).all() for v in [points,normals,velocity,times]):raise ValueError('Nonfinite seeds')
    renderer=hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=a.model.resolve()))
    if times.min()<renderer.meta['time_duration'][0] or times.max()>renderer.meta['time_duration'][1]+1e-5:raise ValueError('Seed time outside model')
    distance=np.empty(len(points),np.float32)
    with torch.no_grad():
        for stamp in np.unique(times):
            indices=np.where(times[:,0]==stamp)[0]
            _,offset=renderer.model.get_current_covariance_and_mean_offset(1.,float(stamp))
            active=(renderer.model.get_marginal_t(float(stamp)).flatten()>.1)&(renderer.model.get_opacity.flatten()>.05)
            means=torch.cat(((renderer.model.get_xyz+offset)[active],renderer.model.get_static_xyz),dim=0)
            if not len(means):raise ValueError('No reference support at a seed timestamp')
            for start in range(0,len(indices),256):
                rows=indices[start:start+256];source=torch.tensor(points[rows],device='cuda')
                distance[rows]=torch.cdist(source,means).min(1).values.cpu().numpy()
    selected=(distance>a.minimum_gap)&(np.linalg.norm(velocity,axis=1)<5)
    report=dict(parent_sha256=renderer.meta['sha256'],source_sha256=digest(a.seeds),source_points=len(points),
                minimum_gap=a.minimum_gap,selected=int(selected.sum()),scope='Explicit animated source-mesh missing support, not a trained model',
                parts={str(int(part)):dict(total=int((parts==part).sum()),selected=int((selected&(parts==part)).sum())) for part in np.unique(parts)},
                conditional_tangent_sigma=.006,conditional_normal_sigma=.0012,temporal_sigma=.05,initial_opacity=.1,initial_color=[.55,.55,.55])
    (a.output/'seed-provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='parts'},indent=2))
    if a.audit_only:return
    if not selected.any():raise ValueError('No uncovered source support; do not create a duplicate model')
    state,iteration=hybrid.load_checkpoint(a.model/'model.pth','cpu');state=list(state);count=int(selected.sum())
    if len(state[1])+count>450000:raise ValueError('Total Gaussian budget exceeded')
    scales,left,right=factor_surface_motion(velocity[selected],normals[selected])
    new={1:points[selected],2:np.full((count,*state[2].shape[1:]),(.55-.5)/.28209479177387814,np.float32),
         3:np.zeros((count,*state[3].shape[1:]),np.float32),4:scales[:,:3],5:left,6:np.full((count,1),np.log(.1/.9),np.float32),
         13:times[selected],14:scales[:,3:],15:right}
    for index,value in new.items():
        original=state[index];addition=torch.from_numpy(np.ascontiguousarray(value)).to(dtype=original.dtype)
        state[index]=torch.cat((original,addition),dim=0)
        if not torch.equal(state[index][:len(original)],original):raise AssertionError('Original Gaussian rows changed')
    hybrid.save_bounded(hybrid.compact((tuple(state),iteration)),a.output/'model.pth',512*1024**2)
    (a.output/'effective.yaml').write_bytes((a.model/'effective.yaml').read_bytes())
    meta=dict(renderer.meta);meta.update(sha256=digest(a.output/'model.pth'),bytes=usage(a.output),
        seed_initialization=dict(parent_sha256=renderer.meta['sha256'],added=count,source_sha256=report['source_sha256'],requires_training=True))
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','model','seeds','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum-gap',type=float,default=.008,choices=[.006,.008,.01,.012])
    p.add_argument('--audit-only',action='store_true')
    run(p.parse_args())
