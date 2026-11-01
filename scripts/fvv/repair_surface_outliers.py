"""Create an isolated candidate by removing distant animated surface outliers.

Requires the explicit source-mesh prior in Hybrid CV coordinates. This is a
candidate initializer, never an automatic replacement for the live model.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import hybrid
from common import digest, usage


def run(a):
    torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    if usage(a.output.parent)>8*1024**3:raise OSError('Candidate storage budget exceeded')
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo.resolve(),model=a.model.resolve()))
    state,iteration=hybrid.load_checkpoint(a.model/'model.pth','cuda')
    with np.load(a.geometry,allow_pickle=False) as data:
        times=data['times'];points=data['points']
    if points.shape!=(len(times),8192,3) or not 1<=len(times)<=64 or not np.isfinite(points).all() or not np.isfinite(times).all():
        raise ValueError('Invalid bounded surface prior')
    nearest=np.abs(state[13].detach().flatten().cpu().numpy()[:,None]-times).argmin(1)
    distance=torch.zeros(len(state[1]),device='cuda')
    with torch.no_grad():
        for k,stamp in enumerate(times):
            ids=torch.tensor(np.flatnonzero(nearest==k),device='cuda')
            if not len(ids):continue
            # API takes absolute timestamp and subtracts each Gaussian's t
            # internally. Subtracting t here again corrupts motion alignment.
            _,offset=r.model.get_current_covariance_and_mean_offset(1.,float(stamp))
            centers=(r.model.get_xyz+offset)[ids]
            surface=torch.tensor(points[k],device='cuda')
            distance[ids]=torch.cat([torch.cdist(chunk[None],surface[None])[0].min(1).values for chunk in centers.split(512)])
        keep=distance<=a.distance
        removed=int((~keep).sum())
        if removed>len(keep)*.03:raise ValueError('More than 3% outliers: check alignment instead of deleting surfaces')
        candidate=list(state)
        for field in [1,2,3,4,5,6,13,14,15]:candidate[field]=candidate[field][keep]
        a.output.mkdir()
        hybrid.save_bounded(hybrid.compact((tuple(candidate),iteration)),a.output/'model.pth',512*1024**2)
    (a.output/'effective.yaml').write_bytes((a.model/'effective.yaml').read_bytes())
    evidence=dict(parent_sha256=r.meta['sha256'],surface_prior_sha256=digest(a.geometry),distance=a.distance,
                  removed=removed,remaining=int(keep.sum()),scope='explicit source mesh prior; nearest timestamp to temporal peak; candidate requires image and motion validation')
    (a.output/'surface-repair.json').write_text(json.dumps(evidence,indent=2))
    meta=dict(r.meta,sha256=digest(a.output/'model.pth'),surface_repair=evidence,bytes=usage(a.output))
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','model','geometry','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--distance',type=float,default=.08,choices=[.08,.1,.12])
    a=p.parse_args();lock=a.output.parent/'.fvv-writer.lock'
    with lock.open('x') as stream:stream.write('isolated surface outlier candidate\n')
    try:run(a)
    finally:lock.unlink(missing_ok=True)
