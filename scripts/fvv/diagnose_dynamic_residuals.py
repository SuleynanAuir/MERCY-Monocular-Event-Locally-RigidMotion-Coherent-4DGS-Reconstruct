"""Read-only multi-time source ownership audit of residual 4D Gaussians."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
import hybrid
from common import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['repo','model','source','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model))
    with np.load(a.source,allow_pickle=False) as d:
        pts=d['points'];times=d['times'].reshape(-1);labels=d['parts'];static=d['static_points'];sl=d['static_parts']
    stamps=np.unique(times);n=len(r.model.get_xyz)
    if len(stamps)>64 or n>450000:raise ValueError('Diagnostic bound exceeded')
    protected=np.array(list(range(144,190))+list(range(203,217)))
    counts={k:np.zeros(n,np.int32) for k in ['active','protected','other','far']}
    distance_sum=np.zeros(n,np.float64);start=time.monotonic()
    with torch.no_grad():
        opacity=r.model.get_opacity.flatten()
        for k,t in enumerate(stamps):
            mask=times==t;surface=np.concatenate([pts[mask],static]);parts=np.concatenate([labels[mask],sl])
            active=(r.model.get_marginal_t(float(t)).flatten()*opacity>.03).cpu().numpy();ids=np.flatnonzero(active)
            _,offset=r.model.get_current_covariance_and_mean_offset(1.,float(t));xyz=(r.model.get_xyz+offset)[ids].cpu().numpy()
            dist,near=cKDTree(surface).query(xyz,workers=4);part=parts[near]
            counts['active'][ids]+=1;distance_sum[ids]+=dist
            own=np.isin(part,protected)
            counts['protected'][ids[(dist<.015)&own]]+=1
            counts['other'][ids[(dist<.015)&~own]]+=1
            counts['far'][ids[dist>.03]]+=1
            if k%8==0:print(json.dumps(dict(sample=k,active=len(ids),elapsed=time.monotonic()-start)),flush=True)
            if time.monotonic()-start>600:raise TimeoutError('Diagnostic timeout')
    den=np.maximum(counts['active'],1)
    candidates=(counts['active']>=3)&(counts['protected']/den>=.8)&(counts['other']==0)
    far=(counts['active']>=3)&(counts['far']/den>=.8)
    np.savez_compressed(a.output/'ownership.npz',**counts,mean_distance=(distance_sum/den).astype(np.float32),duplicate_candidates=np.flatnonzero(candidates),far_candidates=np.flatnonzero(far))
    temporal=r.bucket.values['temporal_dc'].cpu().numpy();groups=r.bucket.values['point_groups'].cpu().numpy()
    stats={}
    for g in range(7):
        x=temporal[:,groups==g]
        stats[str(g)]=dict(points=x.shape[1],temporal_std_percentiles=np.percentile(x.std(0),[50,90,99]).tolist(),second_difference_mean=float(np.abs(x[2:]-2*x[1:-1]+x[:-2]).mean()))
    result=dict(core_sha256=r.meta['sha256'],sidecar_sha256=r.meta['rigid_bucket']['sha256'],source_sha256=digest(a.source),samples=len(stamps),core_points=n,duplicate_candidates=int(candidates.sum()),far_candidates=int(far.sum()),temporal_color=stats,scope='Source-assisted nearest surface votes over active times; hypotheses, not ground truth segmentation. No model changes.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
