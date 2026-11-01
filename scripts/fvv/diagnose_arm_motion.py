"""Infer arm membership and six independent rigid tracks from supplied source."""
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
    for key in ['repo','model','parts','binding','dataset','source','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    with np.load(a.parts,allow_pickle=False) as d:names=json.loads(str(d['metadata_json']))['objects'];matrices=d['object_matrices'].astype(float)
    with np.load(a.binding,allow_pickle=False) as d:registration=np.asarray(json.loads(str(d['metadata_json']))['initial_registration'])
    registration=np.asarray(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    delta=matrices@np.linalg.inv(matrices[0]);moving=np.max(abs(delta-np.eye(4)),axis=(0,2,3))>1e-5
    remaining=set(i for i in range(168,227) if moving[i]);groups=[]
    while remaining:
        i=min(remaining);g=[j for j in sorted(remaining) if np.max(abs(delta[:,j]-delta[:,i]))<1e-4];remaining.difference_update(g);groups.append(g)
    if len(groups)!=6:raise ValueError('Expected six independent rigid arm groups')
    tracks=np.stack([registration@delta[:399,g[0]]@np.linalg.inv(registration) for g in groups],axis=1)
    if not np.allclose(tracks[:,:,:3,:3]@tracks[:,:,:3,:3].transpose(0,1,3,2),np.eye(3),atol=2e-5):raise ValueError('Nonrigid arm transform')
    with np.load(a.source,allow_pickle=False) as d:points=d['points'];normals=d['normals'];labels=d['parts'];times=d['times'].reshape(-1);static=d['static_points'];static_labels=d['static_parts']
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model));peaks=r.model.get_t.detach().cpu().numpy().reshape(-1);stamps=np.unique(times)
    if len(stamps)>64 or len(peaks)>450000:raise ValueError('Diagnostic budget exceeded')
    assigned=np.abs(peaks[:,None]-stamps).argmin(1);nearest_labels=np.full(len(peaks),-1,np.int32);nearest_distance=np.zeros(len(peaks),np.float32)
    clouds=[];normal_clouds=[];part_clouds=[];group_clouds=[];started=time.monotonic()
    with torch.no_grad():
        for k,t in enumerate(stamps):
            rows=np.flatnonzero(assigned==k);mask=times==t;surface=np.concatenate([points[mask],static]);part=np.concatenate([labels[mask],static_labels])
            _,offset=r.model.get_current_covariance_and_mean_offset(1.,float(t));xyz=(r.model.get_xyz+offset)[rows].cpu().numpy();distance,near=cKDTree(surface).query(xyz,workers=4)
            nearest_labels[rows]=part[near];nearest_distance[rows]=distance
            for gi,g in enumerate(groups):
                chosen=mask&np.isin(labels,g);inv=np.linalg.inv(tracks[round(float(t)*60),gi]);clouds.append(points[chosen]@inv[:3,:3].T+inv[:3,3]);normal_clouds.append(normals[chosen]@inv[:3,:3].T);part_clouds.append(labels[chosen]);group_clouds.append(np.full(chosen.sum(),gi,np.int16))
            if time.monotonic()-started>600:raise TimeoutError('Arm binding exceeded ten minutes')
    ids=np.flatnonzero(np.isin(nearest_labels,np.concatenate(groups)))
    if not 1000<len(ids)<len(peaks)*.8:raise ValueError('Unexpected arm retirement count')
    metadata=dict(parent_sha256=r.meta['sha256'],parent_bucket_sha256=r.meta['rigid_bucket']['sha256'],source_sha256=digest(a.source),groups=[dict(id=i,parts=[dict(id=j,name=names[j]) for j in g]) for i,g in enumerate(groups)],retired_dynamic=len(ids),nearest_distance_percentiles=np.percentile(nearest_distance[ids],[50,90,95,99]).tolist(),scope='Explicit source six rigid arm groups; nearest-surface retirement inferred, not segmentation truth. Static supports and deforming RubberBand objects excluded.')
    np.savez_compressed(a.output/'arm-source.npz',indices=ids.astype(np.int64),tracks=tracks.astype(np.float32),points=np.concatenate(clouds).astype(np.float32),normals=np.concatenate(normal_clouds).astype(np.float32),parts=np.concatenate(part_clouds),groups=np.concatenate(group_clouds),metadata_json=np.asarray(json.dumps(metadata)))
    if (a.output/'arm-source.npz').stat().st_size>64*1024**2:raise OSError('Arm source budget exceeded')
    (a.output/'diagnosis.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))


if __name__=='__main__':main()
