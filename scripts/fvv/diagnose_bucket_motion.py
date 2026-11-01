"""Bind only the verified rigid bucket group for an isolated FVV candidate.

Uses the supplied animated source as an explicit prior, never changes a model.
Nearest-surface membership is inferred and is recorded for review.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree
import hybrid
from common import digest


def transform(points,matrix):
    return points@matrix[:3,:3].T+matrix[:3,3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['repo','model','parts','binding','dataset','source','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    if a.source.stat().st_size>64*1024**2:raise ValueError('Source exceeds 64 MiB budget')
    with np.load(a.parts,allow_pickle=False) as data:
        names=json.loads(str(data['metadata_json']))['objects'];matrices=data['object_matrices'].astype(np.float64)
    delta=matrices@np.linalg.inv(matrices[0]);tip=names.index('TipperEnd.001')
    group=np.flatnonzero(np.max(np.abs(delta-delta[:,tip:tip+1]),axis=(0,2,3))<1e-4)
    if not 2<=len(group)<=32:raise ValueError('Unexpected rigid bucket group')
    with np.load(a.binding,allow_pickle=False) as data:
        registration=np.asarray(json.loads(str(data['metadata_json']))['initial_registration'])
    registration=np.asarray(json.loads((a.dataset/'provenance.json').read_text())['world_to_camera_cv'])@registration
    tracks=registration@delta[:399,tip]@np.linalg.inv(registration)
    if not np.allclose(tracks[:,:3,:3]@tracks[:,:3,:3].transpose(0,2,1),np.eye(3),atol=2e-5):raise ValueError('Bucket transforms are not rigid')
    with np.load(a.source,allow_pickle=False) as data:
        points=data['points'];normals=data['normals'];labels=data['parts'];times=data['times'].reshape(-1)
        static_points=data['static_points'];static_labels=data['static_parts']
    r=hybrid.Renderer(SimpleNamespace(repo=a.repo,model=a.model))
    peaks=r.model.get_t.detach().cpu().numpy().reshape(-1);stamps=np.unique(times)
    if not 1<=len(stamps)<=64 or not 1<=len(peaks)<=450000:raise ValueError('Diagnostic point/time budget exceeded')
    assigned=np.abs(peaks[:,None]-stamps[None]).argmin(1)
    nearest_labels=np.full(len(peaks),-1,np.int32);nearest_distance=np.zeros(len(peaks),np.float32)
    canonical=[];canonical_normals=[];canonical_labels=[]
    start=time.monotonic()
    with torch.no_grad():
        for k,stamp in enumerate(stamps):
            if time.monotonic()-start>600:raise TimeoutError('Bucket binding exceeded ten minutes')
            rows=np.flatnonzero(assigned==k);mask=times==stamp
            surface=np.concatenate([points[mask],static_points]);part=np.concatenate([labels[mask],static_labels])
            _,offset=r.model.get_current_covariance_and_mean_offset(1.,float(stamp))
            centers=(r.model.get_xyz+offset)[rows].cpu().numpy()
            distances,near=cKDTree(surface).query(centers,workers=4)
            nearest_labels[rows]=part[near];nearest_distance[rows]=distances
            bucket=mask&np.isin(labels,group);inv=np.linalg.inv(tracks[round(float(stamp)*60)])
            canonical.append(transform(points[bucket],inv));canonical_normals.append(normals[bucket]@inv[:3,:3].T);canonical_labels.append(labels[bucket])
    selected=np.flatnonzero(np.isin(nearest_labels,group))
    if not 1000<len(selected)<len(peaks)*.8:raise ValueError('Unexpected bucket membership')
    xyz=r.model.get_xyz.detach().cpu().numpy()[selected];peak_frames=np.clip(np.rint(peaks[selected]*60).astype(int),0,398)
    inverses=np.linalg.inv(tracks[peak_frames]);rest=np.einsum('nij,nj->ni',inverses[:,:3,:3],xyz)+inverses[:,:3,3]
    cloud=np.concatenate(canonical);normal=np.concatenate(canonical_normals);part=np.concatenate(canonical_labels)
    # Same-part nearest surface; do not attach one side to another part.
    snapped=np.empty_like(rest);snap_normals=np.empty_like(rest)
    for label in np.unique(nearest_labels[selected]):
        rows=np.flatnonzero(nearest_labels[selected]==label);source_rows=np.flatnonzero(part==label)
        _,near=cKDTree(cloud[source_rows]).query(rest[rows],workers=4)
        snapped[rows]=cloud[source_rows[near]];snap_normals[rows]=normal[source_rows[near]]
    motion=[]
    with torch.no_grad():
        for frame in range(0,399,10):
            _,offset=r.model.get_current_covariance_and_mean_offset(1.,frame/60)
            active=(r.model.get_marginal_t(frame/60).flatten()[selected]>.1)&(r.model.get_opacity.flatten()[selected]>.05)
            actual=(r.model.get_xyz+offset)[selected].cpu().numpy();keep=active.cpu().numpy()
            error=np.linalg.norm(actual-transform(rest,tracks[frame]),axis=1)[keep]
            motion.append(dict(frame=frame,active=int(keep.sum()),p50=float(np.median(error)),p95=float(np.percentile(error,95))))
    metadata=dict(parent_sha256=r.meta['sha256'],source_sha256=digest(a.source),parts_sha256=digest(a.parts),
        bucket_parts=[dict(id=int(i),name=names[i]) for i in group],removed_dynamic=len(selected),total_dynamic=len(peaks),
        nearest_distance_percentiles=np.percentile(nearest_distance[selected],[50,90,95,99]).tolist(),
        snap_distance_percentiles=np.percentile(np.linalg.norm(rest-snapped,axis=1),[50,90,95,99]).tolist(),
        motion_deviation=motion,elapsed_seconds=time.monotonic()-start,
        peak_pose_quantization_seconds=1/60,
        scope='Explicit source rigid prior; nearest-surface membership is inferred, not ground-truth segmentation. Peak pose uses the nearest source frame, so deviation includes frame quantization error.')
    temporary=a.output/'bucket-source.partial.npz'
    try:
        np.savez_compressed(temporary,indices=selected.astype(np.int64),tracks=tracks.astype(np.float32),
            points=cloud.astype(np.float32),normals=normal.astype(np.float32),parts=part,
            metadata_json=np.asarray(json.dumps(metadata)))
        if temporary.stat().st_size>64*1024**2:raise OSError('Bucket diagnostic exceeds 64 MiB')
        temporary.replace(a.output/'bucket-source.npz')
    finally:temporary.unlink(missing_ok=True)
    (a.output/'diagnosis.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
