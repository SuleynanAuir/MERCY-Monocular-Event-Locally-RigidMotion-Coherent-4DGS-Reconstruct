"""Prepare actual Lego RGB observations and G0 prior for isolated Hybrid training.

No generated images or fake camera views. A rigid change of world coordinates
places the measured fixed camera at the origin, looking along +Z.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from common import digest, usage


def prepare(args):
    reference=args.reference.resolve(); source=args.source.resolve(); root=args.output.resolve()
    meta=json.loads(reference.read_text())
    camera=meta['camera']; pose=np.asarray(camera['pose_opengl_c2w'],dtype=np.float64)
    if pose.shape != (4,4) or not np.allclose(pose[3],[0,0,0,1]):raise ValueError('Invalid camera')
    cv_pose=pose.copy();cv_pose[:3,1:3]*=-1
    world_to_cv=np.linalg.inv(cv_pose)
    K=np.asarray(camera['K']);w,h=camera['width'],camera['height']
    if not np.allclose(K[:2,2],[w/2,h/2]):raise ValueError('Off-center principal point needs explicit camera support')
    if not math.isclose(float(K[0,0]),float(K[1,1]),rel_tol=1e-5):raise ValueError('Non-square pixel camera')
    frames=meta['samples']
    if len(frames)<30:raise ValueError('Need at least 30 real observations')
    root.mkdir(parents=True,exist_ok=False);(root/'images').mkdir()
    partitions={'train':[], 'val':[], 'test':[]}; evidence=[]
    for i,item in enumerate(frames):
        image=(reference.parent/item['rgb']).resolve()
        if not image.is_file():raise FileNotFoundError(image)
        name=f"{int(item['frame']):06d}"
        (root/'images'/f'{name}.png').symlink_to(image)
        split='val' if i%10==0 else 'test' if i%10==5 else 'train'
        entry=dict(file_path='images/'+name,time=float(item['time_seconds']),
                   transform_matrix=np.diag([1.,-1.,-1.,1.]).tolist())
        partitions[split].append(entry)
        evidence.append(dict(frame=item['frame'],time=item['time_seconds'],split=split,sha256=digest(image)))
    for split,entries in partitions.items():
        # Upstream calls its validation split 'test'. Keep the untouched test
        # observations under a separate manifest, never used for selection.
        target={'train':'transforms_train.json','val':'transforms_test.json','test':'transforms_holdout.json'}[split]
        (root/target).write_text(json.dumps(dict(camera_angle_x=2*math.atan(w/(2*K[0,0])),frames=entries),indent=2))
    data=PlyData.read(source,mmap=True)['vertex'].data
    names=data.dtype.names
    xyz=np.column_stack([data[k] for k in ('x','y','z')])
    valid=np.isfinite(xyz).all(1)
    if 'opacity' in names:valid &= data['opacity'] > -4.6
    ids=np.flatnonzero(valid)
    rng=np.random.default_rng(6666)
    ids=rng.choice(ids,size=min(args.points,len(ids)),replace=False)
    xyz=xyz[ids]@world_to_cv[:3,:3].T+world_to_cv[:3,3]
    if not np.isfinite(xyz).all():raise ValueError('Invalid Gaussian centers')
    rgb=np.clip(.5+.28209479177387814*np.column_stack([data[f'f_dc_{i}'][ids] for i in range(3)]),0,1)
    vertices=np.empty(len(ids),dtype=[(k,'f4') for k in ('x','y','z','nx','ny','nz')]+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate(('x','y','z')):vertices[k]=xyz[:,i]
    for k in ('nx','ny','nz'):vertices[k]=0
    for i,k in enumerate(('red','green','blue')):vertices[k]=np.rint(rgb[:,i]*255).astype('u1')
    PlyData([PlyElement.describe(vertices,'vertex')]).write(root/'points3d.ply')
    extent=float(np.linalg.norm(np.quantile(xyz,.99,axis=0)-np.quantile(xyz,.01,axis=0))/2)
    provenance=dict(asset_kind='actual Lego dataset observations; not synthetic smoke',
        supervision='supplied original RGB frames, single fixed camera; no generated view supervision',
        reference_path=str(reference),reference_sha256=digest(reference),source_record=meta['sources']['video'],
        initialization=str(source),initialization_sha256=digest(source),initialization_points=len(ids),
        initialization_scope='existing G0 geometric/color prior; not reconstruction from scratch',
        world_to_camera_cv=world_to_cv.tolist(),single_view_extent=extent,
        time_duration=[min(x['time'] for x in evidence),max(x['time'] for x in evidence)],
        fov_degrees=math.degrees(2*math.atan(h/(2*K[1,1]))),width=w,height=h,
        split_counts={k:len(v) for k,v in partitions.items()},frames=evidence,
        split_scope='interleaved temporal holdout within supplied sequence, not novel-view generalization; legacy G0 calibration exposure unknown')
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if usage(root)>1024**3:raise OSError('Prepared dataset exceeds 1 GiB budget')
    print(json.dumps({k:v for k,v in provenance.items() if k!='frames'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--points',type=int,default=150000)
    args=p.parse_args()
    if not 1000<=args.points<=500000:p.error('points must be 1000..500000')
    prepare(args)
