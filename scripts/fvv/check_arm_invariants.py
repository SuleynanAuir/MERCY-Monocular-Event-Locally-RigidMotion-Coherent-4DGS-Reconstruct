"""Verify saved arm articulation, unchanged bucket, and retained base parameters."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import hybrid
from rigid_bucket import read_arrays,pose_at,quaternion_rotate
from common import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','candidate','source','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    old,_=hybrid.load_checkpoint(a.previous/'model.pth','cpu');new,_=hybrid.load_checkpoint(a.candidate/'model.pth','cpu')
    with np.load(a.source,allow_pickle=False) as d:ids=d['indices']
    keep=torch.ones(len(old[1]),dtype=torch.bool);keep[torch.from_numpy(ids)]=False
    checks={str(i):torch.equal(old[i][keep],new[i]) for i in [1,2,3,4,5,6,13,14,15]}
    checks.update({str(i):torch.equal(old[i],new[i]) for i in [19,20,21,22,23,24]})
    if not all(checks.values()):raise AssertionError('Retained base changed')
    before=read_arrays(a.previous/'rigid-bucket.npz');after=read_arrays(a.candidate/'rigid-bucket.npz',2);selected=after['point_groups']==0
    head={}
    for key,value in before.items():
        actual=after[key][:,0] if key.startswith('track_') else after[key][:,selected] if key=='temporal_dc' else after[key][selected]
        head[key]=np.array_equal(value,actual)
    if not all(head.values()):raise AssertionError('Original bucket changed')
    rotations=torch.from_numpy(after['track_rotations']);translations=torch.from_numpy(after['track_translations']);rng=np.random.default_rng(1102);results=[]
    for group in range(7):
        points=after['points'][after['point_groups']==group]
        points=torch.from_numpy(points[rng.choice(len(points),512,replace=True)])
        distances=(points[:256]-points[256:]).norm(dim=-1);maximum=0.
        for frame in np.arange(0,398.5,.5):
            q,t=pose_at(rotations[:,group],translations[:,group],frame/60);posed=quaternion_rotate(q,points)+t
            maximum=max(maximum,float(((posed[:256]-posed[256:]).norm(dim=-1)-distances).abs().max()))
        if maximum>1e-5:raise AssertionError('A rigid group changes shape')
        results.append(dict(group=group,max_distance_error=maximum))
    relative=(rotations[:,1:]-rotations[:,0,None]).abs().max().item()
    if relative<1e-4:raise AssertionError('All groups unexpectedly share the bucket rotation')
    result=dict(model_sha256=digest(a.candidate/'model.pth'),sidecar_sha256=digest(a.candidate/'rigid-bucket.npz'),retained_base_exact=checks,bucket_exact=head,groups=results,timestamps=797,relative_rotation_difference=relative,scope='Numerical rigid shape invariant, not a photometric quality guarantee. Group zero is original bucket; groups one to six are arm.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
