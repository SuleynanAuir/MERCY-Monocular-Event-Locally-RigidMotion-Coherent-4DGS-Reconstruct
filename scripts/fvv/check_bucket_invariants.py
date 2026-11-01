"""Check saved non-bucket parameters and actual subframe rigid distances."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import hybrid
from rigid_bucket import Bucket,pose_at,quaternion_rotate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','candidate','source','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    old,_=hybrid.load_checkpoint(a.previous/'model.pth','cpu');new,_=hybrid.load_checkpoint(a.candidate/'model.pth','cpu')
    with np.load(a.source,allow_pickle=False) as data:retired=data['indices']
    keep=torch.ones(len(old[1]),dtype=torch.bool);keep[torch.from_numpy(retired)]=False
    checks={}
    for field in [1,2,3,4,5,6,13,14,15]:checks[str(field)]=torch.equal(old[field][keep],new[field])
    for field in [19,20,21,22,23,24]:checks[str(field)]=torch.equal(old[field],new[field])
    if not all(checks.values()):raise AssertionError('Non-bucket parameters changed')
    meta=json.loads((a.candidate/'manifest.json').read_text())
    if meta['rigid_bucket']['schema']!=1:raise ValueError('Use check_arm_invariants.py for articulated models')
    bucket=Bucket.load(a.candidate,meta['rigid_bucket']);v=bucket.values
    torch.manual_seed(1101);ids=torch.randint(len(v['points']),(1024,),device='cuda');points=v['points'][ids]
    distance=(points[:512]-points[512:]).norm(dim=-1);maximum=0.;normal_error=0.
    with torch.no_grad():
        for frame in np.arange(0,398.5,.5):
            q,t=pose_at(v['track_rotations'],v['track_translations'],frame/60);posed=quaternion_rotate(q,points)+t
            maximum=max(maximum,float(((posed[:512]-posed[512:]).norm(dim=-1)-distance).abs().max()))
            normal_error=max(normal_error,float((q.norm()-1).abs()))
    if maximum>1e-5:raise AssertionError('Rigid bucket changed pairwise distances')
    report=dict(parent_sha256=json.loads((a.previous/'manifest.json').read_text())['sha256'],model_sha256=meta['sha256'],bucket_sha256=meta['rigid_bucket']['sha256'],
        nonbucket_fields_exact=checks,pair_count=512,timestamps=797,max_pair_distance_change=maximum,max_quaternion_norm_error=normal_error,
        scope='Saved-parameter equality and numerical rigid-motion invariant; does not establish photometric quality or correctness of inferred bucket segmentation')
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
