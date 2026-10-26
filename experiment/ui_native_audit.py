"""Audit raw events/trajectories on the same frozen, baked UI stage assets.

This companion is deliberately pre-display: clay highlights are not physical
sensor radiance. RGB display scores come only from ui_capture WebSocket images.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .protocol import save_json, sha256


def audit(root, scene, version, output):
    import torch
    from .model import Adapter, SceneRenderer
    from .training import restore, evaluate, Supervision
    from utils.gaussian_sequence import GaussianSequence
    from gaussian_renderer import render
    from plyfile import PlyData
    from utils.event_refinement import qrotate
    directory=root/scene
    method={'v1':'C1','v2':'C2','v3':'C3'}[version]
    prior=directory/('v3-prior' if version=='v3' else 'v1-prior')
    asset=directory/('supplied-pair/left.npz' if version=='v1' else version+'.npz')
    checkpoint=directory/(version+'-training/checkpoint.pt') if version!='v1' else None
    adapter=Adapter(prior,method).cuda()
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu')
        if state['config']['inputs']['prior.json']!=sha256(prior/'prior.json'):
            raise ValueError('Audit parent differs from UI stage')
        restore(adapter,state['weights'])
    renderer=SceneRenderer(prior,adapter)
    # Different versions have different Gaussian densities. Evaluate motion
    # using one fixed set of V1 rest-space probes, not each model's own rows.
    base_prior=directory/'v1-prior'
    reference=json.loads((base_prior/'prior.json').read_text())
    with np.load(base_prior/'motion.npz',allow_pickle=False) as z:
        probe_ids=z['sample_indices'].copy();probe_groups=z['groups'][probe_ids].copy()
    vertices=PlyData.read(reference['source'])['vertex'].data
    probes=np.column_stack([vertices[k][probe_ids] for k in ('x','y','z')])
    if reference['groups']!=adapter.metadata['groups']:
        raise ValueError('Cannot compare motion with different group identities')
    probe_xyz=torch.tensor(probes,device='cuda');probe_groups_gpu=torch.tensor(probe_groups.astype(np.int64),device='cuda')
    def common_trajectories(frames):
        q,t=adapter.transforms(frames)
        xyz=qrotate(q[:,probe_groups_gpu],probe_xyz)+t[:,probe_groups_gpu]
        gap,angle=adapter.joint_coordinates(q,t)
        return xyz,gap,angle
    renderer.trajectories=common_trajectories
    sup=Supervision(prior)
    output.mkdir(parents=True,exist_ok=False)
    checks=[]
    with GaussianSequence(asset) as sequence:
        if sequence.metadata.get('provenance',{}).get('ui_stage'):
            identity=sequence.metadata['provenance']['ui_stage']
            if identity['prior_sha256']!=sha256(prior/'prior.json'):
                raise ValueError('Baked UI sequence uses a different prior')
            if checkpoint and identity['checkpoint_sha256']!=sha256(checkpoint):
                raise ValueError('Baked UI sequence uses a different trained checkpoint')
        # This uses the serialized poses themselves, not a fresh motion replay.
        indices=torch.from_numpy(sequence.dynamic_indices.astype(np.int64)).cuda()
        def actual_baked_frame(frame):
            sequence.validate_pose_frame(frame)
            pose=torch.from_numpy(np.ascontiguousarray(sequence.poses[:,frame])).cuda()
            renderer.model._xyz=renderer.xyz.index_copy(0,indices,pose[:,:3])
            renderer.model._rotation=renderer.rotation.index_copy(0,indices,pose[:,3:7])
            checks.append(frame)
            return render(renderer.camera,renderer.model,renderer.pipe,renderer.background,0.,0.,0.)
        renderer.render=actual_baked_frame
        result=evaluate(renderer,sup,save_to=output)
    result['ui_asset_identity']=dict(sequence_sha256=sha256(asset),prior_sha256=sha256(prior/'prior.json'),
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,validated_frames=sorted(set(checks)),
        intensity_scope='Native 640px Gaussian intensity before UI display finish; original sensor background',
        image_source='The exact baked UI Gaussian coordinates, rotations and canonical appearance',
        trajectory_scope='Matching checkpoint transforms on fixed source-group probes; not 2D optical flow',
        display_scores='See actual UI WebSocket capture; native RGB scores must not replace clay display scores')
    result['common_trajectory_probes']=dict(reference_v1_prior_sha256=sha256(base_prior/'prior.json'),
        reference_source_sha256=sha256(reference['source']),ids=probe_ids.tolist(),groups=probe_groups.tolist(),
        positions=probes.tolist(),scope='Identical V1 canonical rest-space probes for all three versions; elastic bands excluded')
    save_json(output/'evaluation.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--scene',required=True)
    p.add_argument('--version',required=True,choices=('v1','v2','v3'))
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();audit(args.root,args.scene,args.version,args.output)


if __name__=='__main__':main()
