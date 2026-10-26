"""Recover compact, checked rigid fits from the exact displayed baked positions.

Fits are to fixed samples from the existing geometric binding, not motion GT.
No source observation or reference motion is substituted for method predictions.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.experiment_format import sha256, save_json
from utils.experiment_export import joint_snapshot
from utils.mesh_motion import quaternion_product


def joint_components(q, pairs, axes):
    relative = quaternion_product(q[:,pairs[:,0]]*[1,-1,-1,-1],q[:,pairs[:,1]])
    projected = (relative[...,1:]*axes).sum(-1)
    perpendicular = np.linalg.norm(relative[...,1:]-projected[...,None]*axes,axis=-1)
    return dict(joint_twist_angles=2*np.arctan2(projected,relative[...,0]),
                joint_swing_angles=2*np.arctan2(perpendicular,np.hypot(relative[...,0],projected)))


def fit_motion(rest, points, groups, joints):
    count = int(groups.max())+1
    qs, ts, errors = [], [], []
    for group in range(count):
        selected = groups == group
        a, b = rest[selected].astype(float), points[:,selected].astype(float)
        if len(a) < 3 or np.linalg.matrix_rank(a-a.mean(0), tol=1e-7) < 2:
            raise ValueError(f'Group {group} lacks noncollinear samples')
        ac, bc = a.mean(0), b.mean(1)
        u, _, vt = np.linalg.svd(np.einsum('ni,tnj->tij', a-ac, b-bc[:,None]))
        rotation = vt.transpose(0,2,1) @ u.transpose(0,2,1)
        reflected = np.linalg.det(rotation) < 0
        vt[reflected,-1,:] *= -1
        rotation = vt.transpose(0,2,1) @ u.transpose(0,2,1)
        translation = bc - np.einsum('tij,j->ti',rotation,ac)
        expected = np.einsum('tij,nj->tni',rotation,a)+translation[:,None]
        errors.append(np.linalg.norm(expected-b,axis=-1).max(1))
        qs.append(Rotation.from_matrix(rotation).as_quat()[:,[3,0,1,2]])
        ts.append(translation)
    q, t, error = np.stack(qs,1), np.stack(ts,1), np.stack(errors,1)
    if error.max() > 2e-5:
        raise ValueError(f'Sampled groups are not rigid: residual {error.max():.8g}, group {np.argmax(error.max(0))}')
    structure = SimpleNamespace(**joints)
    snapshots = [joint_snapshot(structure, frame) for frame in q]
    result = dict(group_quaternions=q.astype(np.float32), group_translations=t.astype(np.float32),
                  rigid_fit_max_error=error.astype(np.float32), **joints)
    result.update(joint_components(q,joints['joint_groups'],joints['joint_axes']))
    if all('joint_angles' in row for row in snapshots):
        result['joint_angles'] = np.asarray([row['joint_angles'] for row in snapshots],dtype=np.float64)
    return result


def selected_points(xyz, poses, dynamic, groups):
    ids = np.concatenate([members[np.linspace(0,len(members)-1,min(64,len(members)),dtype=int)]
        for g in sorted(np.unique(groups)) if len(members:=np.flatnonzero(groups==g))])
    if len(ids)>1024: raise ValueError('Too many motion samples')
    lookup = np.full(len(xyz),-1,dtype=np.int64);lookup[dynamic]=np.arange(len(dynamic))
    moving = lookup[ids]>=0
    points = np.broadcast_to(xyz[ids],(len(poses),len(ids),3)).copy()
    points[:,moving] = poses[:,lookup[ids[moving]],:3]
    return ids, points


def prepare(restored, archived, output):
    from plyfile import PlyData
    output.mkdir(parents=True,exist_ok=False)
    manifest = json.loads((restored/'manifest.json').read_text())
    baseline = Path(manifest['baseline']);cache=json.loads((baseline/'cache.json').read_text())
    correction=json.loads((Path(manifest['candidate'])/'correction.json').read_text())['joint_structure']
    metadata=json.loads((restored/'init/initial-binding.json').read_text())
    if sha256(Path(cache['source'])) != metadata['source_ply_sha256']: raise ValueError('V1 binding source mismatch')
    with np.load(restored/'init/initial-binding.npz',allow_pickle=False) as binding:
        labels=binding['labels'];registration=binding['registration']
    vertices=PlyData.read(cache['source'])['vertex'].data
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')]);del vertices
    groups=np.full(len(xyz),-1,dtype=np.int32)
    for group in correction['source_groups']:
        groups[np.isin(labels,group['objects'])]=group['group']
    dynamic=np.load(baseline/'dynamic_indices.npy',allow_pickle=False)
    # Filtered-out attachments retain static coordinates and are not rigid rows.
    active=np.zeros(len(xyz),bool);active[dynamic]=True
    groups[(groups>0)&~active]=-2
    # Unknown/unbound rows are still saved as point trajectories, never rigid fits.
    groups[groups<0]=-1
    poses=np.load(baseline/'poses.npy',mmap_mode='r',allow_pickle=False)
    right=json.loads((Path(manifest['right_baseline'])/'cache.json').read_text())
    transfer=registration @ np.linalg.inv(right['binding']['registration'])
    pivots=np.asarray(correction['joint_pivots'])@transfer[:3,:3].T+transfer[:3,3]
    axes=np.asarray(correction['joint_axes'])@transfer[:3,:3].T;axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    joints=dict(joint_groups=np.asarray(correction['joint_groups']),joint_pivots=pivots,joint_axes=axes)
    records={}
    def write(version,xyz,poses,dynamic,groups,joints,provenance):
        ids,points=selected_points(xyz,poses,dynamic,groups)
        fitted=fit_motion(xyz[ids],points,groups[ids],joints)
        fitted['joint_pairs']=fitted.pop('joint_groups')
        path=output/f'v{version+1}-motion.npz'
        np.savez_compressed(path,point_ids=ids,points=points,point_groups=groups[ids],
            frames=np.arange(len(poses)),times_seconds=np.arange(len(poses))/60,**fitted)
        records[str(version)]=dict(path=path.name,sha256=sha256(path),point_count=len(xyz),
            sampled_points=len(ids),max_fit_residual=float(fitted['rigid_fit_max_error'].max()),
            joint_angles_available='joint_angles' in fitted,provenance=provenance,
            scope='Least-squares rigid transforms fitted to fixed actual rendered samples, checked below 2e-5 model-world units. Geometric attachment groups and declared source-prior joints; not physical joint ground truth. Negative groups excluded from rigid fits.')
        print(json.dumps(dict(version=version,**records[str(version)])),flush=True)
    write(0,xyz,poses,dynamic,groups,joints,dict(source_sha256=sha256(Path(cache['source'])),
        binding_sha256=sha256(restored/'init/initial-binding.npz'),poses_sha256=sha256(baseline/'poses.npy'),
        source_objects_by_group={str(g['group']):g['objects'] for g in correction['source_groups']}))
    del xyz,poses,groups,labels
    with np.load(archived/'v2.npz',allow_pickle=False) as data:
        meta=json.loads(str(data['metadata_json']));vertices=data['base_vertices']
        xyz=np.column_stack([vertices[k] for k in ('x','y','z')]);del vertices
        print('Loading baked V2 poses',flush=True)
        poses=data['poses'].transpose(1,0,2);dynamic=data['dynamic_indices']
    prior=json.loads((archived/'v1-prior/prior.json').read_text())
    if sha256(archived/'v1-prior/prior.json') != meta['provenance']['ui_stage']['prior_sha256']:raise ValueError('V2 prior mismatch')
    if sha256(archived/'v1-prior/motion.npz') != prior['motion_sha256']:raise ValueError('V2 motion grouping mismatch')
    with np.load(archived/'v1-prior/motion.npz',allow_pickle=False) as motion:
        groups=motion['groups'].copy();groups[motion['band_indices']]=-1
        # This adapter has an empty kinematic root 0 and a separate inactive
        # background group. Alias only after verifying identical source motion;
        # its actual baked background samples must then pass the same rigid fit.
        background=len(motion['active'])-1
        if 0 not in groups:
            if (motion['active'][0] or motion['active'][background]
                    or not np.allclose(motion['q'][:,0],motion['q'][:,background],atol=1e-6,rtol=0)
                    or not np.allclose(motion['t'][:,0],motion['t'][:,background],atol=1e-6,rtol=0)):
                raise ValueError('Unobserved root is not the declared fixed background')
            groups[groups==background]=0
    joints=dict(joint_groups=np.asarray([j['groups'] for j in prior['joints']]),
        joint_pivots=np.asarray([j['pivot'] for j in prior['joints']]),joint_axes=np.asarray([j['axis'] for j in prior['joints']]))
    write(1,xyz,poses,dynamic,groups,joints,dict(sequence_sha256=sha256(archived/'v2.npz'),
        prior_sha256=sha256(archived/'v1-prior/prior.json'),motion_groups_sha256=prior['motion_sha256'],
        root_samples='Inactive background group mapped to root after source-transform equality check; actual sampled positions revalidated',
        joint_scope=prior['joint_scope']))
    save_json(output/'evidence.json',dict(format='scgs_sampled_motion_evidence_v1',
        base_manifest_sha256=sha256(restored/'manifest.json'),middle_sequence_sha256=sha256(archived/'v2.npz'),versions=records))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restored',type=Path,required=True);parser.add_argument('--archived',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.restored,args.archived,args.output)
