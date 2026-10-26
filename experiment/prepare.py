"""Build frozen geometric priors, with no held-out RGB fitting."""
import json
import subprocess
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .protocol import save_json, sha256


def prepare(scene, destination, blender):
    from scripts.lamp.prepare_lamp_viewer import camera_pose, register, rigid_deltas
    from scripts.lamp.prepare_object_viewer import register_without_camera, registered_camera
    from scripts.motion.bind_lego_parts import transform, similarity
    from scripts.diagnostics.diagnose_lego_joints import diagnose
    root=Path(__file__).resolve().parents[1];destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'prior.json').exists():raise FileExistsError(destination/'prior.json')
    files=scene['files'];name=scene['scene'];video=scene['video']
    # Every method receives the SAME supplied corrected G0. Its historical
    # training exposure is unknown and is explicitly outside this experiment.
    source=files['corrected']['path']
    vertices=PlyData.read(source)['vertex'].data
    xyz=np.column_stack([vertices[k] for k in ('x','y','z')])
    if name=='fan':
        from .fan_prior import build
        return build(scene,destination,vertices)
    partfile=destination/'parts.npz'
    if not partfile.exists():
        script='scripts/motion/export_blender_parts.py' if name=='lego' else 'scripts/lamp/export_lamp_motion.py'
        with (destination/'export.log').open('w') as log:
            subprocess.run([str(blender),'-b','--disable-autoexec',files['blend']['path'],
                '--python',str(root/script),'--','--output',str(partfile)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with np.load(partfile,allow_pickle=False) as z:parts={k:z[k] for k in z.files}
    meta=json.loads(str(parts['metadata_json']))
    if len(parts['object_matrices'])!=video['frame_count'] or not meta['autoexec_disabled']:
        raise ValueError('Source animation timeline/autoexec mismatch')
    intrinsics=json.loads(Path(files['intrinsics']['path']).read_text())
    fovy=float(np.rad2deg(2*np.arctan(video['height']/(2*intrinsics['fy']))))
    if files['camera']['exists']:
        camera=camera_pose(json.loads(Path(files['camera']['path']).read_text()))
        if name=='lego':
            # Same geometric registration used in the original source binder;
            # no RGB validation or test images enter ICP.
            reg=np.asarray(camera['pose'])@np.diag([.40884]*3+[1])@np.linalg.inv(meta['camera_pose'])
            tree=cKDTree(parts['rest_points']);sample=xyz[::max(1,len(xyz)//40000)]
            for _ in range(250):
                d,ids=tree.query(transform(sample,np.linalg.inv(reg)),workers=4)
                keep=d<np.quantile(d,.85);update=similarity(parts['rest_points'][ids[keep]],sample[keep])
                change=np.max(abs(update-reg));reg=update
                if change<1e-7:break
        else:reg,_=register(parts['rest_points'],xyz,meta['camera_pose'],camera['pose'])
    else:
        reg,_=register_without_camera(parts['rest_points'],xyz)
        camera=registered_camera(meta['camera_pose'],reg,xyz.mean(0),fovy)
    if abs(camera['fovy_degrees']-fovy)>.02:raise ValueError('Camera focal mismatch')
    delta=rigid_deltas(parts['object_matrices'],reg)
    surface=transform(parts['rest_points'],reg)
    distances,nearest=cKDTree(surface).query(xyz,workers=4)
    labels=parts['point_parts'][nearest]
    report=diagnose(parts,max_surface_samples=3000)
    report['source_parts_sha256']=sha256(partfile)
    save_json(destination/'joints.json',report)
    groups=report['groups'];mapping=np.full(delta.shape[1],-1,np.int32)
    for g in groups:mapping[g['objects']]=g['group']
    row_groups=mapping[labels]
    representatives=[g['representative_object'] for g in groups]
    gd=delta[:,representatives]
    bands=np.flatnonzero(row_groups<0).astype(np.int64)
    if len(bands):
        from utils.articulated_motion import ArticulatedBaker
        np.savez_compressed(destination/'band-binding.npz',labels=labels,dynamic_indices=bands,
            registration=reg,nearest_vertex=nearest,metadata_json=np.asarray('{}'))
        baker=ArticulatedBaker(vertices,partfile,destination/'band-binding.npz')
        cache=np.lib.format.open_memmap(destination/'bands.npy',mode='w+',dtype='float32',
                                       shape=(video['frame_count'],len(bands),7))
        for frame in range(video['frame_count']):cache[frame]=baker.frame(frame)
        cache.flush();del cache
    # A nonrigid band gets an identity placeholder group; its baked motion
    # replaces this transform, and it is excluded from learned rigid residuals.
    if len(bands):
        row_groups[bands]=len(groups)
        gd=np.concatenate([gd,np.broadcast_to(np.eye(4),(len(gd),1,4,4))],axis=1)
    centers=np.asarray([xyz[row_groups==g].mean(0) if np.any(row_groups==g)
                        else transform(parts['rest_points'][parts['point_parts']==representatives[g]],reg).mean(0)
                        for g in range(len(groups))])
    if len(bands):centers=np.concatenate([centers,xyz[bands].mean(0,keepdims=True)])
    joints=[]
    for edge in report['candidate_joints']:
        if edge['rank']==2 and edge.get('rest_axis') is not None:
            axis=reg[:3,:3]@np.asarray(edge['rest_axis']);axis/=np.linalg.norm(axis)
            joints.append(dict(groups=edge['groups'],pivot=transform(np.asarray(edge['rest_pivot']),reg).tolist(),axis=axis.tolist()))
    return save_prior(scene,destination,source,xyz,row_groups,gd,centers,camera,joints,bands,
                      dict(registration=reg.tolist(),parts_sha256=sha256(partfile),
                           registration_distance_quantiles=np.quantile(distances,[.5,.9,.99]).tolist(),
                           attachment='nearest source surface; inferred labels, not ground-truth segmentation',
                           source_group_count=len(groups),band_rows=len(bands)))


def save_prior(scene,destination,source,xyz,groups,delta,centers,camera,joints,bands,details):
    count=len(delta);q=Rotation.from_matrix(delta[...,:3,:3].reshape(-1,3,3)).as_quat().reshape(count,-1,4)[...,[3,0,1,2]]
    for i in range(1,count):q[i]*=np.where((q[i]*q[i-1]).sum(-1)<0,-1.,1.)[:,None]
    q[0]=[1,0,0,0];t=delta[...,:3,3];t[0]=0
    active=np.max(abs(delta-np.eye(4)),axis=(0,2,3))>1e-5
    positions=np.einsum('tgij,gj->tgi',delta[...,:3,:3],centers)+t
    # Rotation excitation also activates groups whose centroid lies on an axle.
    movement=np.square(positions-positions[:1]).sum(-1)+np.square(q-q[:1]).sum(-1)
    envelope=movement/np.maximum(movement.max(0),1e-12);envelope[:,~active]=0
    rng=np.random.default_rng(20261026);samples=[]
    for g in np.flatnonzero(active):
        candidates=np.flatnonzero(groups==g)
        if len(candidates):samples.extend(rng.choice(candidates,min(64,len(candidates)),replace=False).tolist())
    np.savez_compressed(destination/'motion.npz',groups=groups.astype(np.int32),q=q.astype(np.float32),
        t=t.astype(np.float32),centers=centers.astype(np.float32),envelope=envelope.astype(np.float32),
        active=active,band_indices=bands,sample_indices=np.asarray(samples,dtype=np.int64))
    record=dict(format='mercy_five_scene_adapter_prior_v1',scene=scene['scene'],source=source,
        source_sha256=sha256(source),motion_sha256=sha256(destination/'motion.npz'),
        video=scene['video'],camera=camera,joints=joints,joint_scope='source-prior revolute candidates; not measured physical ground truth',
        diagonal=float(np.linalg.norm(np.ptp(xyz,axis=0))),gaussians=len(xyz),groups=len(centers),
        details=details,parent_exposure=scene['prior_exposure'],
        common_g0='supplied corrected G0, identical for C1/C2/C3; appearance and density frozen')
    save_json(destination/'prior.json',record)
    print('Prepared prior',scene['scene'],len(xyz),'Gaussians',len(centers),'groups',len(joints),'joints',flush=True)
    return record
