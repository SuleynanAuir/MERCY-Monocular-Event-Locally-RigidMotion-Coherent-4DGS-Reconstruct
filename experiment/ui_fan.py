"""Fan UI stages using video-fitted motion and canonical surface refinement.

Fan has incomplete raw events. Its full-video motion fit is explicitly an
in-sequence UI reconstruction, never a held-out motion prediction result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from .protocol import save_json, sha256


def build(scene, data, root, destination, version, reference_prior=None):
    from .prepare import save_prior
    from .supervision import prepare as supervision
    from utils.fan_surface import regularize
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    motion = json.loads((root/'source-motion/motion-report.json').read_text())
    with np.load(root/'source-motion/fan-motion.npz', allow_pickle=False) as z:
        template_meta = json.loads(str(z['metadata_json']))
        corrected = z['base_vertices'].copy()
        corrected_groups = np.zeros(len(corrected), np.int32)
        corrected_groups[z['dynamic_indices']] = 1
    center = np.asarray(motion['center'])
    angles = np.asarray(motion['angles_rad'])
    delta = np.tile(np.eye(4), (len(angles),2,1,1))
    rotations = Rotation.from_rotvec(np.column_stack((np.zeros((len(angles),2)),angles))).as_matrix()
    delta[:,1,:3,:3] = rotations
    delta[:,1,:3,3] = center-np.einsum('tij,j->ti',rotations,center)
    if version == 'v1':
        source = data/'Fan.ply'
        base = PlyData.read(source)['vertex'].data.copy()
        xyz = np.column_stack([base[k] for k in ('x','y','z')])
        cam = json.loads((data/'fixed_camera.json').read_text())
        world = xyz@np.asarray(cam['R'])+np.asarray(cam['T'])
        u = cam['fx']*world[:,0]/world[:,2]+cam['cx']
        groups = ((u>320)&(xyz[:,2]>0)&(xyz[:,2]<.16)&
                  (np.linalg.norm(xyz[:,:2]-center[:2],axis=1)<.31)).astype(np.int32)
        method = 'Original Fan.ply, basic front-layer rotor binding; same video-fitted timeline as all stages'
    else:
        base, parents, groups, report = regularize(corrected, corrected_groups,
            strength=1, split_budget=300000, maximum_shift=.003, neighbors=256, color_strength=1)
        source = destination/'surface.ply'
        PlyData([PlyElement.describe(base,'vertex')],text=False).write(source)
        np.savez_compressed(destination/'surface-parents.npz', parent_rows=parents, groups=groups)
        save_json(destination/'surface-report.json', report)
        save_json(destination/'display.json', dict(surface_refinement=report,
            auxiliary_trained=False, reason='Complete raw Fan events unavailable'))
        xyz = np.column_stack([base[k] for k in ('x','y','z')])
        method = 'Corrected binding plus bilateral MLS and tangent subdivision; inherited rigid groups'
    centers = np.asarray([xyz[groups == g].mean(0) for g in range(2)])
    prior = save_prior(scene, destination, str(source), xyz, groups, delta, centers,
        template_meta['preview_camera'], [dict(groups=[0,1],pivot=center.tolist(),axis=[0,0,1])],
        np.empty(0,np.int64), dict(method=method, video_motion_sha256=sha256(root/'source-motion/motion-report.json'),
            full_video_motion_exposure=True, raw_event_training=False))
    prior['common_g0'] = 'Exact Fan UI stage canonical geometry; original and surface-refined versions differ'
    prior['joint_scope'] = 'Assumed world-Z fan axle, not independently measured ground truth'
    if reference_prior:
        ref = json.loads(reference_prior.read_text())
        prior['diagonal'] = ref['diagonal']
        prior['frozen_v1_units_prior_sha256'] = sha256(reference_prior)
    save_json(destination/'prior.json', prior)
    template_meta.update(gaussians_per_frame=len(base),
        base_vertices_sha256=hashlib.sha256(base.tobytes()).hexdigest(),
        description=method, default_background='light')
    np.savez_compressed(destination/'canonical-template.npz', base_vertices=base,
                        metadata_json=np.asarray(json.dumps(template_meta)))
    supervision(scene, destination/'supervision')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('inventory','data','root','output'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--version', required=True, choices=('v1','v3'))
    p.add_argument('--reference-prior', type=Path)
    args = p.parse_args()
    scene = next(s for s in json.loads(args.inventory.read_text())['scenes'] if s['scene']=='fan')
    build(scene,args.data,args.root,args.output,args.version,args.reference_prior)


if __name__ == '__main__':
    main()
