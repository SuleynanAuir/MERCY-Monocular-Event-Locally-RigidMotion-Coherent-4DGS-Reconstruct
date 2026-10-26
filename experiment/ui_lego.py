"""Lego data-to-UI geometry stages with exact rigid ownership and elastic bands.

The new surface stage uses the declared Blender geometry prior; it is not an
exact restoration of a missing historical checkpoint or unseen observations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .protocol import save_json, sha256


def prepare(scene, root, source, binding_path, destination, reference_prior=None):
    from .prepare import save_prior
    from .supervision import prepare as supervision
    from scripts.diagnostics.diagnose_lego_joints import diagnose
    from scripts.lamp.prepare_lamp_viewer import rigid_deltas
    from scripts.motion.bind_lego_parts import transform
    from utils.articulated_motion import ArticulatedBaker
    from utils.gaussian_sequence import FORMAT, SPARSE_VERSION, POSE_FIELDS
    from utils.ply_utils import gaussian_degree
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    base = PlyData.read(source)['vertex'].data.copy()
    xyz = np.column_stack([base[k] for k in ('x','y','z')])
    with np.load(root/'parts.npz', allow_pickle=False) as z:
        parts = {k: z[k] for k in z.files}
    with np.load(binding_path, allow_pickle=False) as z:
        binding = {k: z[k] for k in z.files}
    report = diagnose(parts, max_surface_samples=3000)
    reg = binding['registration']
    delta = rigid_deltas(parts['object_matrices'], reg)
    mapping = np.full(delta.shape[1], -1, np.int32)
    for group in report['groups']:
        mapping[group['objects']] = group['group']
    labels = mapping[binding['labels']]
    representatives = [g['representative_object'] for g in report['groups']]
    group_delta = delta[:, representatives]
    bands = np.flatnonzero(labels < 0)
    # One stable identity group also contains deliberately unattached floaters.
    attached = np.zeros(len(base), bool)
    attached[binding['dynamic_indices']] = True
    labels[~attached | (labels < 0)] = len(representatives)
    group_delta = np.concatenate((group_delta,
        np.broadcast_to(np.eye(4), (len(delta),1,4,4))), axis=1)
    bands = bands[attached[bands]]
    centers = np.asarray([xyz[labels == g].mean(0) if np.any(labels == g) else np.zeros(3)
                          for g in range(group_delta.shape[1])])
    joints = []
    for edge in report['candidate_joints']:
        if edge['rank'] == 2 and edge.get('rest_axis') is not None:
            axis = reg[:3,:3]@np.asarray(edge['rest_axis']); axis /= np.linalg.norm(axis)
            joints.append(dict(groups=edge['groups'], pivot=transform(np.asarray(edge['rest_pivot']),reg).tolist(), axis=axis.tolist()))
    if len(bands):
        band_binding = dict(binding, dynamic_indices=bands.astype(np.int32))
        path = destination/'band-binding.npz'
        np.savez_compressed(path, **band_binding)
        baker = ArticulatedBaker(base, root/'parts.npz', path)
        poses = np.lib.format.open_memmap(destination/'bands.npy', mode='w+', dtype='float32',
                                        shape=(len(delta),len(bands),7))
        for frame in range(len(delta)):
            poses[frame] = baker.frame(frame)
        poses.flush(); del poses
    camera = json.loads((root/'inspection/inspection.json').read_text())['camera']
    prior = save_prior(scene, destination, str(source), xyz, labels, group_delta, centers,
        camera, joints, bands, dict(parts_sha256=sha256(root/'parts.npz'),
            binding_sha256=sha256(binding_path), registration=reg.tolist(),
            geometry_scope='This exact UI-stage canonical asset', band_rows=len(bands)))
    prior['common_g0'] = 'Exact canonical source of this UI version; different versions retain their own geometry'
    if reference_prior:
        ref = json.loads(reference_prior.read_text())
        if ref['groups'] != prior['groups'] or ref['camera'] != prior['camera']:
            raise ValueError('Lego stage group/camera mismatch')
        prior['diagonal'], prior['joints'] = ref['diagonal'], ref['joints']
        prior['frozen_v1_units_prior_sha256'] = sha256(reference_prior)
    save_json(destination/'prior.json', prior)
    count, fps = scene['video']['frame_count'], scene['video']['fps']
    meta = dict(format=FORMAT, version=SPARSE_VERSION, name='Lego', frame_count=count, fps=fps,
        duration_seconds=count/fps, gaussians_per_frame=len(base), sh_degree=gaussian_degree(source),
        pose_fields=POSE_FIELDS, frames=[dict(time_seconds=i/fps, file=f'frames/frame_{i:04d}.ply') for i in range(count)],
        base_vertices_sha256=hashlib.sha256(base.tobytes()).hexdigest(), preview_camera=camera,
        motion_is_baked=True, provenance=dict(source_ply_sha256=sha256(source),
            parts_sha256=sha256(root/'parts.npz'), binding_sha256=sha256(binding_path)))
    np.savez_compressed(destination/'canonical-template.npz', base_vertices=base,
                        metadata_json=np.asarray(json.dumps(meta)))
    supervision(scene, destination/'supervision')


def surface(root, corrected, output, spacing=.0016):
    from utils.mesh_surface_sampling import sample_mesh, normals_to_wxyz
    from scripts.motion.bind_lego_parts import transform
    if output.exists():
        raise FileExistsError(output)
    with np.load(root/'source-mesh.npz', allow_pickle=False) as z:
        mesh = {k:z[k] for k in z.files}
    mesh_meta = json.loads(str(mesh['metadata_json']))
    if mesh_meta['parts_sha256'] != sha256(root/'parts.npz') or not mesh_meta['autoexec_disabled']:
        raise ValueError('Source mesh identity mismatch')
    samples = sample_mesh(mesh['vertices'], mesh['triangles'], mesh['part_ids'], mesh['material_gray'],
        mesh_meta['deforming_objects'], spacing)
    print('Surface sample plan:', json.dumps(samples['stats']), flush=True)
    base = PlyData.read(corrected)['vertex'].data.copy()
    xyz = np.column_stack([base[k] for k in ('x','y','z')])
    with np.load(root/'corrected-binding.npz', allow_pickle=False) as z:
        old = {k:z[k] for k in z.files}
    retained = np.flatnonzero(np.isin(old['labels'], mesh_meta['deforming_objects']))
    added = np.zeros(len(samples['xyz']), dtype=base.dtype)
    for i,k in enumerate(('x','y','z')):
        added[k] = samples['xyz'][:,i]
    normal = samples['normals']
    quat = normals_to_wxyz(normal)
    for i in range(4): added[f'rot_{i}'] = quat[:,i]
    for i in range(2): added[f'scale_{i}'] = np.log(spacing*.75)
    added['scale_2'] = np.log(spacing*.10)
    added['opacity'] = np.log(.95/.05)
    colors = .5+.28209479177387814*np.column_stack([base[f'f_dc_{i}'] for i in range(3)])
    result_color = np.repeat(samples['material_gray'][:,None],3,axis=1)
    supported = 0
    # Appearance transfer stays within the same actual rigid part.
    for part in np.unique(samples['part_ids']):
        rows = np.flatnonzero(samples['part_ids'] == part)
        originals = np.flatnonzero((old['labels'] == part) & (base['opacity'] > 0))
        if len(originals) < 4: continue
        distance, nearest = cKDTree(xyz[originals]).query(samples['xyz'][rows], k=4, workers=4)
        valid = distance[:,0] < .012
        result_color[rows[valid]] = np.median(colors[originals[nearest[valid]]],axis=1)
        supported += int(valid.sum())
    for i in range(3): added[f'f_dc_{i}'] = (result_color[:,i]-.5)/.28209479177387814
    combined = np.concatenate((base[retained], added))
    labels = np.concatenate((old['labels'][retained], samples['part_ids']))
    with np.load(root/'parts.npz', allow_pickle=False) as z:
        parts = {k:z[k] for k in z.files}
    points = transform(parts['rest_points'], old['registration'])
    nearest = np.empty(len(added), np.int32)
    distance = np.empty(len(added), np.float32)
    for part in np.unique(samples['part_ids']):
        rows = np.flatnonzero(samples['part_ids'] == part)
        ids = np.flatnonzero(parts['point_parts'] == part)
        dist, local = cKDTree(points[ids]).query(samples['xyz'][rows], workers=4)
        nearest[rows], distance[rows] = ids[local], dist
    matrices = parts['object_matrices']
    delta = matrices@np.linalg.inv(matrices[0])
    moving = np.max(abs(delta-np.eye(4)),axis=(0,2,3)) > 1e-4
    moving[mesh_meta['deforming_objects']] = True
    metadata = json.loads(str(old['metadata_json']))
    metadata.update(method='Exact source rigid-part ownership of surface Gaussians; elastic rows retain original binding',
                    source_mesh_sha256=sha256(root/'source-mesh.npz'), gaussian_count=len(combined))
    output.mkdir(parents=True)
    PlyData([PlyElement.describe(combined,'vertex')],text=False).write(output/'surface.ply')
    np.savez_compressed(output/'binding.npz', registration=old['registration'], labels=labels,
        nearest_vertex=np.concatenate((old['nearest_vertex'][retained],nearest)),
        distance=np.concatenate((old['distance'][retained],distance)),
        dynamic_indices=np.flatnonzero(moving[labels]).astype(np.int32), metadata_json=np.asarray(json.dumps(metadata)))
    save_json(output/'report.json', dict(samples['stats'], input_sha256=sha256(corrected),
        source_mesh_sha256=sha256(root/'source-mesh.npz'), output_sha256=sha256(output/'surface.ply'),
        appearance_supported_samples=supported, retained_elastic_rows=len(retained),
        limitation='New full source-surface prior adaptation, not the historical Lego completion checkpoint; unavailable colors use declared source material prior'))
    save_json(output/'display.json', dict(surface_refinement='Declared Blender rigid surface prior; standard UI soft-clay finish'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','surface'))
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--binding', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--reference-prior', type=Path)
    args = p.parse_args()
    if args.action == 'surface':
        surface(args.root, args.source, args.output)
    else:
        scene = next(s for s in json.loads(args.inventory.read_text())['scenes'] if s['scene']=='lego')
        prepare(scene, args.root, args.source, args.binding, args.output, args.reference_prior)


if __name__ == '__main__':
    main()
