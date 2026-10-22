"""Append unseen mesh-prior surfaces without changing existing Gaussians/motion.

Only actual triangle samples can be added. New points must be visible in a
sampled orbit view, and must not be visible in any sampled observed-camera
view. Neither selection nor a mesh coverage test establishes unseen truth.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scripts.geometry.export_lego_multiview_masks import quaternion_matrices
from scripts.events.export_lego_event_preview import verify_candidate
from utils.mesh_surface_sampling import sample_mesh, normals_to_wxyz, projection
from utils.rigid_checkpoint import sha256_file
from utils.render_sampling import require_same_sampling


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def moved(xyz, group_ids, quaternion, translation):
    rotations = quaternion_matrices(quaternion).astype(np.float32)
    return np.einsum('nij,nj->ni', rotations[group_ids], xyz)+translation[group_ids]


def validate_exposure_reuse(metadata, args, cache, parent_source_sha256, parent_correction_sha256):
    exposure = metadata.get('exposure_pruning')
    if exposure is None:
        return
    if (exposure.get('format') != 'lego_raster_alpha_exposure_pruning_v1'
            or exposure.get('converged_full_400') is not True
            or exposure.get('final_pass_frames') != 400
            or exposure.get('parent_source_sha256') != parent_source_sha256
            or exposure.get('parent_correction_sha256') != parent_correction_sha256
            or exposure.get('camera') != cache['camera']
            or exposure.get('retained_count') != metadata['added_points']):
        raise ValueError('Exposure selection is unconverged or uses another parent/camera')
    require_same_sampling(cache, exposure['render_sampling'])
    for key in ('spacing', 'tangent_factor', 'normal_factor', 'opacity'):
        if exposure['render_model_parameters'][key] != getattr(args, key):
            raise ValueError('Exposure selection has a different Gaussian '+key)


def visibility(xyz, parts, groups, motion, mask_path, tolerance, *, progress_label, mesh_sha256):
    with np.load(mask_path, allow_pickle=False) as archive:
        masks, depths = archive['part_ids'], archive['depth']
        metadata = json.loads(str(archive['metadata_json']))
    if metadata['motion_sha256'] != sha256_file(motion['path']):
        raise ValueError('Visibility masks use a different corrected motion')
    if metadata['mesh_sha256'] != mesh_sha256:
        raise ValueError('Visibility masks use a different source mesh')
    result = np.zeros(len(xyz), bool)
    height, width = masks.shape[1:]
    for index, view in enumerate(metadata['views']):
        rows = np.flatnonzero(~result)
        if not len(rows):
            break
        frame = view['frame']
        world = moved(xyz[rows], groups[rows], motion['quaternions'][frame], motion['translations'][frame])
        xy, z = projection(world, np.asarray(view['camera_pose']), width, height,
                           metadata['camera']['fovy_degrees'])
        finite = np.isfinite(xy).all(1) & np.isfinite(z) & (z > 0)
        pixels = np.zeros_like(xy, dtype=np.int32)
        pixels[finite] = np.rint(xy[finite]).astype(np.int32)
        valid = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        selected = np.flatnonzero(valid)
        x, y = pixels[selected].T
        front = depths[index, y, x]
        # Correct part ownership avoids transferring appearance through a hole
        # to another component; depth avoids keeping its occluded back face.
        visible = ((masks[index, y, x] == parts[rows[selected]]) & np.isfinite(front)
                   & (np.abs(z[selected]-front) <= tolerance))
        result[rows[selected[visible]]] = True
        if index % 8 == 0 or index == len(masks)-1:
            print(json.dumps({'stage': progress_label, 'view': index+1, 'views': len(masks),
                              'visible_points': int(result.sum())}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('mesh', 'motion', 'orbit-masks', 'front-masks', 'baseline', 'candidate',
                 'parent-report', 'binding', 'parts', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--spacing', type=float, default=.0016)
    p.add_argument('--guard-tolerance', type=float, default=.0009)
    p.add_argument('--surface-tolerance', type=float, default=.0024)
    p.add_argument('--minimum-distance-factor', type=float, default=.4)
    p.add_argument('--tangent-factor', type=float, default=.75)
    p.add_argument('--normal-factor', type=float, default=.12)
    p.add_argument('--opacity', type=float, default=.85)
    p.add_argument('--selection', type=Path, help='Reuse a previously verified geometric selection, never resample silently')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    settings = (args.spacing, args.guard_tolerance, args.surface_tolerance,
                args.minimum_distance_factor, args.tangent_factor, args.normal_factor, args.opacity)
    if (not all(np.isfinite(x) and x > 0 for x in settings) or not .0005 <= args.spacing <= .004
            or args.opacity >= 1 or args.tangent_factor > 1.5 or args.normal_factor > .5):
        raise ValueError('Invalid bounded surface-completion settings')
    args.output = args.output.resolve()
    cache, parent_correction, parent_audit = verify_candidate(args.baseline, args.candidate, allow_unapproved=True)
    if 'multiview_completion' in cache:
        raise ValueError('Append only to the reviewed v3 parent, not recursively to previous completion attempts')
    parent_report = json.loads(args.parent_report.read_text())
    if parent_report['rebuilt_sha256'] != sha256_file(cache['source']):
        raise ValueError('Parent report and source differ')
    with np.load(args.motion, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    motion['path'] = args.motion
    motion_meta = json.loads(str(motion['metadata_json']))
    if motion_meta['source_correction_sha256'] != sha256_file(args.candidate/'correction.json'):
        raise ValueError('Mesh motion differs from inherited event correction')
    with np.load(args.binding, allow_pickle=False) as archive:
        parent_labels = archive['labels']
    if sha256_file(args.binding) != parent_correction['joint_structure']['input_sha256']['binding']:
        raise ValueError('Parent binding differs from the event-corrected motion binding')
    ply = PlyData.read(cache['source']); parent = ply['vertex'].data
    if len(parent_labels) != len(parent):
        raise ValueError('Parent binding size differs')
    if args.selection:
        with np.load(args.selection, allow_pickle=False) as archive:
            chosen = {key: archive[key] for key in archive.files}
        selection_meta = json.loads(str(chosen['metadata_json']))
        for key, path in (('mesh', args.mesh), ('motion', args.motion), ('orbit_masks', args.orbit_masks),
                          ('front_masks', args.front_masks), ('parent_source', Path(cache['source']))):
            if selection_meta['input_sha256'][key] != sha256_file(path):
                raise ValueError('Reused selection provenance mismatch: '+key)
        for key in ('spacing', 'guard_tolerance', 'surface_tolerance', 'minimum_distance_factor'):
            if selection_meta[key] != getattr(args, key):
                raise ValueError('Reused selection has a different '+key)
        validate_exposure_reuse(selection_meta, args, cache, sha256_file(cache['source']),
                                sha256_file(args.candidate/'correction.json'))
        if 'exposure_pruning' in selection_meta:
            exposure = selection_meta['exposure_pruning']
            for key in ('source_selection', 'source_completion_report'):
                if sha256_file(exposure[key]) != exposure[key+'_sha256']:
                    raise ValueError('Exposure provenance changed: '+key)
            sidecar_path = args.selection.with_suffix('.json')
            sidecar = json.loads(sidecar_path.read_text())
            if (sidecar.get('output_sha256') != sha256_file(args.selection)
                    or sidecar.get('status') != 'CONVERGED_FULL_400_REQUIRES_REBUILD_AND_EVENT_AUDIT'
                    or any(sidecar.get(key) != value for key, value in exposure.items())):
                raise ValueError('Filtered selection differs from its converged exposure output report')
            from scripts.geometry.prune_lego_completion_exposure import resume_active_mask, SELECTION_KEYS
            with np.load(exposure['source_selection'], allow_pickle=False) as archive:
                original_selection = {key: archive[key] for key in archive.files}
            retained_rows = resume_active_mask(original_selection, chosen, {})
            if any(not np.array_equal(chosen[key], original_selection[key][retained_rows])
                   for key in SELECTION_KEYS):
                raise ValueError('Filtered selection changed source row order or attributes')
    else:
        with np.load(args.mesh, allow_pickle=False) as archive:
            mesh_meta = json.loads(str(archive['metadata_json']))
            sampled = sample_mesh(archive['vertices'], archive['triangles'], archive['part_ids'],
                                  archive['material_gray'], mesh_meta['deforming_objects'], args.spacing)
        if mesh_meta['parts_sha256'] != sha256_file(args.parts):
            raise ValueError('Mesh prior differs from source parts')
        xyz, parts = sampled['xyz'], sampled['part_ids']
        groups = motion['group_ids'][parts]
        if np.any(groups < 0):
            raise ValueError('Rigid completion includes an elastic band')
        print(json.dumps({'stage': 'sampled', **sampled['stats']}), flush=True)
        external = visibility(xyz, parts, groups, motion, args.orbit_masks, args.surface_tolerance,
                              progress_label='orbit_surface_visibility', mesh_sha256=sha256_file(args.mesh))
        selected = np.flatnonzero(external)
        observed = visibility(xyz[selected], parts[selected], groups[selected], motion,
                              args.front_masks, args.guard_tolerance, progress_label='observed_camera_guard',
                              mesh_sha256=sha256_file(args.mesh))
        selected = selected[~observed]
        prior_xyz = np.column_stack([parent[name] for name in ('x', 'y', 'z')])
        covered = np.zeros(len(selected), bool)
        for part in np.unique(parts[selected]):
            rows = np.flatnonzero(parts[selected] == part)
            old = np.flatnonzero(parent_labels == part)
            if len(old):
                distance, _ = cKDTree(prior_xyz[old]).query(xyz[selected[rows]], workers=2)
                covered[rows] = distance < args.spacing*args.minimum_distance_factor
        selected = selected[~covered]
        chosen = {key: sampled[key][selected] for key in ('xyz', 'normals', 'part_ids', 'material_gray', 'triangle_ids')}
        selection_meta = {'format': 'lego_unobserved_prior_surface_selection_v1', 'spacing': args.spacing,
            'sample_stats': sampled['stats'], 'orbit_visible': int(external.sum()),
            'excluded_observed_camera_points': int(observed.sum()), 'excluded_existing_surface_points': int(covered.sum()),
            'added_points': len(selected), 'guard_tolerance': args.guard_tolerance,
            'surface_tolerance': args.surface_tolerance, 'minimum_distance_factor': args.minimum_distance_factor,
            'input_sha256': {key: sha256_file(path) for key, path in {
                'mesh': args.mesh, 'motion': args.motion, 'orbit_masks': args.orbit_masks,
                'front_masks': args.front_masks, 'parent_source': cache['source']}.items()},
            'scope': 'Sampled orbit-visible, observed-camera-occluded actual triangle points. Visibility is sampled, not a guarantee for all continuous times/views.'}
    count = len(chosen['xyz'])
    if not 1 <= count <= 600000:
        raise ValueError(f'{count} additions exceed explicit 1..600000 bound; no silent truncation')
    needed = parent.nbytes + count*parent.dtype.itemsize + 400*640*640*2 + 250*1024**2
    if shutil.disk_usage(args.output.parent).free < needed:
        raise OSError('Not enough data disk for new PLY, full event audit and reserve')
    new = np.zeros(count, parent.dtype)
    xyz, parts = chosen['xyz'], chosen['part_ids']
    quaternion = normals_to_wxyz(chosen['normals'])
    for axis, name in enumerate(('x', 'y', 'z')):
        new[name] = xyz[:, axis]
    for axis in range(4):
        new[f'rot_{axis}'] = quaternion[:, axis]
    for axis, scale in enumerate((args.tangent_factor, args.tangent_factor, args.normal_factor)):
        new[f'scale_{axis}'] = np.log(args.spacing*scale)
    new['opacity'] = np.log(args.opacity/(1-args.opacity))
    # Transfer same-part measured appearance where available; fallback material
    # is explicitly a prior. Two-sided ambient shading is not hidden RGB truth.
    parent_gray = parent['f_dc_0']*.28209479177387814+.5
    gray = chosen['material_gray'].copy()
    color_sources = {}
    for part in np.unique(parts):
        old = (parent_labels == part) & np.isfinite(parent_gray) & (parent_gray > .04) & (parent_gray < .99)
        if np.count_nonzero(old) >= 4:
            gray[parts == part] = np.median(parent_gray[old])
            color_sources[str(int(part))] = 'same_part_parent_gray_median'
        else:
            color_sources[str(int(part))] = 'unobserved_Blender_material_prior'
    light = np.asarray([.35, .65, .68]); light /= np.linalg.norm(light)
    gray *= .8+.2*np.abs(chosen['normals'] @ light)
    gray = np.clip(gray, .06, .9)
    for axis in range(3):
        new[f'f_dc_{axis}'] = (gray-.5)/.28209479177387814
    model, baseline, candidate = (args.output/key for key in ('model', 'baseline', 'candidate'))
    for folder in (model, baseline, candidate):
        folder.mkdir(parents=True, exist_ok=False)
    source = model/'source.ply'
    PlyData([PlyElement.describe(np.concatenate((parent, new)), 'vertex')], text=False, byte_order=ply.byte_order).write(source)
    chosen['metadata_json'] = np.asarray(json.dumps(selection_meta))
    np.savez_compressed(model/'selection.npz', **chosen)
    groups = motion['group_ids'][parts].astype(np.int32)
    extension = model/'completion.npz'
    np.savez_compressed(extension, canonical_xyz=xyz.astype(np.float32), canonical_quaternions=quaternion,
                        part_ids=parts.astype(np.int32), group_ids=groups,
                        metadata_json=np.asarray(json.dumps({'format': 'lego_compact_prior_suffix_v1',
                          'unobserved_geometry_is_prior': True, 'selection_sha256': sha256_file(model/'selection.npz'),
                          'appearance_sources': color_sources, 'elastic_bands_added': False})))
    report = {'format': 'lego_multiview_prior_completion_v1', 'source': parent_report['source'],
        'source_sha256': parent_report['source_sha256'], 'completed_source': str(source), 'completed_sha256': sha256_file(source),
        'parent_source': cache['source'], 'parent_source_sha256': sha256_file(cache['source']),
        'parent_report': str(args.parent_report.resolve()), 'parent_report_sha256': sha256_file(args.parent_report),
        'original_gaussian_count': parent_report['original_gaussian_count'], 'prefix_count': len(parent),
        'added_gaussians': count, 'gaussian_count': len(parent)+count,
        'extension': str(extension), 'extension_sha256': sha256_file(extension),
        'parts': str(args.parts.resolve()), 'parts_sha256': sha256_file(args.parts),
        'parent_correction': str((args.candidate/'correction.json').resolve()),
        'parent_correction_sha256': sha256_file(args.candidate/'correction.json'),
        'mesh_prior': str(args.mesh.resolve()), 'mesh_prior_sha256': sha256_file(args.mesh),
        'parent_baseline': str(args.baseline.resolve()), 'parent_poses_sha256': sha256_file(args.baseline/'poses.npy'),
        'parent_dynamic_indices_sha256': sha256_file(args.baseline/'dynamic_indices.npy'),
        'unobserved_geometry_is_prior': True, 'render_supersample': 1, 'render_downsample_filter': 'area_after_clamp',
        'selection': selection_meta, 'parameters': vars(args) | {'selection': str(args.selection) if args.selection else None},
        'part_additions': {str(int(part)): int(np.sum(parts == part)) for part in np.unique(parts)},
        'scope': 'Original v3 Gaussian prefix and its event-corrected motion preserved exactly. '
                 'Unobserved rigid back/side/top/bottom surfaces are explicitly Blender mesh/material priors, '
                 'not reconstructed monocular evidence. Elastic bands are retained without new surfaces. '
                 'No complete-view or event approval is claimed without separate new audits.'}
    report['parameters'] = {key: str(value) if isinstance(value, Path) else value for key, value in report['parameters'].items()}
    report_path = model/'completion-report.json'; write_json(report_path, report)
    contract = {'format': report['format'], 'report': str(report_path), 'report_sha256': sha256_file(report_path),
                'extension': str(extension), 'extension_sha256': sha256_file(extension)}
    for filename in ('poses.npy', 'dynamic_indices.npy'):
        (baseline/filename).symlink_to((args.baseline/filename).resolve(strict=True))
    cache = copy.deepcopy(cache)
    root_groups = np.flatnonzero(np.asarray(parent_correction['joint_structure']['parents']) == -1)
    cache.update(source=str(source), source_sha256=report['completed_sha256'], gaussian_count=len(parent)+count,
                 rendered_dynamic_count=cache['dynamic_count']+int(np.count_nonzero(~np.isin(groups, root_groups))),
                 multiview_completion=contract, event_supervised=False,
                 binding_scope='Parent baked rows only; appended rigid rows use verified compact group binding.',
                 status='MULTIVIEW_PRIOR_COMPLETION_REQUIRES_NEW_EVENT_AND_VIEW_AUDIT')
    write_json(baseline/'cache.json', cache)
    correction = copy.deepcopy(parent_correction)
    correction['joint_structure']['input_sha256']['baseline_cache_metadata'] = sha256_file(baseline/'cache.json')
    correction.update(multiview_completion=contract, motion_optimization_inherited=True,
                      geometry_prior_completion_performed=True, completion_motion_reoptimized=False,
                      rgb_appearance_parameters_frozen_scope='Correction changes pose only for the new combined PLY; appended surface attributes are separately documented priors.')
    write_json(candidate/'correction.json', correction)
    print(json.dumps({'output': str(args.output), 'added': count, 'part_additions': report['part_additions'],
                      'parent_vertices_untouched': True, 'event_audit': 'pending'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
