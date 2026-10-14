"""Build a NEW prior-assisted rigid-surface Gaussian cloud; preserve static/band rows.

Original source files are untouched. Retired moving rigid rows are replaced in
the new artifact by actual mesh hits with measured RGB, not by UI imagery.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from scripts.complete_lego_parts import aggregate_supported_cells, transform
from utils.rigid_checkpoint import sha256_file


def retained_rows(labels, dynamic_indices, moving_rigid):
    active = np.zeros(len(labels), bool)
    active[dynamic_indices] = True
    retired = active & moving_rigid[labels]
    return np.flatnonzero(~retired).astype(np.int32), active, retired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-ply', 'original-source', 'binding', 'parts', 'surfaces', 'baseline', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--minimum-frames', type=int, default=2)
    p.add_argument('--voxel-pixels', type=float, default=.65)
    p.add_argument('--radius-pixels', type=float, default=.6)
    p.add_argument('--opacity', type=float, default=.95)
    p.add_argument('--max-seeds', type=int, default=200000)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (2 <= args.minimum_frames <= 400 and 0 < args.voxel_pixels <= 2
            and 0 < args.radius_pixels <= 2 and 0 < args.opacity < 1 and 0 < args.max_seeds <= 200000):
        raise ValueError('Invalid bounded surface-rebuild parameters')
    with np.load(args.binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    binding_meta = json.loads(str(binding['metadata_json']))
    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {key: archive[key] for key in archive.files}
    part_meta = json.loads(str(parts['metadata_json']))
    with np.load(args.surfaces, allow_pickle=False) as archive:
        evidence = {key: archive[key] for key in archive.files}
    evidence_meta = json.loads(str(evidence['metadata_json']))
    if (evidence_meta.get('format') != 'lego_observed_completion_surfaces_v1'
            or evidence_meta.get('all_visible_rigid') is not True
            or evidence_meta['parts_sha256'] != sha256_file(args.parts)
            or evidence_meta['objects'] != part_meta['objects']):
        raise ValueError('Need verified all-visible rigid mesh observations')
    camera = json.loads((args.baseline/'cache.json').read_text())['camera']
    camera_inverse = np.linalg.inv(camera['pose'])
    registration = binding['registration']
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving_rigid = np.max(np.abs(delta-np.eye(4)), axis=(0,2,3)) > 1e-4
    moving_rigid[part_meta['deforming_objects']] = False
    if not len(evidence['part_id']) or not moving_rigid[evidence['part_id']].all():
        raise ValueError('Seed evidence contains a static/elastic part or is empty')
    xyz = transform(evidence['canonical_source_xyz'].astype(np.float64), registration)
    normals = evidence['canonical_source_normal'] @ registration[:3,:3].T
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    observation_world = transform(evidence['source_world'].astype(np.float64), registration)
    observation_z = -transform(observation_world, camera_inverse)[:,2]
    focal = 320 / np.tan(np.deg2rad(camera['fovy_degrees'])/2)
    radii = args.radius_pixels * observation_z / focal
    voxel_size = args.voxel_pixels * float(np.median(observation_z)) / focal
    if not np.isfinite(xyz).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Invalid seed surface geometry')
    print('Aggregating', len(xyz), 'visible surface observations', flush=True)
    records, cells = aggregate_supported_cells(xyz, evidence['part_id'], evidence['frame_index'],
                                               evidence['gray'], radii, normals, voxel_size, args.minimum_frames)
    if not records or len(records) > args.max_seeds:
        raise ValueError(f'Supported seeds={len(records)}, cap={args.max_seeds}; no silent truncation. '
                         'Choose a reviewed larger voxel or permitted cap.')
    base_ply = PlyData.read(args.source_ply)
    base = base_ply['vertex'].data
    original = PlyData.read(args.original_source)['vertex'].data
    if len(base) != len(original) or len(base) != len(binding['labels']) or base.dtype != original.dtype:
        raise ValueError('Original/base/binding shape or vertex format differs')
    retained, active, retired = retained_rows(binding['labels'], binding['dynamic_indices'], moving_rigid)
    frozen = np.ascontiguousarray(base[retained])
    frozen_hash = hashlib.sha256(frozen.tobytes()).hexdigest()
    if hashlib.sha256(np.ascontiguousarray(original[retained]).tobytes()).hexdigest() != frozen_hash:
        raise ValueError('Retained static/band original rows differ from the repaired base')
    new = np.zeros(len(records), dtype=base.dtype)
    new_xyz = np.asarray([row['xyz'] for row in records], np.float32)
    new_labels = np.asarray([row['part_id'] for row in records], dtype=binding['labels'].dtype)
    for axis, field in enumerate(('x','y','z')):
        new[field] = new_xyz[:,axis]
    for axis in range(3):
        new[f'f_dc_{axis}'] = (np.asarray([row['gray'] for row in records])-.5) / .28209479177387814
        new[f'scale_{axis}'] = np.log([row['radius'] for row in records])
    new['opacity'] = np.log(args.opacity / (1-args.opacity)); new['rot_0'] = 1
    if any(not np.isfinite(new[name]).all() for name in new.dtype.names):
        raise ValueError('Every newly seeded Gaussian attribute must be finite')
    combined = np.concatenate((frozen, new))
    surface = transform(parts['rest_points'], registration)
    new_nearest = np.empty(len(new), dtype=binding['nearest_vertex'].dtype)
    new_distance = np.empty(len(new), np.float32)
    for part in np.unique(new_labels):
        rows = np.flatnonzero(new_labels == part); points = np.flatnonzero(parts['point_parts'] == part)
        new_distance[rows], near = cKDTree(surface[points]).query(new_xyz[rows], workers=4)
        new_nearest[rows] = points[near]
    if (new_distance > binding_meta['max_attachment_distance']).any():
        raise ValueError('New seeds exceed the existing source-surface attachment tolerance')
    args.output_dir.mkdir(parents=True)
    target = args.output_dir/'source.ply'
    PlyData([PlyElement.describe(combined, 'vertex')], text=False, byte_order=base_ply.byte_order,
            comments=list(base_ply.comments), obj_info=list(base_ply.obj_info)).write(target)
    written = PlyData.read(target)['vertex'].data
    if (written.dtype != base.dtype or len(written) != len(combined)
            or hashlib.sha256(np.ascontiguousarray(written[:len(retained)]).tobytes()).hexdigest() != frozen_hash):
        raise ValueError('Rebuilt PLY failed retained-prefix byte validation')
    retained_path = args.output_dir/'retained_original_indices.npy'
    np.save(retained_path, retained, allow_pickle=False)
    support_lengths = np.asarray([len(row['support_frames']) for row in records], np.int32)
    support_offsets = np.concatenate(([0], np.cumsum(support_lengths)))
    evidence_path = args.output_dir/'seed-evidence.npz'
    np.savez_compressed(evidence_path,
                        selected_surface_observation=np.asarray([row['selected_observation_index'] for row in records], np.int32),
                        part_ids=new_labels, support_offsets=support_offsets,
                        support_frames=np.asarray([frame for row in records for frame in row['support_frames']], np.int32),
                        gray=np.asarray([row['gray'] for row in records], np.float32),
                        radius=np.asarray([row['radius'] for row in records], np.float32))
    report = {
        'format': 'lego_prior_surface_rebuild_v1', 'source': str(args.original_source),
        'source_sha256': sha256_file(args.original_source),
        'rebuilt_source': str(target), 'rebuilt_sha256': sha256_file(target),
        'base_repaired_source': str(args.source_ply), 'base_repaired_sha256': sha256_file(args.source_ply),
        'original_gaussian_count': len(base), 'gaussian_count': len(combined),
        'retained_gaussian_count': len(retained), 'retired_gaussian_count': int(retired.sum()),
        'rebuilt_gaussians': len(new), 'retained_indices_path': str(retained_path),
        'retained_indices_sha256': sha256_file(retained_path), 'frozen_retained_vertex_sha256': frozen_hash,
        'surfaces': str(args.surfaces), 'surfaces_sha256': sha256_file(args.surfaces),
        'seed_evidence': str(evidence_path), 'seed_evidence_sha256': sha256_file(evidence_path),
        'parts': str(args.parts), 'parts_sha256': sha256_file(args.parts), 'binding': str(args.binding),
        'candidate_cells': cells, 'supported_cells': len(records), 'seed_cap_truncation': False,
        'parameters': {'minimum_distinct_times': args.minimum_frames, 'voxel_pixels': args.voxel_pixels,
                       'voxel_scene_units': voxel_size, 'radius_pixels': args.radius_pixels,
                       'opacity': args.opacity, 'isotropic_gaussians': True, 'max_seeds': args.max_seeds,
                       'reference_threshold': evidence_meta['reference_threshold'],
                       'min_facing_cosine': evidence_meta['min_facing_cosine'],
                       'sampled_frames': evidence_meta['sampled_frames'], 'erosion_pixels': evidence_meta['erosion_pixels']},
        'seed_part_counts': {str(int(part)): int((new_labels==part).sum()) for part in np.unique(new_labels)},
        'retired_part_counts': {str(int(part)): int(np.sum(retired & (binding['labels']==part)))
                                for part in np.unique(binding['labels'][retired])},
        'seed_part_names': {str(int(part)): part_meta['objects'][int(part)] for part in np.unique(new_labels)},
        'retired_parts_without_seeds': [
            {'part_id': int(part), 'name': part_meta['objects'][int(part)],
             'retired_gaussians': int(np.sum(retired & (binding['labels']==part))),
             'accepted_surface_observations': int(np.sum(evidence['part_id']==part))}
            for part in np.setdiff1d(np.unique(binding['labels'][retired]), np.unique(new_labels))],
        'support_time_count_histogram': {str(int(count)): int(np.sum(support_lengths==count)) for count in np.unique(support_lengths)},
        'radius_scene_units': {'minimum': float(np.exp(new['scale_0']).min()),
                               'median': float(np.median(np.exp(new['scale_0']))),
                               'maximum': float(np.exp(new['scale_0']).max())},
        'retained_static_and_bands_exact': True, 'original_source_untouched': True,
        'new_geometry_from_actual_visible_mesh_hits': True, 'event_approved': False,
        'status': 'PRIOR_ASSISTED_SURFACE_REBUILD_REQUIRES_EVENT_REFINEMENT_AND_AUDIT',
        'limitations': 'A new, explicitly geometry-prior-assisted Gaussian reconstruction, not a refinement '
                      'that preserves the old moving rows. Static and elastic rows are retained byte-for-byte; '
                      'retired original rigid rows remain recoverable in the untouched input files. New rigid '
                      'surfaces use observed grayscale median, not calibrated reflectance. Distinct support '
                      'times include repeated cycles and are not independent viewpoints or hidden-depth truth. '
                      'Unseen or very dark surfaces are not invented. Full event and visual audits remain required.'}
    old_dynamic = np.flatnonzero(active[retained]).astype(binding['dynamic_indices'].dtype)
    binding['labels'] = np.concatenate((binding['labels'][retained], new_labels))
    binding['nearest_vertex'] = np.concatenate((binding['nearest_vertex'][retained], new_nearest))
    binding['distance'] = np.concatenate((binding['distance'][retained], new_distance))
    binding['dynamic_indices'] = np.concatenate((old_dynamic,
        np.arange(len(retained), len(combined), dtype=binding['dynamic_indices'].dtype)))
    binding_meta.update(source=str(target), prior_binding=str(args.binding), gaussian_count=len(combined),
                        method='Prior-assisted moving rigid surface reinitialization; static/elastic original rows frozen',
                        dynamic_count=len(binding['dynamic_indices']), static_count=len(combined)-len(binding['dynamic_indices']),
                        geometry_rebuild_report=str(args.output_dir/'rebuild-report.json'), geometry_rebuild_format=report['format'],
                        retired_gaussians=int(retired.sum()), rebuilt_gaussians=len(new),
                        event_approved=False, status=report['status'], limitations=report['limitations'])
    labels, dynamic = binding['labels'], binding['dynamic_indices']
    binding_meta['all_part_gaussians'] = {name: int(np.sum(labels==part)) for part,name in enumerate(part_meta['objects'])}
    binding_meta['part_gaussians'] = {name: int(np.sum(labels[dynamic]==part)) for part,name in enumerate(part_meta['objects'])
                                    if np.any(labels[dynamic]==part)}
    binding['metadata_json'] = np.asarray(json.dumps(binding_meta))
    np.savez_compressed(args.output_dir/'initial-binding.npz', **binding)
    (args.output_dir/'initial-binding.json').write_text(json.dumps(binding_meta, indent=2)+'\n')
    (args.output_dir/'rebuild-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('format','original_gaussian_count','gaussian_count','retained_gaussian_count',
                                                 'retired_gaussian_count','rebuilt_gaussians','candidate_cells','seed_part_counts')}, indent=2), flush=True)
    if report['retired_parts_without_seeds']:
        print(json.dumps({'coverage_warning_parts_without_seeds': report['retired_parts_without_seeds']}, indent=2), flush=True)


if __name__=='__main__':
    main()
