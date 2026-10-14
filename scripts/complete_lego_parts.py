"""Append small rigidly bound Gaussians on repeatedly observed missing surfaces.

The existing repaired PLY is frozen byte-for-byte as the vertex prefix. New
geometry is explicitly prior-assisted completion, not recovered hidden truth.
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

from utils.rigid_checkpoint import sha256_file


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def aggregate_supported_cells(xyz, labels, frames, gray, radii, normals, voxel_size, minimum_frames):
    if voxel_size <= 0 or minimum_frames < 1:
        raise ValueError('Invalid aggregation thresholds')
    cells = np.floor(xyz / voxel_size).astype(np.int64)
    keys = np.column_stack((labels, cells))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    starts = np.concatenate(([0], np.flatnonzero(np.diff(inverse[order])) + 1, [len(order)]))
    records = []
    for group, key in enumerate(unique):
        rows = order[starts[group]:starts[group+1]]
        observed_frames = np.unique(frames[rows])
        if len(observed_frames) < minimum_frames:
            continue
        # Choose an actual ray/triangle hit nearest the cell median. Averaging
        # surface positions could place a new point across an edge or a hole.
        center = np.median(xyz[rows], axis=0)
        selected = rows[np.argmin(np.linalg.norm(xyz[rows] - center, axis=-1))]
        per_frame_gray = [np.median(gray[rows][frames[rows] == frame]) for frame in observed_frames]
        records.append({'part_id': int(key[0]), 'xyz': xyz[selected], 'normal': normals[selected],
                        'selected_observation_index': int(selected),
                        'gray': float(np.median(per_frame_gray)), 'radius': float(np.median(radii[rows])),
                        'support_frames': observed_frames.tolist(), 'support_observations': len(rows)})
    return records, len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ply', type=Path, required=True)
    parser.add_argument('--original-source', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--parts', type=Path, required=True)
    parser.add_argument('--surfaces', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--minimum-frames', type=int, default=3)
    parser.add_argument('--voxel-pixels', type=float, default=.5)
    parser.add_argument('--radius-pixels', type=float, default=.65)
    parser.add_argument('--opacity', type=float, default=.85)
    parser.add_argument('--max-added', type=int, default=30000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (1 <= args.minimum_frames <= 400 and 0 < args.voxel_pixels <= 2 and 0 < args.radius_pixels <= 2
            and 0 < args.opacity < 1 and 0 < args.max_added <= 30000):
        raise ValueError('Invalid or excessive completion parameters')
    metadata = json.loads((args.baseline / 'cache.json').read_text())
    camera = metadata['camera']; camera_inverse = np.linalg.inv(camera['pose'])
    with np.load(args.binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    binding_meta = json.loads(str(binding['metadata_json']))
    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {key: archive[key] for key in archive.files}
    parts_meta = json.loads(str(parts['metadata_json']))
    with np.load(args.surfaces, allow_pickle=False) as archive:
        surfaces = {key: archive[key] for key in archive.files}
    surface_meta = json.loads(str(surfaces['metadata_json']))
    if (surface_meta['format'] != 'lego_observed_completion_surfaces_v1'
            or surface_meta['parts_sha256'] != sha256_file(args.parts)
            or surface_meta['objects'] != parts_meta['objects']):
        raise ValueError('Completion surface provenance differs')
    if not len(surfaces['part_id']):
        raise ValueError('No observed missing surfaces')
    if np.isin(surfaces['part_id'], parts_meta['deforming_objects']).any():
        raise ValueError('Completion cannot attach a new splat to an elastic band')
    registration = binding['registration']
    xyz = transform(surfaces['canonical_source_xyz'].astype(np.float64), registration)
    normals = surfaces['canonical_source_normal'] @ registration[:3, :3].T
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    observation_world = transform(surfaces['source_world'].astype(np.float64), registration)
    observation_z = -transform(observation_world, camera_inverse)[:, 2]
    focal = 320 / np.tan(np.deg2rad(camera['fovy_degrees']) / 2)
    radii = args.radius_pixels * observation_z / focal
    voxel_size = args.voxel_pixels * float(np.median(observation_z)) / focal
    if not np.isfinite(xyz).all() or not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError('Invalid surface geometry')
    records, candidate_cells = aggregate_supported_cells(
        xyz, surfaces['part_id'], surfaces['frame_index'], surfaces['gray'], radii, normals,
        voxel_size, args.minimum_frames)
    records.sort(key=lambda row: (-len(row['support_frames']), -row['support_observations'], row['part_id']))
    supported_count = len(records); records = records[:args.max_added]
    if not records:
        raise ValueError(f'No cells met {args.minimum_frames} distinct-frame support; candidate cells={candidate_cells}')
    source = PlyData.read(args.source_ply)
    original_vertices = source['vertex'].data
    original_count = len(original_vertices)
    if len(binding['labels']) != original_count or metadata['gaussian_count'] != original_count:
        raise ValueError('Source, binding, and baseline counts differ')
    prefix_hash = hashlib.sha256(np.ascontiguousarray(original_vertices).tobytes()).hexdigest()
    new_vertices = np.zeros(len(records), dtype=original_vertices.dtype)
    required = ['x', 'y', 'z', 'opacity', *[f'f_dc_{i}' for i in range(3)],
                *[f'scale_{i}' for i in range(3)], *[f'rot_{i}' for i in range(4)]]
    if any(field not in new_vertices.dtype.names for field in required):
        raise ValueError('Unsupported Gaussian PLY fields')
    new_xyz = np.asarray([row['xyz'] for row in records], np.float32)
    new_labels = np.asarray([row['part_id'] for row in records], dtype=binding['labels'].dtype)
    for axis, field in enumerate(('x', 'y', 'z')):
        new_vertices[field] = new_xyz[:, axis]
    for i in range(3):
        new_vertices[f'f_dc_{i}'] = (np.asarray([row['gray'] for row in records]) - .5) / .28209479177387814
        new_vertices[f'scale_{i}'] = np.log([row['radius'] for row in records])
    new_vertices['opacity'] = np.log(args.opacity / (1 - args.opacity))
    new_vertices['rot_0'] = 1
    if any(not np.isfinite(new_vertices[field]).all() for field in new_vertices.dtype.names):
        raise ValueError('All appended Gaussian attributes must be finite')
    combined = np.concatenate((original_vertices, new_vertices))
    if hashlib.sha256(np.ascontiguousarray(combined[:original_count]).tobytes()).hexdigest() != prefix_hash:
        raise ValueError('Completion modified a frozen original row')
    surface = transform(parts['rest_points'], registration)
    nearest = np.empty(len(records), dtype=binding['nearest_vertex'].dtype)
    distance = np.empty(len(records), np.float32)
    for part in np.unique(new_labels):
        rows = np.flatnonzero(new_labels == part)
        points = np.flatnonzero(parts['point_parts'] == part)
        distance[rows], near = cKDTree(surface[points]).query(new_xyz[rows], workers=4)
        nearest[rows] = points[near]
    if (distance > binding_meta['max_attachment_distance']).any():
        raise ValueError('Completion exceeds the existing part attachment tolerance')
    args.output_dir.mkdir(parents=True)
    target = args.output_dir / 'source.ply'
    output = PlyData([PlyElement.describe(combined, 'vertex')], text=False,
                     byte_order=source.byte_order, comments=list(source.comments), obj_info=list(source.obj_info))
    output.write(target)
    verified = PlyData.read(target)['vertex'].data
    if (verified.dtype != original_vertices.dtype
            or hashlib.sha256(np.ascontiguousarray(verified[:original_count]).tobytes()).hexdigest() != prefix_hash):
        raise ValueError('Written PLY did not preserve the frozen prefix')
    report = {
        'format': 'lego_part_completion_v1', 'source': str(args.original_source),
        'source_sha256': sha256_file(args.original_source),
        'base_repaired_source': str(args.source_ply), 'base_repaired_sha256': sha256_file(args.source_ply),
        'completed_source': str(target), 'completed_sha256': sha256_file(target),
        'original_gaussian_count': original_count, 'gaussian_count': len(combined), 'added_gaussians': len(records),
        'frozen_prefix_vertex_sha256': prefix_hash, 'surfaces': str(args.surfaces),
        'surfaces_sha256': sha256_file(args.surfaces), 'binding': str(args.binding), 'parts': str(args.parts),
        'candidate_cells': candidate_cells, 'supported_cells': supported_count,
        'parameters': {'minimum_distinct_frames': args.minimum_frames, 'voxel_pixels': args.voxel_pixels,
                       'voxel_scene_units': voxel_size, 'radius_pixels': args.radius_pixels,
                       'isotropic_gaussian': True, 'opacity': args.opacity, 'max_added': args.max_added},
        'part_additions': {parts_meta['objects'][int(part)]: int((new_labels == part).sum()) for part in np.unique(new_labels)},
        'part_addition_ids': {str(int(part)): int((new_labels == part).sum()) for part in np.unique(new_labels)},
        'support': [{'part_id': row['part_id'], 'frames': row['support_frames'],
                     'observations': row['support_observations'], 'gray_median': row['gray']} for row in records],
        'original_rows_frozen': True, 'all_new_attributes_finite': True, 'source_background_and_holes_excluded': True,
        'event_approved': False, 'status': 'PRIOR_ASSISTED_COMPLETION_REQUIRES_EVENT_REFINEMENT_AND_AUDIT',
        'limitations': 'New Gaussians are explicit Blender-geometry-assisted completion of repeatedly observed '
                      'reference-view bright missing surfaces. Color is median observed grayscale, not calibrated albedo. '
                      'Rigid source-part ownership prevents internal tearing of each added part; occlusion, appearance, '
                      'and event consistency still require full rerendering and audit. This does not prove hidden 3D truth.'}
    binding['labels'] = np.concatenate((binding['labels'], new_labels))
    binding['nearest_vertex'] = np.concatenate((binding['nearest_vertex'], nearest))
    binding['distance'] = np.concatenate((binding['distance'], distance))
    binding['dynamic_indices'] = np.concatenate((binding['dynamic_indices'],
                                                  np.arange(original_count, len(combined), dtype=binding['dynamic_indices'].dtype)))
    binding_meta.update(source=str(target), prior_binding=str(args.binding), gaussian_count=len(combined),
                        method='Append-only, repeatedly observed missing rigid-surface completion; all repaired prefix rows frozen',
                        dynamic_count=len(binding['dynamic_indices']), static_count=len(combined)-len(binding['dynamic_indices']),
                        geometry_completion_report=str(args.output_dir / 'completion-report.json'),
                        geometry_completion_format=report['format'], added_gaussians=len(records),
                        event_approved=False, status=report['status'], limitations=report['limitations'])
    for key in ('part_gaussians', 'all_part_gaussians'):
        counts = binding_meta.get(key, {})
        for part, count in report['part_additions'].items():
            counts[part] = counts.get(part, 0) + count
        binding_meta[key] = counts
    binding['metadata_json'] = np.asarray(json.dumps(binding_meta))
    np.savez_compressed(args.output_dir / 'initial-binding.npz', **binding)
    (args.output_dir / 'initial-binding.json').write_text(json.dumps(binding_meta, indent=2)+'\n')
    (args.output_dir / 'completion-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('format', 'original_gaussian_count', 'gaussian_count', 'added_gaussians',
                                                 'candidate_cells', 'supported_cells', 'part_addition_ids')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
