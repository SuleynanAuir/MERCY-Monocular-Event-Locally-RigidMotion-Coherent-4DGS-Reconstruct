"""Camera-aligned, all-layer ownership proposals; not event-approved motion.

Geometry mode uses the camera-consistent initial registration for every splat,
including hidden surfaces. Screen mode additionally transfers initial Blender
part IDs through the full Gaussian depth column, subject to a 3-D distance gate.
The latter is an experimental ownership proposal, NOT an occlusion oracle: a
different hidden part can share a foreground pixel. Both require rendered
inspection and the unchanged raw-event audit before use.

No Gaussian positions, colors, opacity or scales are modified or copied.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def initial_registration(metadata):
    if 'initial_registration' not in metadata:
        raise ValueError('Binding metadata has no camera-consistent initial_registration')
    matrix = np.asarray(metadata['initial_registration'], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid initial registration')
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Registration is not affine')
    scale = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if scale.min() <= 0 or not np.allclose(scale, scale[0], rtol=1e-5):
        raise ValueError('Registration must be a nonsingular similarity')
    if np.linalg.det(matrix[:3, :3]) <= 0:
        raise ValueError('Registration must not reflect rigid parts')
    return matrix


def project_mask(xyz, camera, mask):
    """Return front-facing, in-frame part IDs without visibility truncation."""
    height, width = mask.shape
    view = transform(xyz, np.linalg.inv(np.asarray(camera['pose'])))
    depth = -view[:, 2]
    focal = height / (2 * np.tan(np.deg2rad(camera['fovy_degrees']) / 2))
    safe = np.maximum(depth, 1e-8)
    u = focal * view[:, 0] / safe + (width - 1) / 2
    v = -focal * view[:, 1] / safe + (height - 1) / 2
    # Clip before conversion to avoid overflow from points behind the camera.
    px = np.rint(np.clip(u, -1, width)).astype(np.int64)
    py = np.rint(np.clip(v, -1, height)).astype(np.int64)
    inside = (depth > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    labels[inside] = mask[py[inside], px[inside]]
    return labels


def validate_correspondence(labels, nearest, parts, part_meta):
    if not np.array_equal(parts['point_parts'][nearest], labels):
        raise ValueError('Nearest surface indices do not match assigned parts')
    offsets = parts['part_offsets']
    for label in part_meta['deforming_objects']:
        local = nearest[labels == label] - offsets[label]
        count = parts[f'deformed_{label}'].shape[1]
        if np.any(local < 0) or np.any(local >= count):
            raise ValueError(f'Elastic part {label} has an invalid local vertex index')


def attachment_limit(prior_limit, mode, screen_limit):
    """Keep accepted screen ownership dynamic even beyond the old geometry gate."""
    return max(prior_limit, screen_limit) if mode == 'screen' else prior_limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', required=True, type=Path)
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--source-ply', required=True, type=Path)
    parser.add_argument('--camera', required=True, type=Path,
                        help='Inspection/cache JSON containing camera, or the camera object itself')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('geometry', 'screen'))
    parser.add_argument('--part-mask', type=Path)
    parser.add_argument('--screen-max-distance', type=float, default=.065,
                        help='Reject screen relabeling beyond this source-part surface distance. '
                             'May exceed the prior attachment gate; large values can merge unrelated depth layers.')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'screen' and args.part_mask is None:
        parser.error('screen mode requires --part-mask')
    if not np.isfinite(args.screen_max_distance) or args.screen_max_distance <= 0:
        parser.error('--screen-max-distance must be finite and positive')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {key: archive[key] for key in archive.files}
    with np.load(args.binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    prior = json.loads(str(binding['metadata_json']))
    part_meta = json.loads(str(parts['metadata_json']))
    camera_data = json.loads(args.camera.read_text())
    camera = camera_data.get('camera', camera_data)
    matrix = initial_registration(prior)
    camera_pose = np.asarray(camera['pose'], dtype=float)
    if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
        raise ValueError('Invalid camera pose')
    if not 0 < camera['fovy_degrees'] < 180:
        raise ValueError('Invalid camera FOV')
    vertices = PlyData.read(args.source_ply, mmap=True)['vertex'].data
    xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')])
    if len(xyz) != len(binding['labels']) or not np.isfinite(xyz).all():
        raise ValueError('Source Gaussian count/coordinates do not match binding')
    surface = transform(parts['rest_points'], matrix)
    distance, nearest = cKDTree(surface).query(xyz, workers=args.workers)
    labels = parts['point_parts'][nearest].copy()
    original_labels = binding['labels']
    # Keep the prior rubber-band identity, but recompute valid correspondences
    # in the new registration. Never reuse a rigid triangle sample as a band ID.
    for label in part_meta['deforming_objects']:
        rows = np.flatnonzero(original_labels == label)
        points = np.flatnonzero(parts['point_parts'] == label)
        if len(rows):
            dist, near = cKDTree(surface[points]).query(xyz[rows], workers=args.workers)
            labels[rows], distance[rows], nearest[rows] = label, dist, points[near]
    geometry_labels = labels.copy()
    proposed_count = accepted_count = rejected_count = 0
    if args.mode == 'screen':
        with np.load(args.part_mask, allow_pickle=False) as archive:
            mask = archive['part_ids']
            mask_meta = json.loads(str(archive['metadata_json']))
        if mask.ndim != 2 or mask_meta['objects'] != part_meta['objects']:
            raise ValueError('Part-mask shape or object ordering mismatch')
        expected = (camera.get('height', 640), camera.get('width', 640))
        if mask.shape != expected:
            raise ValueError('Part-mask and camera image sizes disagree')
        if abs(mask_meta['fovy_degrees'] - camera['fovy_degrees']) > .1:
            raise ValueError('Part-mask and camera FOV disagree')
        if np.any(mask < -1) or np.any(mask >= len(part_meta['objects'])):
            raise ValueError('Part-mask contains out-of-range IDs')
        proposed = project_mask(xyz, camera, mask)
        eligible = ((proposed >= 0) & (proposed != labels)
                    & ~np.isin(labels, part_meta['deforming_objects'])
                    & ~np.isin(original_labels, part_meta['deforming_objects'])
                    & ~np.isin(proposed, part_meta['deforming_objects']))
        proposed_count = int(eligible.sum())
        for label in np.unique(proposed[eligible]):
            rows = np.flatnonzero(eligible & (proposed == label))
            points = np.flatnonzero(parts['point_parts'] == label)
            dist, near = cKDTree(surface[points]).query(xyz[rows], workers=args.workers)
            keep = dist <= args.screen_max_distance
            accepted_count += int(keep.sum())
            rows = rows[keep]
            labels[rows], distance[rows], nearest[rows] = label, dist[keep], points[near[keep]]
        rejected_count = proposed_count - accepted_count
    validate_correspondence(labels, nearest, parts, part_meta)
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[part_meta['deforming_objects']] = True
    maximum_distance = attachment_limit(prior['max_attachment_distance'], args.mode,
                                        args.screen_max_distance)
    attached = distance <= maximum_distance
    newly_attached_by_distance = attached & (distance > prior['max_attachment_distance'])
    dynamic_indices = np.flatnonzero(moving[labels] & attached).astype(np.int32)
    counts = np.bincount(labels, minlength=len(moving))
    dynamic_counts = np.bincount(labels[dynamic_indices], minlength=len(moving))
    report = {
        'source': str(args.source_ply), 'parts': str(args.parts), 'prior_binding': str(args.binding),
        'method': 'camera-consistent registration; all-depth nearest-surface ownership'
                  + (' plus distance-gated full-column initial part IDs' if args.mode == 'screen' else ''),
        'mode': args.mode, 'registration': matrix.tolist(), 'initial_registration': matrix.tolist(),
        'previous_registration': binding['registration'].tolist(),
        'max_attachment_distance': maximum_distance,
        'prior_max_attachment_distance': prior['max_attachment_distance'],
        'attachment_limit_increased': maximum_distance > prior['max_attachment_distance'],
        'gaussians_beyond_prior_attachment_limit': int(newly_attached_by_distance.sum()),
        'distance_percentiles': dict(zip(['p50', 'p90', 'p95', 'p99'],
                                         np.percentile(distance, [50, 90, 95, 99]).tolist())),
        'unattached': int((~attached).sum()), 'gaussian_count': len(xyz),
        'dynamic_count': len(dynamic_indices), 'static_count': len(xyz) - len(dynamic_indices),
        'part_count': len(moving), 'moving_part_count': int(moving.sum()),
        'changed_gaussians': int(np.sum(labels != original_labels)),
        'geometry_changed_gaussians': int(np.sum(geometry_labels != original_labels)),
        'screen_proposed_reassignments': proposed_count, 'screen_accepted_reassignments': accepted_count,
        'screen_distance_rejections': rejected_count, 'screen_max_distance': args.screen_max_distance,
        'part_mask': str(args.part_mask) if args.part_mask else None,
        'deforming_objects': [part_meta['objects'][i] for i in part_meta['deforming_objects']],
        'part_gaussians': {name: int(dynamic_counts[i]) for i, name in enumerate(part_meta['objects']) if moving[i]},
        'all_part_gaussians': {name: int(counts[i]) for i, name in enumerate(part_meta['objects'])},
        'appearance_and_rest_geometry_unchanged': True, 'event_approved': False,
        'status': 'INITIALIZATION_ONLY_NOT_APPROVED',
        'limitations': 'Source-shape prior, not observed hidden geometry. Full-column IDs may confuse '
                      'different occluding parts; a large screen distance gate can move unrelated hidden '
                      'surfaces or the static background with a foreground part. The global attachment gate '
                      'is increased to include accepted screen assignments, which can also attach geometric '
                      'fallback rows beyond the previous limit. Rubber-band IDs retained; new local correspondences validated. '
                      'Must inspect full motion and refine/audit raw events; no event approval is implied.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, registration=matrix, labels=labels.astype(np.int16),
                        nearest_vertex=nearest.astype(np.int32), distance=distance.astype(np.float32),
                        dynamic_indices=dynamic_indices, metadata_json=np.asarray(json.dumps(report)))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('mode', 'changed_gaussians', 'dynamic_count',
                     'screen_accepted_reassignments', 'screen_distance_rejections', 'unattached')}), flush=True)


if __name__ == '__main__':
    main()
