"""Experimental same-motion-group ray-depth geometry repair from Blender prior.

Projects moving rigid Gaussian centres onto the permitted initial source mesh's
visible depth, without changing their individual initial pixel footprints. All
depth layers in a matching pixel column are eligible: this may collapse hidden
surfaces onto the visible sheet and is NOT measured hidden geometry. Static and
elastic rows, colors, opacity and rotations remain unchanged. Audit required.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.rebind_lego_coherent import transform, validate_correspondence
from scripts.repair_lego_part_depth import sha256_file, field_hash, projection_footprint, residual_stats


def sample_part_depth(uv, part_ids, depth):
    """Bilinear z only when all four samples share the nearest pixel's part ID."""
    h, w = part_ids.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    x = np.clip(uv[:, 0], 0, w - 1); y = np.clip(uv[:, 1], 0, h - 1)
    px = np.rint(x).astype(int); py = np.rint(y).astype(int)
    labels = part_ids[py, px].astype(np.int32)
    labels[~inside] = -1
    ix = np.minimum(x.astype(int), w - 2); iy = np.minimum(y.astype(int), h - 2)
    dx = x - ix; dy = y - iy
    values = np.stack([depth[iy, ix], depth[iy, ix + 1], depth[iy + 1, ix], depth[iy + 1, ix + 1]], axis=1)
    neighbor_ids = np.stack([part_ids[iy, ix], part_ids[iy, ix + 1], part_ids[iy + 1, ix], part_ids[iy + 1, ix + 1]], axis=1)
    valid = (inside & (labels >= 0) & np.all(neighbor_ids == labels[:, None], axis=1)
             & np.all(np.isfinite(values) & (values > 0), axis=1))
    sampled = ((1 - dx) * (1 - dy) * values[:, 0] + dx * (1 - dy) * values[:, 1]
               + (1 - dx) * dy * values[:, 2] + dx * dy * values[:, 3])
    return labels, sampled, valid


def agreeing_motion_rows(assigned, projected, groups, moving, elastic, dynamic):
    safe = np.clip(projected, 0, len(groups) - 1)
    return (dynamic & moving[assigned] & ~np.isin(assigned, elastic)
            & (projected >= 0) & moving[safe] & ~np.isin(safe, elastic)
            & (groups[assigned] == groups[safe]))


def translated_column_depth(original_depth, prior_front, gaussian_front):
    """Translate a depth column without collapsing its front/back separation."""
    return original_depth + prior_front - gaussian_front


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-ply', 'binding', 'parts', 'camera', 'part-mask', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--fallback-report', type=Path, help='Optional compatible per-part radial repair report; no-evidence/edge rows use its part scale')
    p.add_argument('--depth-mode', choices=('collapse', 'preserve-thickness'), default='collapse')
    p.add_argument('--depth-report-npz', type=Path, help='Required for preserve-thickness: original GS rasterized depth and alpha')
    p.add_argument('--minimum-ratio', type=float, default=.5)
    p.add_argument('--maximum-ratio', type=float, default=2.)
    p.add_argument('--motion-cluster-distance', type=float, default=.002)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.depth_mode == 'preserve-thickness' and args.depth_report_npz is None:
        p.error('preserve-thickness requires --depth-report-npz')
    if not 0 < args.minimum_ratio <= 1 <= args.maximum_ratio or not np.isfinite([args.minimum_ratio, args.maximum_ratio]).all():
        p.error('Invalid depth-ratio bounds')
    if not np.isfinite(args.motion_cluster_distance) or args.motion_cluster_distance <= 0:
        p.error('Invalid motion cluster threshold')
    from scipy.spatial import cKDTree
    from scipy.cluster.hierarchy import fclusterdata
    from plyfile import PlyData
    with np.load(args.parts, allow_pickle=False) as data:
        parts = {k: data[k] for k in data.files}
    with np.load(args.binding, allow_pickle=False) as data:
        binding = {k: data[k] for k in data.files}
    with np.load(args.part_mask, allow_pickle=False) as data:
        mask = data['part_ids']; ray_distance = data['ray_distance']
        mask_meta = json.loads(str(data['metadata_json']))
    part_meta = json.loads(str(parts['metadata_json']))
    prior = json.loads(str(binding['metadata_json']))
    record = json.loads(args.camera.read_text()); camera = record.get('camera', record)
    pose = np.asarray(camera['pose'], dtype=float)
    if mask_meta['objects'] != part_meta['objects'] or mask.shape != ray_distance.shape:
        raise ValueError('Part-mask identity/shape mismatch')
    if mask.shape != (camera.get('height', 640), camera.get('width', 640)) or abs(mask_meta['fovy_degrees'] - camera['fovy_degrees']) > .001:
        raise ValueError('Camera and source mask disagree')
    if np.any(mask < -1) or np.any(mask >= len(part_meta['objects'])):
        raise ValueError('Invalid source part IDs')
    registration = binding['registration']
    scale = float(np.cbrt(np.linalg.det(registration[:3, :3])))
    expected = pose @ np.diag([scale, scale, scale, 1.]) @ np.linalg.inv(part_meta['camera_pose'])
    if not np.allclose(registration, expected, atol=1e-5, rtol=1e-5):
        raise ValueError('A camera-consistent binding registration is required')
    h, w = mask.shape
    focal = h / (2 * np.tan(np.deg2rad(camera['fovy_degrees']) / 2))
    yy, xx = np.mgrid[:h, :w]
    prior_z = ray_distance * scale / np.sqrt(1 + ((xx - (w - 1) / 2) / focal) ** 2 + ((yy - (h - 1) / 2) / focal) ** 2)
    source_hash = sha256_file(args.source_ply)
    # Copy-on-write mmap never persists changes to the original PLY. The older
    # plyfile non-mmap reader loops over every scalar and is prohibitively slow.
    ply = PlyData.read(args.source_ply, mmap='c'); vertices = ply['vertex'].data
    labels = binding['labels'].copy(); original_labels = labels.copy()
    if len(labels) != len(vertices):
        raise ValueError('Binding and source counts disagree')
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    log_scales = np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64)
    if not np.isfinite(xyz).all() or not np.isfinite(log_scales).all():
        raise ValueError('Invalid input geometry')
    protected = [k for k in vertices.dtype.names if k not in ('x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2')]
    protected_before = field_hash(vertices, protected)
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    frames = np.unique(np.linspace(0, len(delta) - 1, min(17, len(delta))).astype(int))
    features = delta[frames, :, :3, :].transpose(1, 0, 2, 3).reshape(len(matrices[0]), -1)
    groups = fclusterdata(features, args.motion_cluster_distance, criterion='distance', method='complete') - 1
    elastic = part_meta['deforming_objects']
    dynamic = np.zeros(len(xyz), bool); dynamic[binding['dynamic_indices']] = True
    movable = dynamic & moving[labels] & ~np.isin(labels, elastic)
    view = transform(xyz, np.linalg.inv(pose)); original_depth = -view[:, 2]
    uv = focal * view[:, :2] / np.maximum(original_depth[:, None], 1e-8)
    uv[:, 1] *= -1; uv += np.array([(w - 1) / 2, (h - 1) / 2])
    proposed, target_z, interior = sample_part_depth(uv, mask, prior_z)
    agreement = agreeing_motion_rows(labels, proposed, groups, moving, elastic, dynamic)
    desired_z = target_z.copy()
    if args.depth_mode == 'preserve-thickness':
        with np.load(args.depth_report_npz, allow_pickle=False) as data:
            gs_front = data['gs_depth']; gs_alpha = data['gs_alpha']; gs_valid = data['valid']
        if any(value.shape != mask.shape for value in (gs_front, gs_alpha, gs_valid)):
            raise ValueError('Original depth diagnostic shape mismatch')
        depth_json = args.depth_report_npz.with_suffix('.json')
        if depth_json.exists():
            diagnostic = json.loads(depth_json.read_text())
            if Path(diagnostic['source']).resolve() != args.source_ply.resolve():
                raise ValueError('Depth diagnostic uses another source')
        gs_front = np.where(gs_valid & (gs_alpha > .8), gs_front, np.nan)
        _, sampled_front, front_valid = sample_part_depth(uv, mask, gs_front)
        desired_z = translated_column_depth(original_depth, target_z, sampled_front)
        interior &= front_valid
    raw_ratio = desired_z / np.maximum(original_depth, 1e-8)
    ratio_valid = np.isfinite(raw_ratio) & (raw_ratio >= args.minimum_ratio) & (raw_ratio <= args.maximum_ratio)
    exact = agreement & interior & (original_depth > 0) & ratio_valid
    radial_scale = np.ones(len(xyz), np.float64)
    fallback = None
    if args.fallback_report:
        fallback = json.loads(args.fallback_report.read_text())
        if fallback.get('format') != 'lego_part_depth_repair_v1' or fallback['source_sha256'] != source_hash or fallback['gaussian_count'] != len(xyz):
            raise ValueError('Fallback radial report does not match the original source')
        for part_record in fallback['part_repairs']:
            part = part_record['part']; factor = part_record['scale']
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError('Invalid fallback scale')
            radial_scale[movable & (labels == part)] = factor
    radial_scale[exact] = raw_ratio[exact]
    labels[exact] = proposed[exact]
    changed = movable & (radial_scale != 1)
    origin = pose[:3, 3]
    xyz_repaired = origin + radial_scale[:, None] * (xyz - origin)
    new_scales = log_scales + np.log(radial_scale[:, None])
    for i, name in enumerate(('x', 'y', 'z')):
        vertices[name][changed] = xyz_repaired[changed, i]
    for i in range(3):
        vertices[f'scale_{i}'][changed] = new_scales[changed, i]
    final_xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    final_scales = np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64)
    if not (np.array_equal(final_xyz[~movable], xyz[~movable]) and np.array_equal(final_scales[~movable], log_scales[~movable])
            and np.array_equal(labels[~movable], original_labels[~movable])):
        raise AssertionError('Static/elastic/unattached rows changed')
    protected_after = field_hash(vertices, protected)
    if protected_before != protected_after:
        raise AssertionError('Protected attributes changed')
    old_front, old_uv, old_foot = projection_footprint(xyz, log_scales, camera)
    new_front, new_uv, new_foot = projection_footprint(final_xyz, final_scales, camera)
    if not np.array_equal(old_front, new_front):
        raise AssertionError('Camera-plane crossing')
    nearest = binding['nearest_vertex'].copy(); distance = binding['distance'].copy()
    surface = transform(parts['rest_points'], registration)
    for part in np.unique(labels[movable]):
        rows = np.flatnonzero(movable & (labels == part))
        points = np.flatnonzero(parts['point_parts'] == part)
        dist, near = cKDTree(surface[points]).query(final_xyz[rows], workers=args.workers)
        nearest[rows] = points[near]; distance[rows] = dist
    validate_correspondence(labels, nearest, parts, part_meta)
    final_depth = -transform(final_xyz, np.linalg.inv(pose))[:, 2]
    part_rows = []
    for part in np.unique(original_labels[movable]):
        rows = movable & (original_labels == part)
        hit = rows & exact
        part_rows.append({'part': int(part), 'name': part_meta['objects'][part], 'gaussian_count': int(rows.sum()),
                          'ray_repaired_rows': int(hit.sum()), 'fallback_changed_rows': int((rows & ~exact & changed).sum()),
                          'ratio_percentiles': np.percentile(radial_scale[rows], [0, 50, 100]).tolist(),
                          'ray_depth_error_before': residual_stats((original_depth - desired_z)[hit]),
                          'ray_depth_error_after': residual_stats((final_depth - desired_z)[hit])})
    report = {'format': 'lego_part_depth_repair_v1', 'repair_method': ('same-motion-group source-visible ray-depth projection' if args.depth_mode == 'collapse'
                        else 'same-motion-group source-prior front-depth offset preserving pixel-column thickness'),
              'source': str(args.source_ply), 'source_sha256': source_hash, 'gaussian_count': len(vertices),
              'repaired_source': str(args.output_dir / 'source.ply'), 'binding': str(args.binding),
              'parts': str(args.parts), 'part_mask': str(args.part_mask), 'camera': camera,
              'fallback_report': str(args.fallback_report) if args.fallback_report else None,
              'depth_mode': args.depth_mode, 'depth_report_npz': str(args.depth_report_npz) if args.depth_report_npz else None,
              'parameters': {'minimum_ratio': args.minimum_ratio, 'maximum_ratio': args.maximum_ratio,
                             'motion_cluster_distance': args.motion_cluster_distance, 'depth_sampling': 'same-source-part four-neighbor bilinear; otherwise per-part fallback'},
              'moving_rigid_rows': int(movable.sum()), 'ray_repaired_rows': int(exact.sum()),
              'fallback_changed_rows': int((changed & ~exact).sum()), 'unchanged_moving_rows': int((movable & ~changed).sum()),
              'pixel_motion_disagreement_rows': int((movable & ~agreement).sum()),
              'edge_or_invalid_depth_rows': int((agreement & ~interior).sum()),
              'out_of_ratio_bound_rows': int((agreement & interior & ~ratio_valid).sum()),
              'repaired_gaussians': int(changed.sum()), 'reassigned_within_motion_group': int((labels != original_labels).sum()),
              'part_repairs': part_rows, 'immutable_fields': protected,
              'immutable_attributes_sha256_before': protected_before, 'immutable_attributes_sha256_after': protected_after,
              'colors_opacity_rotations_unchanged': True, 'static_and_elastic_geometry_unchanged': True,
              'gaussian_count_unchanged': True, 'labels_unchanged': bool(np.array_equal(labels, original_labels)),
              'dynamic_membership_unchanged': True, 'registration_unchanged': True, 'original_source_preserved': True,
              'max_first_view_center_error_pixels': float(np.max(np.linalg.norm(new_uv - old_uv, axis=1))),
              'max_first_view_footprint_axis_error_pixels': float(np.max(np.abs(new_foot - old_foot))),
              'max_first_view_footprint_axis_relative_error': float(np.max(np.abs(new_foot - old_foot) / np.maximum(old_foot, 1e-20))),
              'ray_depth_error_before': residual_stats((original_depth - desired_z)[exact]),
              'ray_depth_error_after': residual_stats((final_depth - desired_z)[exact]),
              'depth_error_scope': 'Gaussian centre z compared with target z (prior front for collapse; original z plus prior-front minus GS-front for preserve-thickness), not rasterizer depth or ground truth.',
              'geometry_changed': True, 'event_approved': False, 'status': 'GEOMETRY_PRIOR_REPAIR_NOT_EVENT_APPROVED',
              'limitations': 'Prior geometry experiment, not recovered hidden truth. Collapse mode maps every matching depth layer to the '
                             'visible source sheet; preserve-thickness mode translates same-pixel columns using source/GS front depth, '
                             'but different columns or parts can still reorder. Original normal/rotation is retained, so new surface normals may not match. '
                             'Initial per-point projection/footprint preservation does not preserve composited occlusion. '
                             'Static and elastic rows unchanged; full render and raw-event validation required.'}
    args.output_dir.mkdir(parents=True)
    ply.write(args.output_dir / 'source.ply')
    report['repaired_sha256'] = sha256_file(args.output_dir / 'source.ply')
    maximum_distance = max(float(prior['max_attachment_distance']), float(distance[binding['dynamic_indices']].max()) + 1e-7)
    counts = np.bincount(labels, minlength=len(moving)); dc = np.bincount(labels[binding['dynamic_indices']], minlength=len(moving))
    prior.update(source=str(args.output_dir / 'source.ply'), prior_binding=str(args.binding),
                 method=report['repair_method'], geometry_repair_report=str(args.output_dir / 'repair-report.json'),
                 geometry_repair_format=report['format'], max_attachment_distance=maximum_distance,
                 changed_gaussians=int((labels != original_labels).sum()),
                 part_gaussians={name: int(dc[i]) for i, name in enumerate(part_meta['objects']) if moving[i] or i in elastic},
                 all_part_gaussians={name: int(counts[i]) for i, name in enumerate(part_meta['objects'])},
                 distance_percentiles=dict(zip(['p50', 'p90', 'p95', 'p99'], np.percentile(distance, [50, 90, 95, 99]).tolist())),
                 unattached=int((distance > maximum_distance).sum()), appearance_and_rest_geometry_unchanged=False,
                 appearance_unchanged=True, event_approved=False, status=report['status'], limitations=report['limitations'])
    binding.update(labels=labels.astype(np.int16), nearest_vertex=nearest.astype(np.int32), distance=distance.astype(np.float32),
                   metadata_json=np.asarray(json.dumps(prior)))
    np.savez_compressed(args.output_dir / 'initial-binding.npz', **binding)
    (args.output_dir / 'initial-binding.json').write_text(json.dumps(prior, indent=2) + '\n')
    (args.output_dir / 'repair-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('ray_repaired_rows', 'fallback_changed_rows', 'pixel_motion_disagreement_rows',
                     'reassigned_within_motion_group', 'max_first_view_center_error_pixels', 'ray_depth_error_after')}), flush=True)


if __name__ == '__main__':
    main()
