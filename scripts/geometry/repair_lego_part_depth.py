"""Repair per-part depth using an explicitly allowed Blender geometry prior.

Each supported moving rigid part receives one camera-centred radial scale. Its
Gaussian log scales receive the same factor, preserving individual first-view
centres and footprints up to float rounding. Composited pixels are NOT guaranteed
identical because depth ordering between parts can change. This is a NEW geometry
model, not an unchanged PLY or hidden-depth ground truth; raw-event audit required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.motion.rebind_lego_coherent import transform, validate_correspondence


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def field_hash(vertices, fields):
    digest = hashlib.sha256()
    for name in fields:
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(np.ascontiguousarray(vertices[name]).view(np.uint8))
    return digest.hexdigest()


def erode_mask(mask, pixels=1):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(pixels):
        padded = np.pad(result, 1, constant_values=False)
        result = np.logical_and.reduce([padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
                                        for dy in range(3) for dx in range(3)])
    return result


def estimate_radial_scale(prior_depth, gs_depth, valid, minimum_pixels=16,
                          minimum_scale=.8, maximum_scale=1.25):
    keep = valid & np.isfinite(prior_depth) & np.isfinite(gs_depth) & (prior_depth > 0) & (gs_depth > 0)
    count = int(keep.sum())
    raw = float(np.median(prior_depth[keep] / gs_depth[keep])) if count else None
    if count < minimum_pixels:
        return 1., {'support_pixels': count, 'raw_median_ratio': raw,
                    'scale': 1., 'status': 'insufficient_evidence_kept_unchanged', 'clamped': False}
    scale = float(np.clip(raw, minimum_scale, maximum_scale))
    return scale, {'support_pixels': count, 'raw_median_ratio': raw, 'scale': scale,
                   'status': 'supported_geometry_prior', 'clamped': scale != raw}


def radial_repair(xyz, log_scales, origin, scale):
    return origin + scale * (xyz - origin), log_scales + np.log(scale)


def projection_footprint(xyz, log_scales, camera):
    view = transform(xyz, np.linalg.inv(np.asarray(camera['pose'], dtype=np.float64)))
    depth = -view[:, 2]
    front = depth > 1e-8
    focal = camera.get('height', 640) / (2 * np.tan(np.deg2rad(camera['fovy_degrees']) / 2))
    uv = focal * view[front, :2] / depth[front, None]
    footprint = focal * np.exp(log_scales[front]) / depth[front, None]
    return front, uv, footprint


def residual_stats(values):
    values = np.asarray(values)
    if not len(values):
        return {'pixels': 0}
    return {'pixels': int(len(values)), 'median_absolute': float(np.median(np.abs(values))),
            'mean_absolute': float(np.mean(np.abs(values))), 'median_signed': float(np.median(values)),
            'p90_absolute': float(np.percentile(np.abs(values), 90))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-ply', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--depth-report-npz', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--erosion-pixels', type=int, default=1)
    p.add_argument('--minimum-pixels', type=int, default=16)
    p.add_argument('--alpha-threshold', type=float, default=.8)
    p.add_argument('--minimum-scale', type=float, default=.8)
    p.add_argument('--maximum-scale', type=float, default=1.25)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.erosion_pixels < 0 or args.minimum_pixels < 1:
        p.error('Invalid support thresholds')
    if not 0 <= args.alpha_threshold <= 1 or not 0 < args.minimum_scale <= 1 <= args.maximum_scale:
        p.error('Invalid alpha or radial-scale bounds')
    if not np.isfinite([args.minimum_scale, args.maximum_scale]).all():
        p.error('Scale bounds must be finite')
    from scipy.spatial import cKDTree
    from plyfile import PlyData

    with np.load(args.binding, allow_pickle=False) as data:
        binding = {key: data[key] for key in data.files}
    with np.load(args.parts, allow_pickle=False) as data:
        parts = {key: data[key] for key in data.files}
    with np.load(args.depth_report_npz, allow_pickle=False) as data:
        depth = {key: data[key] for key in data.files}
    binding_meta = json.loads(str(binding['metadata_json']))
    part_meta = json.loads(str(parts['metadata_json']))
    camera_record = json.loads(args.camera.read_text())
    camera = camera_record.get('camera', camera_record)
    pose = np.asarray(camera['pose'], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not 0 < camera['fovy_degrees'] < 180:
        raise ValueError('Invalid camera')
    shape = (camera.get('height', 640), camera.get('width', 640))
    for key in ('gs_depth', 'gs_alpha', 'prior_camera_z', 'valid', 'part_ids'):
        if depth[key].shape != shape:
            raise ValueError(f'Depth diagnostic shape mismatch: {key}')
    mask = depth['part_ids']
    if np.any(mask < -1) or np.any(mask >= len(part_meta['objects'])):
        raise ValueError('Diagnostic part IDs exceed the supplied part archive')
    # A depth diagnostic generated under another source or camera scale is unsafe.
    diagnostic_json = args.depth_report_npz.with_suffix('.json')
    diagnostic = json.loads(diagnostic_json.read_text()) if diagnostic_json.exists() else None
    if diagnostic and Path(diagnostic['source']).resolve() != args.source_ply.resolve():
        raise ValueError('Depth diagnostic was rendered from a different source PLY')
    registration = np.asarray(binding['registration'], dtype=np.float64)
    scale = float(np.cbrt(np.linalg.det(registration[:3, :3])))
    expected_registration = pose @ np.diag([scale, scale, scale, 1.]) @ np.linalg.inv(part_meta['camera_pose'])
    if not np.allclose(registration, expected_registration, atol=1e-5, rtol=1e-5):
        raise ValueError('Binding registration is not consistent with the fixed prior/source camera')
    if diagnostic and not np.isclose(diagnostic['camera_registration_scale'], scale, atol=1e-6, rtol=1e-5):
        raise ValueError('Depth prior scale does not match binding registration')
    source_sha256 = sha256_file(args.source_ply)
    ply = PlyData.read(args.source_ply, mmap=False)
    vertices = ply['vertex'].data
    labels = binding['labels']
    if len(vertices) != len(labels):
        raise ValueError('Source and binding Gaussian counts differ')
    original_xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')]).astype(np.float64)
    original_log_scales = np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64)
    if not np.isfinite(original_xyz).all() or not np.isfinite(original_log_scales).all():
        raise ValueError('Invalid source geometry/scales')
    immutable_fields = [name for name in vertices.dtype.names if name not in ('x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2')]
    immutable_before = field_hash(vertices, immutable_fields)
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    rigid = np.ones(len(moving), bool); rigid[part_meta['deforming_objects']] = False
    movable = moving & rigid
    origin = pose[:3, 3]
    valid = (depth['valid'].astype(bool) & (depth['gs_alpha'] > args.alpha_threshold)
             & np.isfinite(depth['gs_depth']) & (depth['gs_depth'] > 0)
             & np.isfinite(depth['prior_camera_z']) & (depth['prior_camera_z'] > 0))
    part_scales = np.ones(len(moving), dtype=np.float64)
    repaired = np.zeros(len(vertices), dtype=bool)
    records = []
    pixel_scale = np.ones(shape, dtype=np.float64)
    for part in np.flatnonzero(movable):
        region = erode_mask(mask == part, args.erosion_pixels) & valid
        factor, record = estimate_radial_scale(depth['prior_camera_z'], depth['gs_depth'], region,
                                               args.minimum_pixels, args.minimum_scale, args.maximum_scale)
        rows = np.flatnonzero(labels == part)
        part_scales[part] = factor
        pixel_scale[mask == part] = factor
        record.update(part=int(part), name=part_meta['objects'][part], gaussian_count=int(len(rows)),
                      repaired_gaussians=int(len(rows)) if factor != 1 else 0,
                      initial_depth_error=residual_stats((depth['gs_depth'] - depth['prior_camera_z'])[region]),
                      predicted_scaled_depth_error=residual_stats((factor * depth['gs_depth'] - depth['prior_camera_z'])[region]))
        if factor != 1 and len(rows):
            new_xyz, new_scales = radial_repair(original_xyz[rows], original_log_scales[rows], origin, factor)
            for i, name in enumerate(('x', 'y', 'z')):
                vertices[name][rows] = new_xyz[:, i]
            for i in range(3):
                vertices[f'scale_{i}'][rows] = new_scales[:, i]
            repaired[rows] = True
        records.append(record)
    final_xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')]).astype(np.float64)
    final_log_scales = np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64)
    fixed = ~movable[labels]
    if not (np.array_equal(final_xyz[fixed], original_xyz[fixed])
            and np.array_equal(final_log_scales[fixed], original_log_scales[fixed])):
        raise AssertionError('Static or elastic Gaussian geometry changed')
    immutable_after = field_hash(vertices, immutable_fields)
    if immutable_before != immutable_after:
        raise AssertionError('A protected Gaussian attribute changed')
    old_front, old_uv, old_footprint = projection_footprint(original_xyz, original_log_scales, camera)
    new_front, new_uv, new_footprint = projection_footprint(final_xyz, final_log_scales, camera)
    if not np.array_equal(old_front, new_front):
        raise AssertionError('Repair moved a Gaussian through the camera plane')
    max_projection_error = float(np.max(np.linalg.norm(new_uv - old_uv, axis=1))) if len(old_uv) else 0.
    max_footprint_error = float(np.max(np.abs(new_footprint - old_footprint))) if len(old_uv) else 0.
    relative_footprint_error = float(np.max(np.abs(new_footprint - old_footprint) / np.maximum(old_footprint, 1e-20))) if len(old_uv) else 0.
    surface = transform(parts['rest_points'], registration)
    nearest = binding['nearest_vertex'].copy(); distance = binding['distance'].copy()
    for part in np.flatnonzero(movable):
        rows = np.flatnonzero(labels == part)
        if not len(rows):
            continue
        points = np.flatnonzero(parts['point_parts'] == part)
        dist, near = cKDTree(surface[points]).query(final_xyz[rows], workers=args.workers)
        nearest[rows] = points[near]; distance[rows] = dist
    validate_correspondence(labels, nearest, parts, part_meta)
    # Geometry repair does not change part ownership or unexpectedly freeze a
    # moving splat. Retain the exact dynamic membership, including elastic bands.
    dynamic_indices = binding['dynamic_indices'].copy()
    maximum_distance = max(float(binding_meta['max_attachment_distance']),
                           float(distance[dynamic_indices].max()) + 1e-7 if len(dynamic_indices) else 0.)
    all_moving_pixels = valid & np.isin(mask, np.flatnonzero(movable))
    report = {
        'format': 'lego_part_depth_repair_v1', 'source': str(args.source_ply),
        'repaired_source': str(args.output_dir / 'source.ply'), 'source_sha256': source_sha256,
        'gaussian_count': len(vertices), 'binding': str(args.binding), 'parts': str(args.parts),
        'depth_report_npz': str(args.depth_report_npz), 'camera': camera,
        'parameters': {'erosion_pixels': args.erosion_pixels, 'minimum_pixels': args.minimum_pixels,
                       'alpha_threshold': args.alpha_threshold, 'minimum_scale': args.minimum_scale,
                       'maximum_scale': args.maximum_scale},
        'part_repairs': records, 'repaired_gaussians': int(repaired.sum()),
        'supported_parts': sum(r['status'] == 'supported_geometry_prior' for r in records),
        'unsupported_parts_kept': sum(r['status'] != 'supported_geometry_prior' for r in records),
        'immutable_fields': immutable_fields, 'immutable_attributes_sha256_before': immutable_before,
        'immutable_attributes_sha256_after': immutable_after,
        'colors_opacity_rotations_unchanged': True, 'static_and_elastic_geometry_unchanged': True,
        'gaussian_count_unchanged': True, 'labels_unchanged': True, 'dynamic_membership_unchanged': True,
        'registration_unchanged': True, 'original_source_preserved': True,
        'max_first_view_center_error_pixels': max_projection_error,
        'max_first_view_footprint_axis_error_pixels': max_footprint_error,
        'max_first_view_footprint_axis_relative_error': relative_footprint_error,
        'footprint_metric': 'focal * each local Gaussian scale / camera-z; rotations unchanged. Not a composited-render comparison.',
        'initial_depth_error_moving_pixels': residual_stats((depth['gs_depth'] - depth['prior_camera_z'])[all_moving_pixels]),
        'predicted_scaled_depth_error_moving_pixels': residual_stats((pixel_scale * depth['gs_depth'] - depth['prior_camera_z'])[all_moving_pixels]),
        'depth_error_scope': 'Diagnostic source z multiplied by per-mask-part scale; predicted only, not repaired rasterizer depth. Re-render required.',
        'geometry_changed': True, 'event_approved': False, 'status': 'GEOMETRY_PRIOR_REPAIR_NOT_EVENT_APPROVED',
        'limitations': 'Allowed Blender geometry prior, not hidden-depth ground truth. Per-part radial scaling changes '
                      'actual part sizes and relative depth. Individual initial projections/footprints are preserved, '
                      'but composited appearance can change with occlusion ordering. Raw events must refine/audit the repaired model.'}
    args.output_dir.mkdir(parents=True)
    repaired_source = args.output_dir / 'source.ply'
    ply.write(repaired_source)
    report['repaired_sha256'] = sha256_file(repaired_source)
    binding_meta.update(source=str(repaired_source), prior_binding=str(args.binding),
                        method='Per-moving-rigid-part camera-ray depth repair from permitted geometry prior; labels retained',
                        geometry_repair_report=str(args.output_dir / 'repair-report.json'),
                        geometry_repair_format=report['format'], gaussian_count=len(vertices),
                        dynamic_count=len(dynamic_indices), static_count=len(vertices) - len(dynamic_indices),
                        max_attachment_distance=maximum_distance,
                        unattached=int((distance > maximum_distance).sum()),
                        distance_percentiles=dict(zip(['p50', 'p90', 'p95', 'p99'],
                                                     np.percentile(distance, [50, 90, 95, 99]).tolist())),
                        appearance_and_rest_geometry_unchanged=False, appearance_unchanged=True,
                        event_approved=False, status=report['status'], limitations=report['limitations'])
    binding.update(nearest_vertex=nearest.astype(np.int32), distance=distance.astype(np.float32),
                   dynamic_indices=dynamic_indices, metadata_json=np.asarray(json.dumps(binding_meta)))
    np.savez_compressed(args.output_dir / 'initial-binding.npz', **binding)
    (args.output_dir / 'initial-binding.json').write_text(json.dumps(binding_meta, indent=2) + '\n')
    (args.output_dir / 'repair-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('format', 'gaussian_count', 'repaired_gaussians',
                      'supported_parts', 'unsupported_parts_kept', 'max_first_view_center_error_pixels',
                      'max_first_view_footprint_axis_relative_error')}), flush=True)


if __name__ == '__main__':
    main()
