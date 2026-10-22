"""A bounded original-support restoration experiment, never an accepted model.

Recover only existing original rows of named moving parts in a NEW PLY/binding.
By default a row must be behind the original alpha-composited depth by a stated
margin; --all is a clearly labelled ablation. XYZ/local scales are reset to the
original source values so a previously wrong part's radial correction is not
retained. No Gaussian is added, deleted, recoloured or made transparent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from scripts.motion.rebind_lego_coherent import transform, validate_correspondence
from scripts.geometry.repair_lego_part_depth import field_hash, projection_footprint, sha256_file


RESTORED_FIELDS = ('x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2')


def select_original_support(xyz, labels, parts, camera, depth=None, alpha=None, *, margin=.003, restore_all=False):
    xyz, labels = np.asarray(xyz), np.asarray(labels)
    if (xyz.shape != (len(labels), 3) or labels.ndim != 1 or not np.isfinite(xyz).all()
            or not np.isfinite(margin) or margin < 0):
        raise ValueError('Invalid original geometry/labels/occlusion margin')
    target = np.isin(labels, parts)
    if restore_all:
        return target, {'mode': 'all_original_support_ablation', 'eligible_original_rows': int(target.sum())}
    depth, alpha = np.asarray(depth), np.asarray(alpha)
    height, width = camera.get('height', 640), camera.get('width', 640)
    if depth.shape != (height, width) or alpha.shape != depth.shape:
        raise ValueError('Original depth/alpha dimensions disagree with camera')
    pose = np.asarray(camera['pose'], dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not 0 < camera['fovy_degrees'] < 180:
        raise ValueError('Invalid source camera')
    rows = np.flatnonzero(target)
    view = transform(xyz[rows], np.linalg.inv(pose))
    z = -view[:, 2]
    focal = height/(2*np.tan(np.deg2rad(camera['fovy_degrees'])/2))
    uv = focal*view[:, :2]/np.maximum(z[:, None], 1e-8)
    uv[:, 1] *= -1; uv += [(width-1)/2, (height-1)/2]
    pixel = np.rint(np.clip(uv, -1, [width, height])).astype(np.int64)
    inside = ((z > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < width)
              & (pixel[:, 1] >= 0) & (pixel[:, 1] < height))
    keep = np.zeros(len(rows), dtype=bool)
    sample = np.flatnonzero(inside)
    front, opacity = depth[pixel[sample, 1], pixel[sample, 0]], alpha[pixel[sample, 1], pixel[sample, 0]]
    evidence = np.isfinite(front) & (front > 0) & np.isfinite(opacity) & (opacity > .8)
    keep[sample] = evidence & (z[sample]-front > margin)
    mask = np.zeros(len(labels), dtype=bool); mask[rows[keep]] = True
    return mask, {'mode': 'occluded_original_support_only', 'eligible_original_rows': len(rows),
                  'inside_original_camera': int(inside.sum()), 'confident_depth_samples': int(evidence.sum()),
                  'occlusion_margin_scene': margin, 'alpha_threshold': .8,
                  'selected_original_rows': int(mask.sum()),
                  'visibility_scope': 'Behind alpha-composited original Gaussian depth, not a ground-truth visibility oracle.'}


def restore_geometry(vertices, original, mask):
    mask = np.asarray(mask)
    if len(vertices) != len(original) or mask.dtype != bool or mask.shape != (len(vertices),):
        raise ValueError('Original/current rows and restoration mask must match')
    for name in RESTORED_FIELDS:
        if name not in vertices.dtype.names or name not in original.dtype.names:
            raise ValueError(f'Missing original geometry field: {name}')
        if not np.isfinite(original[name][mask]).all():
            raise ValueError(f'Non-finite original geometry field: {name}')
    for name in RESTORED_FIELDS:
        vertices[name][mask] = original[name][mask]


def union_dynamic(previous, restored):
    previous, restored = np.asarray(previous), np.asarray(restored)
    if (previous.ndim != 1 or previous.dtype.kind not in 'iu' or restored.dtype != bool or restored.ndim != 1
            or np.any(previous < 0) or np.any(previous >= len(restored))):
        raise ValueError('Invalid previous dynamic rows or restoration mask')
    dynamic = restored.copy(); dynamic[previous] = True
    return np.flatnonzero(dynamic).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-ply', 'binding', 'original-source-ply', 'original-binding', 'parts', 'camera', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--depth-report-npz', type=Path)
    parser.add_argument('--parts-to-restore', type=int, nargs='+', default=[229, 235, 237])
    parser.add_argument('--occlusion-margin', type=float, default=.003)
    parser.add_argument('--all', dest='restore_all', action='store_true', help='Ablation: restore all original target-part rows, including front-visible ones')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.restore_all and args.depth_report_npz is None:
        parser.error('--depth-report-npz is required without --all')
    if args.workers < 1 or not np.isfinite(args.occlusion_margin) or args.occlusion_margin < 0:
        parser.error('Invalid worker count/occlusion margin')
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {key: archive[key] for key in archive.files}
    with np.load(args.binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    with np.load(args.original_binding, allow_pickle=False) as archive:
        original_binding = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(parts['metadata_json']))
    binding_meta = json.loads(str(binding['metadata_json']))
    requested = sorted(set(args.parts_to_restore))
    matrices = parts['object_matrices'].astype(np.float64)
    moving = np.max(abs(matrices @ np.linalg.inv(matrices[0])-np.eye(4)), axis=(0, 2, 3)) > 1e-4
    if (not requested or min(requested) < 0 or max(requested) >= len(moving)
            or any(part in metadata['deforming_objects'] or not moving[part] for part in requested)):
        raise ValueError('Only existing moving rigid source parts can be restored')
    camera_record = json.loads(args.camera.read_text())
    camera = camera_record.get('camera', camera_record)
    depth = alpha = None
    if not args.restore_all:
        diagnostic = json.loads(args.depth_report_npz.with_suffix('.json').read_text())
        if Path(diagnostic['source']).resolve() != args.original_source_ply.resolve():
            raise ValueError('Occlusion depth must come from the original source PLY')
        with np.load(args.depth_report_npz, allow_pickle=False) as archive:
            depth, alpha = archive['gs_depth'], archive['gs_alpha']
    ply = PlyData.read(args.source_ply, mmap='c'); vertices = ply['vertex'].data
    original = PlyData.read(args.original_source_ply, mmap=True)['vertex'].data
    if (len(vertices) != len(original) or len(vertices) != len(binding['labels'])
            or len(vertices) != len(original_binding['labels']) or vertices.dtype.names != original.dtype.names):
        raise ValueError('Original and current Gaussian rows/fields are incompatible')
    protected = [name for name in vertices.dtype.names if name not in RESTORED_FIELDS]
    protected_before = field_hash(vertices, protected)
    if protected_before != field_hash(original, protected):
        raise ValueError('Current and original protected appearance/rotation attributes already differ')
    xyz = np.column_stack([original[name] for name in ('x', 'y', 'z')]).astype(np.float64)
    selected, selection = select_original_support(xyz, original_binding['labels'], requested, camera, depth, alpha,
                                                  margin=args.occlusion_margin, restore_all=args.restore_all)
    # Do not steal currently elastic rows even if their old nearest rigid label
    # differs: the elastic attachment is a distinct source-model contract.
    conflict = selected & np.isin(binding['labels'], metadata['deforming_objects'])
    selected[conflict] = False
    selection['current_elastic_conflicts_retained'] = int(conflict.sum())
    selection['selected_original_rows'] = int(selected.sum())
    if not selected.any():
        raise ValueError('No original target-part rows satisfy the requested restoration policy')
    before_xyz = np.column_stack([vertices[name][selected] for name in ('x', 'y', 'z')]).astype(float)
    before_scale = np.column_stack([vertices[f'scale_{i}'][selected] for i in range(3)]).astype(float)
    unchanged_geometry_hash = field_hash(vertices[~selected], RESTORED_FIELDS)
    restore_geometry(vertices, original, selected)
    if field_hash(vertices, protected) != protected_before or field_hash(vertices[~selected], RESTORED_FIELDS) != unchanged_geometry_hash:
        raise AssertionError('Restoration changed an unselected row or protected attribute')
    labels, nearest, distances = binding['labels'].copy(), binding['nearest_vertex'].copy(), binding['distance'].copy()
    labels[selected] = original_binding['labels'][selected]
    surface = transform(parts['rest_points'], binding['registration'])
    for part in requested:
        rows = np.flatnonzero(selected & (labels == part))
        surface_rows = np.flatnonzero(parts['point_parts'] == part)
        if len(rows):
            distance, neighbor = cKDTree(surface[surface_rows]).query(xyz[rows], workers=args.workers)
            nearest[rows] = surface_rows[neighbor]; distances[rows] = distance
    validate_correspondence(labels, nearest, parts, metadata)
    dynamic = union_dynamic(binding['dynamic_indices'], selected)
    if not np.all(np.isin(binding['dynamic_indices'], dynamic)):
        raise AssertionError('A prior dynamic row was removed')
    after_xyz = xyz[selected]
    after_scale = np.column_stack([vertices[f'scale_{i}'][selected] for i in range(3)]).astype(float)
    before_front, before_uv, before_footprint = projection_footprint(before_xyz, before_scale, camera)
    after_front, after_uv, after_footprint = projection_footprint(after_xyz, after_scale, camera)
    if not np.array_equal(before_front, after_front):
        raise AssertionError('Restoration crossed the initial camera plane')
    footprint_relative = np.abs(after_footprint-before_footprint)/np.maximum(before_footprint, 1e-20)
    records = [{'part': part, 'name': metadata['objects'][part],
                'original_rows': int(np.sum(original_binding['labels'] == part)),
                'current_rows_before': int(np.sum(binding['labels'] == part)),
                'restored_original_rows': int(np.sum(selected & (labels == part))),
                'rows_after': int(np.sum(labels == part))} for part in requested]
    report = {'format': 'lego_original_occluded_support_restoration_v1', 'source': str(args.source_ply),
              'original_source': str(args.original_source_ply), 'source_sha256': sha256_file(args.source_ply),
              'original_source_sha256': sha256_file(args.original_source_ply), 'binding': str(args.binding),
              'original_binding': str(args.original_binding), 'parts': str(args.parts), 'selection': selection,
              'part_support': records, 'gaussian_count': len(vertices), 'dynamic_before': len(binding['dynamic_indices']),
              'dynamic_after': len(dynamic), 'original_dynamic_membership_retained': True,
              'geometry_reset_fields': list(RESTORED_FIELDS), 'colors_opacity_rotations_unchanged': True,
              'gaussians_added': 0, 'gaussians_deleted': 0, 'source_row_order_unchanged': True,
              'protected_attribute_sha256': protected_before,
              'selection_mask_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
              'max_initial_projected_center_error_pixels': float(np.linalg.norm(after_uv-before_uv, axis=-1).max()) if len(after_uv) else 0.,
              'max_initial_footprint_relative_error': float(footprint_relative.max()) if footprint_relative.size else 0.,
              'status': 'INITIALIZATION_ONLY_NOT_EVENT_APPROVED', 'event_approved': False,
              'limitations': 'Original nearest-surface part labels are hypotheses, not known segmentation. Occlusion is checked only against initial composited GS depth. '
                            'Resetting original depth removes the previous radial repair on selected rows; composited occlusion can change. '
                            'This creates no new geometry or hidden-surface evidence and requires rendered/event validation.'}
    target_parent = args.output_dir.parent
    while not target_parent.exists():
        target_parent = target_parent.parent
    if shutil.disk_usage(target_parent).free < args.source_ply.stat().st_size*1.2+128*1024**2:
        raise OSError('Insufficient space for a separate reversible source copy')
    args.output_dir.mkdir(parents=True)
    ply.write(args.output_dir/'source.ply')
    np.save(args.output_dir/'restored_rows_mask.npy', selected, allow_pickle=False)
    report['repaired_source_sha256'] = sha256_file(args.output_dir/'source.ply')
    binding_meta.update(source=str(args.output_dir/'source.ply'), prior_binding=str(args.binding),
                        method='Bounded original occluded part support restoration; original geometry/scales reset on selected existing rows',
                        restoration_report=str(args.output_dir/'restore-report.json'), dynamic_count=len(dynamic),
                        static_count=len(vertices)-len(dynamic), event_approved=False, status=report['status'],
                        max_attachment_distance=max(binding_meta['max_attachment_distance'], float(distances[dynamic].max())+1e-7),
                        restored_original_rows=int(selected.sum()),
                        restored_label_rows=int(np.sum(selected & (labels != binding['labels']))),
                        part_gaussians={name: int(np.sum(labels[dynamic] == part))
                                        for part, name in enumerate(metadata['objects']) if moving[part]},
                        all_part_gaussians={name: int(np.sum(labels == part)) for part, name in enumerate(metadata['objects'])},
                        limitations=report['limitations'])
    binding.update(labels=labels, nearest_vertex=nearest.astype(np.int32), distance=distances.astype(np.float32),
                   dynamic_indices=dynamic, metadata_json=np.asarray(json.dumps(binding_meta)))
    np.savez_compressed(args.output_dir/'initial-binding.npz', **binding)
    (args.output_dir/'initial-binding.json').write_text(json.dumps(binding_meta, indent=2)+'\n')
    (args.output_dir/'restore-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('selection', 'part_support', 'gaussian_count', 'dynamic_after',
                                                 'max_initial_projected_center_error_pixels', 'max_initial_footprint_relative_error')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
