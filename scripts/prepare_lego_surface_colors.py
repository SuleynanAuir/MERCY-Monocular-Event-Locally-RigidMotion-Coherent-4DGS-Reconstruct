"""Prepare guarded STATIC DC-color priors; never modify the source Gaussians.

Colors follow the compact-completion same-part median/material and two-sided
normal-shade convention. They are unobserved appearance priors, not recovered
back-side evidence. Position, covariance, opacity and motion remain untouched.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.diagnose_lego_sparse_points import load_inputs, sha256, summarize


C0 = .28209479177387814
LIGHT = np.asarray([.35, .65, .68], np.float64)
LIGHT /= np.linalg.norm(LIGHT)
LUMA = np.asarray([.299, .587, .114], np.float32)


def luminance(rgb):
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite RGB DC colors [N,3]')
    return np.sum(rgb*LUMA, axis=1)


def validate_guard_arrays(indices, candidates, maxima, threshold):
    indices, candidates, maxima = map(np.asarray, (indices, candidates, maxima))
    for name, values in (('guarded indices', indices), ('candidate indices', candidates)):
        if (values.ndim != 1 or values.dtype.kind not in 'iu' or np.any(values < 0)
                or np.any(values > np.iinfo(np.int32).max) or np.any(values[1:] <= values[:-1])):
            raise ValueError(f'{name} must be sorted distinct nonnegative int32-compatible rows')
    if (maxima.shape != candidates.shape or maxima.dtype.kind != 'f'
            or not np.isfinite(maxima).all() or np.any(maxima < 0)
            or type(threshold) not in (float, int) or not np.isfinite(threshold) or not 0 < threshold <= .01):
        raise ValueError('Invalid original-camera exposure maxima/threshold')
    if not np.array_equal(indices, candidates[maxima <= threshold]):
        raise ValueError('Guarded rows differ from the recorded exposure decision')
    return indices.astype(np.int32)


def thin_axis_normals(quaternions, log_scales):
    """World-space local +Z, matching general_utils.build_rotation wxyz.

    L=R@diag(scales), so the thin axis is R[:, :, 2], not a matrix row.
    Existing static source splats must actually have their thinnest axis at 2.
    """
    q, log_scales = np.asarray(quaternions, np.float64), np.asarray(log_scales)
    if (q.ndim != 2 or q.shape[1] != 4 or log_scales.shape != (len(q), 3)
            or not np.isfinite(q).all() or not np.isfinite(log_scales).all()
            or np.any(log_scales[:, 2] > np.minimum(log_scales[:, 0], log_scales[:, 1]))):
        raise ValueError('Expected finite wxyz quaternions and original local-axis-2 thin splats')
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm <= 1e-12) or not np.isfinite(norm).all():
        raise ValueError('Invalid zero/overflow Gaussian quaternion')
    w, x, y, z = (q/norm[:, None]).T
    normals = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals.astype(np.float32)


def part_color_table(parent_gray, parent_labels, mesh_parts, material_gray, required_parts):
    """Replicate the completion builder's exact same-part color-selection rule.

    A fallback part must have one material grayscale value. Unlike silently
    averaging a multi-material part, this is exactly the builder's per-sample
    material rule; unsupported ambiguous fallbacks raise for explicit review.
    """
    parent_gray, parent_labels, mesh_parts, material_gray, required_parts = map(np.asarray,
        (parent_gray, parent_labels, mesh_parts, material_gray, required_parts))
    if (parent_gray.ndim != 1 or parent_labels.shape != parent_gray.shape
            or parent_labels.dtype.kind not in 'iu' or np.any(parent_labels < 0)
            or mesh_parts.ndim != 1 or mesh_parts.dtype.kind not in 'iu' or np.any(mesh_parts < 0)
            or material_gray.shape != mesh_parts.shape or not np.isfinite(material_gray).all()
            or np.any(material_gray < 0) or np.any(material_gray > 1)
            or required_parts.ndim != 1 or required_parts.dtype.kind not in 'iu' or np.any(required_parts < 0)):
        raise ValueError('Invalid parent/material part-color table inputs')
    count = int(max(np.max(parent_labels, initial=0), np.max(mesh_parts, initial=0),
                    np.max(required_parts, initial=0)))+1
    table = np.full(count, np.nan, np.float32)
    records = {}
    order = np.argsort(parent_labels, kind='stable')
    sorted_labels = parent_labels[order]
    for part in np.unique(required_parts):
        begin = int(np.searchsorted(sorted_labels, part, side='left'))
        end = int(np.searchsorted(sorted_labels, part, side='right'))
        values = parent_gray[order[begin:end]]
        values = values[np.isfinite(values) & (values > .04) & (values < .99)]
        if len(values) >= 4:
            table[part] = np.median(values)
            source = 'same_part_parent_gray_median'
        else:
            materials = np.unique(material_gray[mesh_parts == part])
            if len(materials) != 1:
                raise ValueError(f'Part {part} needs explicit material attribution: {len(materials)} fallback values')
            table[part] = materials[0]
            source = 'unobserved_Blender_material_prior'
        records[str(int(part))] = {'base_gray': float(table[part]), 'source': source,
                                   'qualifying_parent_rows': int(len(values))}
    return table, records


def shaded_gray(base_gray, normals):
    base_gray, normals = np.asarray(base_gray, np.float32), np.asarray(normals, np.float32)
    if (base_gray.ndim != 1 or normals.shape != (len(base_gray), 3)
            or not np.isfinite(base_gray).all() or not np.isfinite(normals).all()
            or np.any(base_gray < 0) or np.any(base_gray > 1)
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Expected finite grayscale and unit world normals')
    # Match the builder: float32 gray array, float64 light, in-place multiply.
    result = base_gray.copy()
    result *= .8+.2*np.abs(normals @ LIGHT)
    return np.clip(result, .06, .9)


def color_change_selection(original_gray, target_gray, nonroot_reference, minimum_change,
                           preserve_dark_below=None, static_reference_only=False):
    """Apply explicit safety exclusions without changing or capping any target."""
    original_gray, target_gray, nonroot_reference = map(np.asarray,
                                                       (original_gray, target_gray, nonroot_reference))
    if (original_gray.ndim != 1 or target_gray.shape != original_gray.shape
            or nonroot_reference.shape != original_gray.shape or nonroot_reference.dtype != bool
            or not np.isfinite(original_gray).all() or not np.isfinite(target_gray).all()
            or type(minimum_change) not in (float, int) or not np.isfinite(minimum_change)
            or not 0 <= minimum_change < 1 or type(static_reference_only) is not bool):
        raise ValueError('Invalid color-change selection inputs')
    if preserve_dark_below is not None and (
            type(preserve_dark_below) not in (float, int) or not np.isfinite(preserve_dark_below)
            or not 0 <= preserve_dark_below <= 1):
        raise ValueError('Dark preservation threshold must be a finite grayscale value in [0,1]')
    delta = target_gray-original_gray
    magnitude = np.abs(delta) > minimum_change
    dark_brightening = ((original_gray < preserve_dark_below) & (delta > 0)
                       if preserve_dark_below is not None else np.zeros(len(delta), bool))
    nonroot_block = nonroot_reference if static_reference_only else np.zeros(len(delta), bool)
    blocked_dark = magnitude & dark_brightening
    blocked_nonroot = magnitude & nonroot_block
    selected = magnitude & ~dark_brightening & ~nonroot_block
    reasons = {
        'guarded_rows': int(len(delta)), 'passes_minimum_absolute_change': int(magnitude.sum()),
        'below_or_equal_minimum_change': int((~magnitude).sum()),
        'blocked_dark_brightening': int(blocked_dark.sum()),
        'blocked_nonroot_reference': int(blocked_nonroot.sum()),
        'blocked_by_both_safety_rules': int((blocked_dark & blocked_nonroot).sum()),
        'blocked_by_any_safety_rule': int((magnitude & ~selected).sum()),
        'selected_rows': int(selected.sum()),
        'selected_dark_rows_darkened': int((selected & (delta < 0)
                                             & (original_gray < preserve_dark_below)).sum())
                                      if preserve_dark_below is not None else 0,
        'selected_nonroot_reference_rows': int((selected & nonroot_reference).sum()),
    }
    return selected, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'selection', 'parts', 'binding', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--key', default='guarded_surface_indices')
    parser.add_argument('--minimum-change', type=float, default=.08,
                        help='Skip color changes at/below this RGB-gray difference; never delete those Gaussians')
    parser.add_argument('--preserve-dark-below', type=float,
                        help='Below this original luma, prohibit brightening but allow darkening')
    parser.add_argument('--static-reference-only', action='store_true',
                        help='Reject nearest-all-rigid-reference parts belonging to nonroot motion groups')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not np.isfinite(args.minimum_change) or not 0 <= args.minimum_change < 1 or not 1 <= args.workers <= 8:
        raise ValueError('Invalid bounded color-change threshold or CPU workers')
    if args.preserve_dark_below is not None and (not np.isfinite(args.preserve_dark_below)
                                                or not 0 <= args.preserve_dark_below <= 1):
        raise ValueError('Dark preservation threshold must lie in [0,1]')
    if args.output.suffix != '.npz' or args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new NPZ and JSON output pair')
    args.output = args.output.resolve()
    for name in ('baseline', 'candidate', 'selection', 'parts', 'binding'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    started = time.monotonic()
    # Reuse the strict immutable-parent partition and reference registration.
    inputs, eligible, _, _, reference, reference_parts, _, _ = load_inputs(args.baseline, args.binding, args.parts)
    cache, completion = inputs['cache'], inputs['completion']
    paths, hashes = dict(inputs['paths']), dict(inputs['hashes'])
    paths.update(selection=args.selection, correction=args.candidate/'correction.json',
                 event_audit=args.candidate/'event-audit.json', mesh_prior=Path(completion['mesh_prior']))
    for name in ('selection', 'correction', 'event_audit', 'mesh_prior'):
        hashes[name] = sha256(paths[name])
    if hashes['mesh_prior'] != completion['mesh_prior_sha256']:
        raise ValueError('Completion mesh-prior checksum differs')
    record = json.loads(paths['correction'].read_text())
    audit = json.loads(paths['event_audit'].read_text())
    if (record.get('format') != 'event_supervised_hard_joint_alignment_v4'
            or cache['frame_count'] != 400 or record.get('frame_count') != 400
            or record['joint_structure']['input_sha256']['baseline_cache_metadata'] != hashes['baseline_cache']):
        raise ValueError('Expected the verified 400-frame hard-joint source model')
    for key, expected in (('source_ply', hashes['source_ply']), ('correction', hashes['correction']),
                          ('baseline_cache_metadata', hashes['baseline_cache']), ('dynamic_indices', hashes['dynamic_indices'])):
        if audit.get('artifact_sha256', {}).get(key) != expected:
            raise ValueError('Candidate event-audit identity differs: '+key)
    with np.load(args.selection, allow_pickle=False) as archive:
        guard_meta = json.loads(str(archive['metadata_json']))
        indices = validate_guard_arrays(archive[args.key], archive['candidate_indices'], archive['maximum_contribution'],
                                        guard_meta['threshold_output_pixels'])
    if (guard_meta.get('format') != 'lego_cleanup_original_camera_guard_v1'
            or guard_meta.get('frames_checked') != 400 or guard_meta.get('original_camera') != cache['camera']
            or guard_meta.get('selected_count') != len(indices)):
        raise ValueError('Missing exact full-timeline original-camera exposure guard')
    for key in ('source_ply', 'baseline_cache', 'correction', 'event_audit'):
        if guard_meta.get('input_sha256', {}).get(key) != hashes[key]:
            raise ValueError('Exposure guard belongs to another source: '+key)
    if not len(indices) or not np.isin(indices, eligible, assume_unique=True).all():
        raise ValueError('Color candidates must all be guarded, original retained STATIC rows')
    del eligible
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    vertices = PlyData.read(paths['source_ply'], mmap='r')['vertex'].data
    with np.load(args.binding, allow_pickle=False) as archive:
        parent_labels = archive['labels']
    with np.load(paths['mesh_prior'], allow_pickle=False) as archive:
        mesh_parts, materials = archive['part_ids'], archive['material_gray']
        mesh_meta = json.loads(str(archive['metadata_json']))
    if mesh_meta.get('objects') != inputs['objects'] or mesh_meta.get('parts_sha256') != hashes['parts']:
        raise ValueError('Mesh material identities disagree with registered reference parts')
    xyz = np.column_stack([vertices[name][indices] for name in ('x', 'y', 'z')])
    tree = cKDTree(reference)
    distances, nearest = tree.query(xyz, k=1, workers=args.workers)
    nearest_parts = reference_parts[nearest].astype(np.int32)
    original_parts = parent_labels[indices].astype(np.int32)
    del tree, reference, reference_parts
    quaternions = np.column_stack([vertices[f'rot_{i}'][indices] for i in range(4)])
    log_scales = np.column_stack([vertices[f'scale_{i}'][indices] for i in range(3)])
    normals = thin_axis_normals(quaternions, log_scales)
    prefix = completion['prefix_count']
    parent_gray = vertices['f_dc_0'][:prefix]*C0+.5
    original_rgb = np.column_stack([vertices[f'f_dc_{channel}'][indices]*C0+.5 for channel in range(3)])
    original_gray = luminance(original_rgb)
    original_channel_spread = np.ptp(original_rgb, axis=1)
    required = np.union1d(nearest_parts, original_parts)
    table, color_records = part_color_table(parent_gray, parent_labels, mesh_parts, materials, required)
    target = shaded_gray(table[nearest_parts], normals)
    original_label_target = shaded_gray(table[original_parts], normals)
    delta = target-original_gray
    assignment_same = nearest_parts == original_parts
    source_is_median = np.array([color_records[str(int(part))]['source'] == 'same_part_parent_gray_median'
                                 for part in nearest_parts], bool)
    group_by_part = np.full(len(inputs['objects']), -1, np.int32)
    for group in record['joint_structure']['source_groups']:
        group_by_part[group['objects']] = group['group']
    roots = np.flatnonzero(np.asarray(record['joint_structure']['parents']) == -1)
    nonroot_assignment = ~np.isin(group_by_part[nearest_parts], roots)
    changed, selection_reasons = color_change_selection(original_gray, target, nonroot_assignment,
        args.minimum_change, args.preserve_dark_below, args.static_reference_only)
    selected = indices[changed]
    if not len(selected):
        raise ValueError('No color changes pass the recorded threshold and safety rules')
    per_part = []
    for part in np.unique(nearest_parts):
        rows = nearest_parts == part
        per_part.append({'nearest_part': int(part), 'name': inputs['objects'][part],
                         'guarded_rows': int(rows.sum()), 'changed_rows': int((rows & changed).sum()),
                         'different_original_label_rows': int((rows & ~assignment_same).sum()),
                         **color_records[str(int(part))]})
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError('Source changed while preparing colors: '+name)
    metadata = {
        'format': 'lego_static_surface_color_prior_v1', 'source': str(paths['source_ply']),
        'source_sha256': hashes['source_ply'], 'source_ply_sha256': hashes['source_ply'],
        'input_sha256': hashes, 'input_paths': {key: str(path) for key, path in paths.items()},
        'baseline': str(args.baseline), 'candidate': str(args.candidate), 'selection_key': args.key,
        'guarded_rows': len(indices), 'changed_rows': len(selected),
        'skipped_small_color_changes': selection_reasons['below_or_equal_minimum_change'],
        'minimum_absolute_gray_change': args.minimum_change,
        'preserve_dark_below': args.preserve_dark_below, 'static_reference_only': args.static_reference_only,
        'selection_reason_counts': selection_reasons,
        'safety_rule_semantics': 'Skip, not recolor, original-luma-below-threshold rows whose target is brighter. '
                                 'If static-reference-only, reject nearest-all-rigid-reference parts in nonroot '
                                 'groups; do not substitute a second-nearest static part. No additional target '
                                 'or change-amplitude clipping is applied.',
        'original_gray_convention': 'Source DC RGB luminance .299R+.587G+.114B, matching MotionRenderer. Parent color TABLE intentionally uses f_dc_0 exactly as the completion builder does.',
        'original_color_channel_spread': summarize(original_channel_spread),
        'original_nonmonochrome_rows_above_1e_5': int((original_channel_spread > 1e-5).sum()),
        'dc_gray_convention': 'Linear grayscale RGB in [0,1], NOT SH DC coefficients; apply (dc_gray-.5)/0.28209479177387814 on all three f_dc channels.',
        'allowed_fields': ['f_dc_0', 'f_dc_1', 'f_dc_2'],
        'positions_rotations_scales_opacity_motion_frozen': True, 'gaussian_rows_deleted': 0,
        'geometry_and_opacity_frozen': True, 'source_files_modified': False, 'cpu_only': True,
        'original_camera_guard': {'frames_checked': 400, 'threshold_output_pixels': guard_meta['threshold_output_pixels'],
                                  'maximum_selected_contribution': guard_meta['maximum_selected_contribution'],
                                  'scope': 'Inherited exact-source exposure screen, not a new acceptance audit.'},
        'reference_geometry': 'Nearest sample over every registered rigid source part, with elastic parts excluded; not nearest assigned-part-only.',
        'normal_convention': 'Normalize original wxyz quaternion; world normal is R[:,2], the verified minimum-scale local axis 2. Sign cancels through absolute shading.',
        'shading_formula': 'clip(base_gray * (0.8 + 0.2 * abs(dot(world_normal, normalize([.35,.65,.68])))), .06, .9)',
        'light': LIGHT.tolist(), 'part_color_rule': 'Exact completion-builder rule: >=4 finite original parent same-part f_dc_0-derived gray values strictly in (.04,.99) -> median; otherwise exact uniform Blender material_gray. Ambiguous multi-material fallbacks reject.',
        'label_assignment_counts': {'same_original_and_nearest_part': int(assignment_same.sum()),
                                    'different_original_and_nearest_part': int((~assignment_same).sum()),
                                    'changed_same_part': int((changed & assignment_same).sum()),
                                    'changed_different_part': int((changed & ~assignment_same).sum()),
                                    'nearest_part_is_nonroot_rigid_group': int(nonroot_assignment.sum())},
        'color_source_counts': {'parent_same_part_gray_median': int(source_is_median.sum()),
                                'Blender_material_prior': int((~source_is_median).sum())},
        'all_guarded_statistics': {'old_gray': summarize(original_gray), 'target_gray': summarize(target),
                                   'absolute_change': summarize(np.abs(delta)), 'nearest_mesh_distance': summarize(distances),
                                   'difference_from_original_label_target': summarize(np.abs(target-original_label_target))},
        'changed_statistics': {'old_gray': summarize(original_gray[changed]), 'target_gray': summarize(target[changed]),
                               'absolute_change': summarize(np.abs(delta[changed])),
                               'brighter_rows': int((delta[changed] > 0).sum()), 'darker_rows': int((delta[changed] < 0).sum())},
        'part_table': color_records, 'per_nearest_part': per_part,
        'input_identity_rechecked': True, 'seconds': time.monotonic()-started,
        'unobserved_appearance_is_prior': True, 'approved_for_export': False,
        'scope': 'Only a DC-color proposal for previously guarded static splats. Geometry/opacity are preserved '
                 'to avoid holes. Same-part medians and normal shading are priors, NOT observed back-side truth. '
                 'Nearest-part attribution can differ from old labels and is recorded, not asserted correct. '
                 'This proposal changes no model and requires actual Gaussian multiview preview, all-400 RGB '
                 'comparison and unchanged event-gate checks before any deployment.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, indices=selected.astype(np.int32), dc_gray=target[changed].astype(np.float32),
                        original_gray=original_gray[changed].astype(np.float32),
                        nearest_part=nearest_parts[changed], original_part=original_parts[changed],
                        nearest_mesh_distance=distances[changed].astype(np.float32),
                        metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'guarded_rows': len(indices), 'changed_rows': len(selected),
                      'same_nearest_original_part': int(assignment_same.sum()),
                      'median_absolute_gray_change': float(np.median(np.abs(delta[changed]))),
                      'seconds': metadata['seconds'], 'source_modified': False}), flush=True)


if __name__ == '__main__':
    main()
