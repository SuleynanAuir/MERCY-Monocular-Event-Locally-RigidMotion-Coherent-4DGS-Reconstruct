"""CPU-only static cabin/back surface PROPOSAL; never edit or approve a model.

The bounded target is explicit identity-root parts within 0..141. Actual mesh
triangles provide geometry; same-part appearance is an explicitly disclosed
prior. A separate GPU experiment must protect observed raster contributions
and assess full-timeline events, roof detail and unseen-view coverage.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.diagnose_lego_sparse_points import load_inputs, sha256, summarize
from scripts.prepare_lego_roof_rebuild import gaussian_attributes, geometry_retirement
from scripts.prepare_lego_surface_colors import C0, part_color_table, shaded_gray
from utils.mesh_surface_sampling import sample_mesh


FORMAT = 'lego_static_back_detail_proposal_v1'
MAX_NEW = 500000


def validate_target_parts(requested, group_ids, quaternions, translations, parents,
                          excluded_parts=()):
    """Require explicit sorted cabin/body IDs and identity-root motion at ALL times."""
    requested, group_ids, quaternions, translations, parents = map(np.asarray,
        (requested, group_ids, quaternions, translations, parents))
    if (requested.ndim != 1 or requested.dtype.kind not in 'iu' or not len(requested)
            or np.any(requested < 0) or np.any(requested > 141)
            or np.any(requested[1:] <= requested[:-1])
            or group_ids.ndim != 1 or group_ids.dtype.kind not in 'iu'
            or np.any(requested >= len(group_ids)) or parents.ndim != 1
            or parents.dtype.kind not in 'iu' or not len(parents)
            or quaternions.ndim != 3 or quaternions.shape[1:] != (len(parents), 4)
            or translations.shape != quaternions.shape[:2]+(3,)
            or not len(quaternions) or not np.isfinite(quaternions).all()
            or not np.isfinite(translations).all()
            or np.isin(requested, np.asarray(excluded_parts)).any()):
        raise ValueError('Invalid explicit bounded rigid target parts or motion schema')
    groups = group_ids[requested]
    if (np.any(groups < 0) or np.any(groups >= len(parents))
            or np.any(parents[groups] != -1)
            or not np.allclose(quaternions[:, groups], [1, 0, 0, 0], rtol=0, atol=1e-7)
            or not np.allclose(translations[:, groups], 0, rtol=0, atol=1e-7)):
        raise ValueError('Every target must have identity-root motion for all supplied frames')
    return requested.astype(np.int32)


def assemble_retired(original_indices, eligible_indices, suffix_parts, prefix, total,
                     dynamic_indices, target_parts):
    """Retire only already-eligible originals and exact target completion rows."""
    original, eligible, parts, dynamic, target = map(np.asarray,
        (original_indices, eligible_indices, suffix_parts, dynamic_indices, target_parts))
    if (type(prefix) is not int or type(total) is not int or not 0 < prefix < total
            or total > np.iinfo(np.int32).max
            or any(a.ndim != 1 or a.dtype.kind not in 'iu' for a in
                   (original, eligible, parts, dynamic, target))
            or parts.shape != (total-prefix,) or np.any(parts < 0)
            or not len(target) or np.any(target < 0) or np.any(target > 141)
            or np.any(target[1:] <= target[:-1])
            or np.any(original < 0) or np.any(original >= prefix)
            or np.any(original[1:] <= original[:-1])
            or np.any(eligible < 0) or np.any(eligible >= prefix)
            or np.any(eligible[1:] <= eligible[:-1])
            or not np.isin(original, eligible).all()
            or np.any(dynamic < 0) or np.any(dynamic >= total)
            or np.any(dynamic[1:] <= dynamic[:-1])):
        raise ValueError('Invalid protected static/prefix/suffix identity partitions')
    suffix = (prefix+np.flatnonzero(np.isin(parts, target))).astype(np.int32)
    retired = np.concatenate((original.astype(np.int32), suffix))
    if np.intersect1d(retired, dynamic).size:
        raise ValueError('Proposal retirement touches protected baked dynamic motion')
    return retired, suffix


def bounded_samples(vertices, triangles, part_ids, material, target_parts, spacing):
    """Exact-triangle, orientation-aware sampling with a fail-closed output cap."""
    if type(spacing) not in (float, int) or not .0015 <= spacing <= .0025:
        raise ValueError('Back proposal spacing must lie in [.0015,.0025]')
    selected = np.flatnonzero(np.isin(part_ids, target_parts)).astype(np.int32)
    if not np.array_equal(np.unique(part_ids[selected]), target_parts):
        raise ValueError('Target mesh part set is incomplete')
    result = sample_mesh(vertices, triangles[selected], part_ids[selected], material[selected],
                         [], spacing, max_samples=5000000)
    if not 0 < len(result['xyz']) <= MAX_NEW:
        raise ValueError(f'Proposal has {len(result["xyz"])} samples, exceeds explicit {MAX_NEW} '
                         'cap; choose fewer explicit parts or a permitted spacing. Never truncate.')
    result['triangle_ids'] = selected[result['triangle_ids']]
    return result, selected


def recheck_inputs(paths, hashes):
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError('Input changed during back proposal preparation: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'parts', 'binding', 'masks', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--part-ids', type=int, nargs='+', default=list(range(142)),
                        help='Explicit sorted unique IDs in 0..141; cap failures require a smaller selection.')
    parser.add_argument('--spacing', type=float, default=.0015)
    parser.add_argument('--tangent-factor', type=float, default=.70)
    parser.add_argument('--normal-factor', type=float, default=.10)
    parser.add_argument('--opacity', type=float, default=.95)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.output.suffix != '.npz' or args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Choose a new NPZ/JSON diagnostic proposal output pair')
    if not 1 <= args.workers <= 8:
        raise ValueError('CPU workers must lie in [1,8]')
    args.output = args.output.resolve()
    for name in ('baseline', 'candidate', 'parts', 'binding', 'masks'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    started = time.monotonic()
    from scipy.spatial import cKDTree
    from plyfile import PlyData
    from scripts.render_lego_multiview import verify_inputs
    cache, record, _, _, checked_paths, checked_hashes, motion_error, _ = verify_inputs(
        args.baseline, args.candidate, args.masks)
    if not cache.get('color_residual'):
        raise ValueError('Back experiment must start from the latest verified color-residual parent')
    inputs, eligible, xyz, labels, reference, reference_parts, scales, _ = load_inputs(
        args.baseline, args.binding, args.parts)
    paths, hashes = dict(inputs['paths']), dict(inputs['hashes'])
    paths.update(checked_paths)
    hashes.update(checked_hashes)
    with np.load(paths['motion'], allow_pickle=False) as archive:
        group_ids, q, t = archive['group_ids'], archive['quaternions'], archive['translations']
    target = validate_target_parts(np.asarray(args.part_ids, np.int32), group_ids, q, t,
        np.asarray(record['joint_structure']['parents']), inputs['excluded_deforming_parts'])
    with np.load(paths['mesh'], allow_pickle=False) as archive:
        vertices, triangles, part_ids = archive['vertices'], archive['triangles'], archive['part_ids']
        material = archive['material_gray']
        mesh_meta = json.loads(str(archive['metadata_json']))
    if (mesh_meta['objects'] != inputs['objects'] or mesh_meta['parts_sha256'] != hashes['parts']
            or mesh_meta['binding_sha256'] != hashes['binding']
            or not np.allclose(mesh_meta['registration'], inputs['registration'], rtol=0, atol=1e-12)):
        raise ValueError('Back mesh identity/registration differs from source binding')
    samples, chosen_triangles = bounded_samples(vertices, triangles, part_ids, material, target, args.spacing)
    target_vertices = vertices[np.unique(triangles[chosen_triangles])]
    bounds = np.stack((target_vertices.min(0), target_vertices.max(0))).astype(np.float64)
    new_q, new_scale, new_opacity = gaussian_attributes(samples['normals'], args.spacing,
        args.tangent_factor, args.normal_factor, args.opacity)
    region = np.all((xyz >= bounds[0]-.01) & (xyz <= bounds[1]+.01), axis=1)
    positions = np.flatnonzero(region)
    wanted = np.isin(reference_parts, target)
    if not wanted.any() or wanted.all():
        raise ValueError('Need target AND non-target rigid reference samples for boundary protection')
    target_tree, other_tree = cKDTree(reference[wanted]), cKDTree(reference[~wanted])
    d_target, _ = target_tree.query(xyz[region], workers=args.workers)
    d_other, _ = other_tree.query(xyz[region], workers=args.workers)
    selected, retirement_stats = geometry_retirement(xyz[region], d_target, d_other,
        scales[region].max(1), bounds)
    original = eligible[positions[selected]]
    retirement_stats['eligible_static_rows'] = len(eligible)
    retirement_stats['legacy_label_matches_target_diagnostic_only'] = int(np.isin(labels[region][selected], target).sum())
    del target_tree, other_tree, reference, reference_parts, xyz, scales
    completion = inputs['completion']
    with np.load(paths['completion_extension'], allow_pickle=False) as archive:
        suffix_parts, suffix_groups = archive['part_ids'], archive['group_ids']
    suffix_target = np.isin(suffix_parts, target)
    roots = np.flatnonzero(np.asarray(record['joint_structure']['parents']) == -1)
    if not np.isin(suffix_groups[suffix_target], roots).all():
        raise ValueError('Target completion rows unexpectedly have moving groups')
    dynamic = np.load(args.baseline/'dynamic_indices.npy', allow_pickle=False)
    retired, retired_suffix = assemble_retired(original, eligible, suffix_parts,
        completion['prefix_count'], cache['gaussian_count'], dynamic, target)
    source = PlyData.read(paths['source_ply'], mmap='r')['vertex'].data
    parent_gray = (source['f_dc_0']*C0+.5).copy()
    # Existing optimized colors remain an explicit input to the same-part prior.
    # This does not transfer visibility or pretend hidden colors were observed.
    with np.load(paths['color_residual'], allow_pickle=False) as archive:
        parent_gray[archive['indices']] += archive['delta_gray']
    with np.load(args.binding, allow_pickle=False) as archive:
        parent_labels = archive['labels']
    table, color_table = part_color_table(parent_gray[:completion['prefix_count']],
        parent_labels, part_ids, material, target)
    gray = shaded_gray(table[samples['part_ids']], samples['normals'])
    recheck_inputs(paths, hashes)
    count = len(samples['xyz'])
    metadata = {
        'format': FORMAT, 'source': str(paths['source_ply']), 'source_sha256': hashes['source_ply'],
        'source_ply_sha256': hashes['source_ply'], 'source_gaussian_count': cache['gaussian_count'],
        'input_paths': {k: str(v) for k, v in paths.items()}, 'input_sha256': hashes,
        'target_part_ids': target.tolist(), 'allowed_part_ids': target.tolist(),
        'target_part_names': [inputs['objects'][i] for i in target],
        'new_count': count, 'maximum_new_count': MAX_NEW, 'retired_count': len(retired),
        'retired_original_static_count': len(original), 'retired_compact_target_count': len(retired_suffix),
        'original_prefix_count': completion['prefix_count'], 'dynamic_rows_protected': len(dynamic),
        'new_geometry_bounds': bounds.tolist(), 'retirement_bounds': (bounds+np.array([[-.01], [.01]])).tolist(),
        'retirement_statistics': retirement_stats,
        'retirement_rule': {'maximum_target_distance': .01, 'nearest_part_margin': .002,
            'non_target_clearance': '3*maximum_original_scale + .001', 'roi_padding': .01,
            'original_binding_labels_used_to_retire': False,
            'distance_reference': 'All registered rigid rest samples. Nearest sample distances are not exact triangle distances; sample gaps can overestimate non-target clearance. Fresh multi-view actual renders are required.',
            'compact_suffix_rule': 'All and only existing completion rows for exact target_part_ids.'},
        'retired_target_distance': summarize(d_target[selected]), 'retired_other_distance': summarize(d_other[selected]),
        'sampling': samples['stats'], 'spacing': args.spacing, 'tangent_factor': args.tangent_factor,
        'normal_factor': args.normal_factor, 'opacity': args.opacity,
        'triangle_ids_convention': 'Global indices into exact input full-mesh triangles.',
        'quaternion_convention': 'Unit wxyz, local +Z maps to actual oriented triangle normal.',
        'dc_gray_convention': 'Grayscale image intensity in [0,1], not SH coefficients.',
        'color_prior_table': color_table, 'observed_color_count': 0, 'prior_color_count': count,
        'color_residual': cache['color_residual'],
        'color_residual_preservation': 'Input source and residual untouched; retained rows keep their exact runtime residual. New geometry uses same-part parent medians including the verified residual, not new observed texture.',
        'material_gray_unique': np.unique(material).tolist(),
        'new_gray_statistics': summarize(gray),
        'per_part': {str(int(p)): {'new_count': int((samples['part_ids']==p).sum()),
                                 'observed_color_count': 0} for p in target},
        'maximum_verified_motion_error': motion_error, 'input_identity_rechecked': True,
        'original_files_modified': False, 'all_motion_frozen': True, 'cpu_only': True,
        'unobserved_geometry_is_prior': True, 'unobserved_appearance_is_prior': True,
        'requires_fresh_raster_exposure_protection': True, 'approved_for_export': False,
        'seconds': time.monotonic()-started,
        'scope': 'UNAPPROVED static cabin/back geometry proposal only. No model, UI or audit is changed. '
                 'No moving parts, elastic bands or platform are targeted. The back has no observed ground truth; '
                 'new surfaces and all their colors are explicit mesh/same-part priors, not reconstructed unseen detail. '
                 'This raw retirement set is unsafe to deploy before fresh raster exposure protection and independent RGB/event/multiview tests.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, retired_indices=retired.astype(np.int32),
        new_xyz=samples['xyz'].astype(np.float32), new_quaternions=new_q, new_log_scale=new_scale,
        new_opacity_logit=new_opacity, new_dc_gray=gray.astype(np.float32), part_ids=samples['part_ids'],
        triangle_ids=samples['triangle_ids'], new_normals=samples['normals'],
        observed_color=np.zeros(count, bool), observation_support=np.zeros(count, np.int32),
        metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'new_count': count, 'retired_count': len(retired),
        'target_part_count': len(target), 'source_modified': False, 'approved_for_export': False}), flush=True)


if __name__ == '__main__':
    main()
