"""Read-only CPU diagnosis of far-from-prior, sparse/oversized static Gaussians.

No PLY, opacity, motion, service or event result is changed. Candidate indices
are hypotheses for a separate actual-Gaussian preview, not deletion authority.
Mesh distance is to ALL rigid reference samples, never just an assigned part.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


DEFAULT_CONSERVATIVE = {'mesh_distance': .025, 'knn8_distance': .008, 'maximum_scale': .010}
DEFAULT_BALANCED = {'mesh_distance': .015, 'knn8_distance': .005, 'maximum_scale': .006}
QUANTILES = (0., .01, .05, .25, .5, .75, .9, .95, .99, .999, 1.)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows, chunk=65536):
    digest = hashlib.sha256()
    for start in range(0, len(rows), chunk):
        digest.update(np.ascontiguousarray(rows[start:start+chunk]).tobytes())
    return digest.hexdigest()


def static_rows(total_count, prefix_count, retained_count, dynamic_indices, labels,
                deforming_parts, moving_parts=()):
    """Protect baked motion, elastic labels, rebuilt rows and compact suffix."""
    for name, value in (('total_count', total_count), ('prefix_count', prefix_count),
                        ('retained_count', retained_count)):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if not retained_count <= prefix_count <= total_count or total_count > np.iinfo(np.int32).max:
        raise ValueError('Invalid retained/parent/completed row boundaries')
    labels, dynamic_indices = np.asarray(labels), np.asarray(dynamic_indices)
    if (labels.shape != (prefix_count,) or labels.dtype.kind not in 'iu' or np.any(labels < 0)
            or dynamic_indices.ndim != 1 or dynamic_indices.dtype.kind not in 'iu'
            or np.any(dynamic_indices < 0) or np.any(dynamic_indices >= prefix_count)
            or np.any(dynamic_indices[1:] <= dynamic_indices[:-1])):
        raise ValueError('Invalid original parent binding labels or sorted dynamic rows')
    active = np.zeros(retained_count, bool)
    active[dynamic_indices[dynamic_indices < retained_count]] = True
    elastic = np.isin(labels[:retained_count], np.asarray(deforming_parts))
    moving = np.isin(labels[:retained_count], np.asarray(moving_parts))
    eligible = ~active & ~elastic & ~moving
    indices = np.flatnonzero(eligible).astype(np.int32)
    return indices, {
        'source_gaussians': total_count, 'parent_prefix_count': prefix_count,
        'retained_prefix_count': retained_count, 'rebuilt_rows_protected': prefix_count-retained_count,
        'compact_suffix_rows_protected': total_count-prefix_count,
        'all_baked_dynamic_rows_protected': int(len(dynamic_indices)),
        'baked_dynamic_in_retained': int(active.sum()),
        'elastic_labels_in_retained': int(elastic.sum()),
        'moving_rigid_labels_in_retained': int(moving.sum()),
        'additional_label_protected_rows': int((~active & (elastic | moving)).sum()),
        'eligible_static_rows': int(len(indices)),
    }


def validate_rules(conservative, balanced):
    for rule in (conservative, balanced):
        if set(rule) != {'mesh_distance', 'knn8_distance', 'maximum_scale'}:
            raise ValueError('Diagnostic rule must specify distance, 8NN radius and scale thresholds')
        for value in rule.values():
            if (type(value) not in (float, int) or not np.isfinite(value) or value <= 0):
                raise ValueError('Diagnostic thresholds must be finite positive scene-unit lengths')
    if any(balanced[key] > conservative[key] for key in conservative):
        raise ValueError('Balanced thresholds must not exceed conservative thresholds')


def classify(mesh_distance, knn8_distance, maximum_scale, conservative=None, balanced=None):
    conservative = DEFAULT_CONSERVATIVE if conservative is None else conservative
    balanced = DEFAULT_BALANCED if balanced is None else balanced
    validate_rules(conservative, balanced)
    mesh_distance, knn8_distance, maximum_scale = map(np.asarray,
                                                    (mesh_distance, knn8_distance, maximum_scale))
    if (mesh_distance.ndim != 1 or knn8_distance.shape != mesh_distance.shape
            or maximum_scale.shape != mesh_distance.shape
            or any(not np.isfinite(x).all() or np.any(x < 0)
                   for x in (mesh_distance, knn8_distance, maximum_scale))):
        raise ValueError('Diagnostic metrics must be matching finite, nonnegative vectors')
    result = {}
    for name, rule in (('conservative', conservative), ('balanced', balanced)):
        far = mesh_distance > rule['mesh_distance']
        sparse = knn8_distance > rule['knn8_distance']
        large = maximum_scale > rule['maximum_scale']
        result[name] = far & (sparse | large)
        result[name+'_far'] = far
        result[name+'_sparse'] = sparse
        result[name+'_large'] = large
    return result


def decode_attributes(vertices, indices):
    required = ('scale_0', 'scale_1', 'scale_2', 'opacity')
    if vertices.dtype.names is None or any(name not in vertices.dtype.names for name in required):
        raise ValueError('Expected Gaussian log-scale and opacity-logit PLY fields')
    log_scales = np.column_stack([vertices[name][indices] for name in required[:3]]).astype(np.float64)
    logits = vertices['opacity'][indices].astype(np.float64)
    if not np.isfinite(log_scales).all() or np.isnan(logits).any():
        raise ValueError('Invalid nonfinite Gaussian log-scales or NaN opacity logits')
    with np.errstate(over='ignore', under='ignore'):
        scales = np.exp(log_scales)
    if (not np.isfinite(scales).all() or np.any(scales <= 0)
            or np.any(scales > np.finfo(np.float32).max)):
        raise ValueError('Gaussian scales cannot be represented as positive finite float32')
    opacity = np.empty(len(logits), np.float64)
    positive = logits >= 0
    opacity[positive] = 1/(1+np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    opacity[~positive] = exponent/(1+exponent)
    return scales.astype(np.float32), opacity.astype(np.float32)


def neighborhood_metrics(xyz, reference_xyz, reference_parts, *, workers=2, chunk_size=65536, progress=None):
    """All-rigid nearest sample and eighth OTHER static-neighbor distance."""
    from scipy.spatial import cKDTree
    xyz, reference_xyz, reference_parts = map(np.asarray, (xyz, reference_xyz, reference_parts))
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 9
            or reference_xyz.ndim != 2 or reference_xyz.shape[1] != 3 or not len(reference_xyz)
            or reference_parts.shape != (len(reference_xyz),) or reference_parts.dtype.kind not in 'iu'
            or np.any(reference_parts < 0) or not np.isfinite(xyz).all() or not np.isfinite(reference_xyz).all()):
        raise ValueError('Need >=9 finite static points and nonempty labeled rigid reference samples')
    if type(workers) is not int or not 1 <= workers <= 8 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Invalid bounded CPU worker/chunk settings')
    distance = np.empty(len(xyz), np.float32)
    nearest_parts = np.empty(len(xyz), np.int32)
    tree = cKDTree(np.asarray(reference_xyz, np.float64))
    for start in range(0, len(xyz), chunk_size):
        end = min(start+chunk_size, len(xyz))
        value, nearest = tree.query(xyz[start:end], k=1, workers=workers)
        distance[start:end] = value
        nearest_parts[start:end] = reference_parts[nearest]
        if progress is not None:
            progress('all_rigid_mesh_distance', end, len(xyz))
    del tree
    tree = cKDTree(np.asarray(xyz, np.float64))
    radius = np.empty(len(xyz), np.float32)
    for start in range(0, len(xyz), chunk_size):
        end = min(start+chunk_size, len(xyz))
        # k=9 includes exactly one query row plus eight other rows. Coincident
        # points legitimately yield zero radius and are not classified sparse.
        distances, _ = tree.query(xyz[start:end], k=9, workers=workers)
        radius[start:end] = distances[:, 8]
        if progress is not None:
            progress('static_8nn_radius', end, len(xyz))
    return distance, radius, nearest_parts


def summarize(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'count': int(values.size), 'finite_count': int(finite.size),
            'nonfinite_count': int(values.size-finite.size),
            'quantiles': {str(q): float(value) for q, value in zip(QUANTILES, np.quantile(finite, QUANTILES))}
                         if len(finite) else {}}


def load_inputs(baseline, binding_path, parts_path):
    """Validate partition/provenance without CUDA, Torch or a mutable PLY copy."""
    from plyfile import PlyData
    cache_path = baseline/'cache.json'
    cache = json.loads(cache_path.read_text())
    contract = cache.get('multiview_completion')
    if not isinstance(contract, dict) or contract.get('format') != 'lego_multiview_prior_completion_v1':
        raise ValueError('Expected the compact multiview-completed baseline')
    paths = {'baseline_cache': cache_path, 'source_ply': Path(cache['source']),
             'completion_report': Path(contract['report']), 'completion_extension': Path(contract['extension']),
             'binding': binding_path, 'parts': parts_path, 'dynamic_indices': baseline/'dynamic_indices.npy'}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if (hashes['source_ply'] != cache.get('source_sha256')
            or hashes['completion_report'] != contract.get('report_sha256')
            or hashes['completion_extension'] != contract.get('extension_sha256')):
        raise ValueError('Completed model source/report/extension checksum mismatch')
    completion = json.loads(paths['completion_report'].read_text())
    paths['parent_report'] = Path(completion['parent_report'])
    paths['parent_source'] = Path(completion['parent_source'])
    for name in ('parent_report', 'parent_source'):
        hashes[name] = sha256(paths[name])
        if hashes[name] != completion.get(name+'_sha256'):
            raise ValueError('Completion parent checksum mismatch: '+name)
    parent = json.loads(paths['parent_report'].read_text())
    if (parent.get('format') != 'lego_prior_surface_rebuild_v1'
            or parent.get('binding_sha256') != hashes['binding']
            or parent.get('parts_sha256') != hashes['parts']
            or completion.get('parts_sha256') != hashes['parts']
            or parent.get('rebuilt_sha256') != hashes['parent_source']
            or completion.get('completed_sha256') != hashes['source_ply']
            or completion.get('gaussian_count') != cache.get('gaussian_count')):
        raise ValueError('Parent binding, rigid reference parts or source provenance differs')
    with np.load(binding_path, allow_pickle=False) as archive:
        registration, labels, dynamic = archive['registration'], archive['labels'], archive['dynamic_indices']
        binding_meta = json.loads(str(archive['metadata_json']))
    prefix, retained = parent['gaussian_count'], parent['retained_gaussian_count']
    if (completion.get('prefix_count') != prefix or cache['gaussian_count']-prefix != completion.get('added_gaussians')
            or prefix-retained != parent.get('rebuilt_gaussians')
            or binding_meta.get('source_sha256') != hashes['parent_source']
            or not np.array_equal(dynamic, np.load(paths['dynamic_indices'], allow_pickle=False))
            or len(dynamic) != cache.get('dynamic_count')):
        raise ValueError('Retained/rebuilt/compact or baked-motion partition differs')
    if (registration.shape != (4, 4) or not np.isfinite(registration).all()
            or not np.allclose(registration[3], [0, 0, 0, 1], rtol=0, atol=1e-8)):
        raise ValueError('Invalid source-to-Gaussian registration')
    singular = np.linalg.svd(registration[:3, :3], compute_uv=False)
    if (singular.min() <= 0 or np.linalg.det(registration[:3, :3]) <= 0
            or not np.allclose(singular, singular.mean(), rtol=1e-5, atol=1e-10)):
        raise ValueError('Expected a proper uniform-similarity reference registration')
    with np.load(parts_path, allow_pickle=False) as archive:
        rest, reference_parts = archive['rest_points'], archive['point_parts']
        matrices = archive['object_matrices'].astype(np.float64)
        part_meta = json.loads(str(archive['metadata_json']))
    objects, deforming = part_meta['objects'], part_meta['deforming_objects']
    if (rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all()
            or reference_parts.shape != (len(rest),) or reference_parts.dtype.kind not in 'iu'
            or np.any(reference_parts < 0) or np.any(reference_parts >= len(objects))
            or labels.shape != (prefix,) or np.any(labels < 0) or np.any(labels >= len(objects))
            or matrices.shape != (cache['frame_count'], len(objects), 4, 4) or not np.isfinite(matrices).all()):
        raise ValueError('Invalid canonical reference, part identities or source motion schema')
    delta = matrices @ np.linalg.inv(matrices[0])
    moving = np.max(np.abs(delta-np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[deforming] = False
    rows, counts = static_rows(cache['gaussian_count'], prefix, retained, dynamic, labels,
                               deforming, np.flatnonzero(moving))
    rigid_reference = ~np.isin(reference_parts, deforming)
    source_reference = rest[rigid_reference].astype(np.float64)
    reference_xyz = source_reference @ registration[:3, :3].T + registration[:3, 3]
    reference_parts = reference_parts[rigid_reference].astype(np.int32)
    del rest, source_reference, matrices, delta
    vertices = PlyData.read(paths['source_ply'], mmap='r')['vertex'].data
    parent_vertices = PlyData.read(paths['parent_source'], mmap='r')['vertex'].data
    if (len(vertices) != cache['gaussian_count'] or len(parent_vertices) != prefix
            or vertices.dtype != parent_vertices.dtype
            or rows_sha256(vertices[:prefix]) != rows_sha256(parent_vertices)
            or rows_sha256(vertices[:retained]) != parent.get('frozen_retained_vertex_sha256')):
        raise ValueError('Completed source changed parent/frozen retained Gaussian rows')
    xyz = np.column_stack([vertices[name][rows] for name in ('x', 'y', 'z')])
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite retained static positions')
    scales, opacity = decode_attributes(vertices, rows)
    opacity_logits = vertices['opacity'][rows]
    inputs = {'cache': cache, 'completion': completion, 'paths': paths, 'hashes': hashes,
              'objects': objects, 'counts': counts, 'registration': registration,
              'registration_scale': float(singular.mean()),
              'excluded_deforming_parts': list(deforming), 'reference_rigid_samples': len(reference_xyz),
              'reference_rigid_part_count': int(len(np.unique(reference_parts))),
              'source_reference_samples': int(rigid_reference.size),
              'opacity_logit_saturation': {
                  'positive_infinity_exact_opacity_one': int(np.isposinf(opacity_logits).sum()),
                  'negative_infinity_exact_opacity_zero': int(np.isneginf(opacity_logits).sum()),
                  'note': 'Infinite logits activate to finite opacity 1 or 0; NaN logits are rejected. No source values are changed.'}}
    return inputs, rows, xyz, labels[rows], reference_xyz, reference_parts, scales, opacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'binding', 'parts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name, defaults in (('conservative', DEFAULT_CONSERVATIVE), ('balanced', DEFAULT_BALANCED)):
        parser.add_argument('--'+name+'-distance', type=float, default=defaults['mesh_distance'])
        parser.add_argument('--'+name+'-knn8', type=float, default=defaults['knn8_distance'])
        parser.add_argument('--'+name+'-scale', type=float, default=defaults['maximum_scale'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=65536)
    parser.add_argument('--diagnostic-limit', type=int, default=20000,
                        help='Limit detailed rows only; candidate indices/counts are NEVER truncated')
    parser.add_argument('--surface-outlier-distance', type=float, default=.005,
                        help='Geometry-only hypotheses for separate full-frame exposure protection; never auto-delete')
    args = parser.parse_args()
    rules = {name: {'mesh_distance': getattr(args, name+'_distance'),
                    'knn8_distance': getattr(args, name+'_knn8'),
                    'maximum_scale': getattr(args, name+'_scale')}
             for name in ('conservative', 'balanced')}
    validate_rules(rules['conservative'], rules['balanced'])
    if not 1 <= args.workers <= 8 or not 1024 <= args.chunk_size <= 262144 or not 1 <= args.diagnostic_limit <= 100000:
        raise ValueError('Invalid bounded CPU/output parameters')
    if not np.isfinite(args.surface_outlier_distance) or args.surface_outlier_distance <= 0:
        raise ValueError('Surface-outlier distance must be finite and positive')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ('baseline', 'binding', 'parts'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    started = time.monotonic()
    inputs, rows, xyz, labels, reference_xyz, reference_parts, scales, opacity = load_inputs(
        args.baseline, args.binding, args.parts)
    print(json.dumps({'stage': 'verified_static_partition', **inputs['counts'],
                      'rigid_reference_samples': inputs['reference_rigid_samples']}), flush=True)
    def progress(stage, done, count):
        if done == count or (done//args.chunk_size) % 8 == 0:
            print(json.dumps({'stage': stage, 'done': done, 'total': count}), flush=True)
    distance, radius, nearest_parts = neighborhood_metrics(
        xyz, reference_xyz, reference_parts, workers=args.workers, chunk_size=args.chunk_size, progress=progress)
    del reference_xyz, reference_parts
    maximum_scale = scales.max(axis=1)
    masks = classify(distance, radius, maximum_scale, rules['conservative'], rules['balanced'])
    surface_outliers = distance > args.surface_outlier_distance
    with np.errstate(divide='ignore', over='ignore'):
        density = 8/((4/3)*np.pi*radius.astype(np.float64)**3)
    metrics = {'mesh_distance': distance, 'knn8_distance': radius, 'knn8_ball_density': density,
               'maximum_scale': maximum_scale, 'minimum_scale': scales.min(axis=1), 'opacity': opacity}
    variant_stats, per_part = {}, []
    for name in ('conservative', 'balanced'):
        chosen = masks[name]
        variant_stats[name] = {
            'candidate_count': int(chosen.sum()), 'fraction_of_eligible_static': float(chosen.mean()),
            'far_count': int(masks[name+'_far'].sum()),
            'far_and_sparse_count': int((masks[name+'_far'] & masks[name+'_sparse']).sum()),
            'far_and_oversized_count': int((masks[name+'_far'] & masks[name+'_large']).sum()),
            'far_but_neither_sparse_nor_oversized': int((masks[name+'_far'] & ~chosen).sum()),
            'metrics': {key: summarize(value[chosen]) for key, value in metrics.items()},
        }
    for part in np.unique(labels):
        part_rows = labels == part
        per_part.append({'part_id': int(part), 'name': inputs['objects'][part],
                         'eligible_static_rows': int(part_rows.sum()),
                         'conservative_count': int((part_rows & masks['conservative']).sum()),
                         'balanced_count': int((part_rows & masks['balanced']).sum())})
    # All selected indices are retained. Only supplementary point-by-point
    # metrics are capped, deterministically by combined geometric risk.
    diagnostic = np.flatnonzero(masks['balanced'])
    score = (distance[diagnostic]/rules['balanced']['mesh_distance']
             * np.maximum(radius[diagnostic]/rules['balanced']['knn8_distance'],
                          maximum_scale[diagnostic]/rules['balanced']['maximum_scale']))
    order = np.argsort(-score, kind='stable')[:args.diagnostic_limit]
    diagnostic, score = diagnostic[order], score[order]
    for name, path in inputs['paths'].items():
        if sha256(path) != inputs['hashes'][name]:
            raise ValueError('Input changed during diagnosis: '+name)
    metadata = {
        'format': 'lego_static_sparse_point_diagnosis_v1', 'source': str(inputs['paths']['source_ply']),
        'baseline': str(args.baseline), 'candidate_indices_default': 'conservative_indices',
        'artifact_sha256': inputs['hashes'], 'input_paths': {key: str(value) for key, value in inputs['paths'].items()},
        'counts': inputs['counts'], 'thresholds_scene_units': rules,
        'selection_expression': 'mesh_distance > threshold AND (8NN_radius > threshold OR maximum_Gaussian_sigma > threshold)',
        'surface_outlier_hypotheses': {
            'distance_threshold_scene_units': args.surface_outlier_distance,
            'count': int(surface_outliers.sum()),
            'distance_exceedance_counts': {str(threshold): int((distance > threshold).sum())
                                           for threshold in (.0025, .005, .01, .02)},
            'scope': 'Geometry-distance-only candidate subset, NOT classified as noise. No sparse/large-scale '
                     'gate is applied here. Needs separate actual-Gaussian original-camera exposure protection '
                     'over all 400 frames, and new RGB/event/multiview checks before any suppression trial.'},
        'density_neighborhood': 'Eight OTHER retained static Gaussian centers; dynamic, rebuilt and compact suffix rows excluded.',
        'reference_geometry': 'All registered rigid rest-surface samples across every source part, including top/base/static parts; elastic parts excluded.',
        'reference_sample_count': inputs['reference_rigid_samples'],
        'reference_part_count': inputs['reference_rigid_part_count'],
        'excluded_elastic_parts': inputs['excluded_deforming_parts'],
        'registration': inputs['registration'].tolist(), 'registration_uniform_scale': inputs['registration_scale'],
        'completion_mesh_spacing_scene_units': inputs['completion'].get('parameters', {}).get('spacing'),
        'scene_length_note': 'Distances and scales use canonical Gaussian scene units, not claimed physical meters. Gaussian scale means exp(log_scale), i.e. standard deviation, not 3-sigma radius.',
        'all_eligible_metrics': {key: summarize(value) for key, value in metrics.items()},
        'original_static_opacity_logit_saturation': inputs['opacity_logit_saturation'],
        'variants': variant_stats, 'per_part_counts': per_part,
        'diagnostic_detail_rows': int(len(diagnostic)), 'diagnostic_detail_limit': args.diagnostic_limit,
        'diagnostic_detail_order': 'Descending (mesh_distance/balanced_distance) * max(8NN_radius/balanced_knn8, maximum_scale/balanced_scale); stable source-row ties.',
        'candidate_indices_truncated': False, 'diagnostic_details_truncated': int(masks['balanced'].sum()) > len(diagnostic),
        'inputs_rechecked_after_diagnosis': True, 'cpu_only': True, 'source_files_modified': False,
        'approved_for_cleanup': False, 'event_acceptance_changed': False,
        'seconds': time.monotonic()-started,
        'scope': 'Read-only geometric hypotheses for actual-Gaussian previews. Low density alone NEVER selects a row. '
                 'Opacity is reported, not used as a deletion trigger. Source registration, finite reference sampling '
                 'and the Blender geometry prior can be wrong; distance is to sampled mesh points, not exact triangles '
                 'or observed unseen truth. Every dynamic/band/rebuilt/completion row is protected. No PLY is changed '
                 'and no event approval, all-view completeness or safe-deletion claim is made.',
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'diagnosis.npz',
        indices=rows[masks['conservative']], conservative_indices=rows[masks['conservative']],
        balanced_indices=rows[masks['balanced']], surface_outlier_indices=rows[surface_outliers],
        diagnostic_indices=rows[diagnostic],
        xyz=xyz[diagnostic], labels=labels[diagnostic].astype(np.int32),
        nearest_reference_part=nearest_parts[diagnostic], mesh_distance=distance[diagnostic],
        knn8_distance=radius[diagnostic], knn8_ball_density=density[diagnostic],
        scales=scales[diagnostic], opacity=opacity[diagnostic], rank_score=score.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    (args.output/'diagnosis.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output/'diagnosis.npz'), 'eligible_static': len(rows),
                      'conservative_count': variant_stats['conservative']['candidate_count'],
                      'balanced_count': variant_stats['balanced']['candidate_count'],
                      'surface_outlier_count': int(surface_outliers.sum()),
                      'seconds': metadata['seconds'], 'source_modified': False}), flush=True)


if __name__ == '__main__':
    main()
