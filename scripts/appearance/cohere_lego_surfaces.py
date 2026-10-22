"""Propose whole-surface Gaussian ownership using conservative tangent adjacency.

This changes only binding arrays, not Gaussian geometry or appearance. Connected
surfaces are geometric hypotheses, not known physical parts: touching coplanar
parts can connect, and a sharp corner can split one part. Large or ambiguous
components retain their prior ownership. Raw-event refinement/audit is required.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.motion.rebind_lego_coherent import transform, validate_correspondence


def disk_normals(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=1)
    if not np.isfinite(quaternion).all() or np.any(norm < 1e-12):
        raise ValueError('Invalid Gaussian rotations')
    w, x, y, z = (quaternion / norm[:, None]).T
    return np.column_stack((2 * (x * z + w * y), 2 * (y * z - w * x),
                            1 - 2 * (x * x + y * y))).astype(np.float32)


def opacity_weights(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if np.isnan(logits).any():
        raise ValueError('NaN opacity')
    result = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    result[np.isneginf(logits)] = 0
    return result


def tangent_edges(xyz, normals, radius, rows, neighbors, distances, excluded,
                  radius_factor, max_radius, normal_cos, plane_ratio, plane_slack):
    """Symmetric edge predicate; neither endpoint may be an elastic-band row."""
    rows = np.asarray(rows)
    difference = xyz[neighbors] - xyz[rows, None]
    normal_a, normal_b = normals[rows, None], normals[neighbors]
    orientation = np.abs(np.sum(normal_a * normal_b, axis=-1))
    plane = np.maximum(np.abs(np.sum(difference * normal_a, axis=-1)),
                       np.abs(np.sum(difference * normal_b, axis=-1)))
    reach = np.minimum(max_radius, radius_factor * np.maximum(radius[rows, None], radius[neighbors]))
    return ((neighbors != rows[:, None]) & (distances <= reach)
            & (orientation >= normal_cos) & (plane <= plane_ratio * distances + plane_slack)
            & ~excluded[rows, None] & ~excluded[neighbors])


def component_vote(components, groups, weights, minimum_size, maximum_size, majority):
    """Weighted group consensus; ambiguous/oversized components are not changed."""
    count = int(np.max(components)) + 1
    group_count = int(np.max(groups)) + 1
    sizes = np.bincount(components, minlength=count)
    votes = np.bincount(components * group_count + groups, weights=weights,
                        minlength=count * group_count).reshape(count, group_count)
    winner = np.argmax(votes, axis=1)
    total = votes.sum(axis=1)
    purity = votes[np.arange(count), winner] / np.maximum(total, 1e-12)
    eligible = ((sizes >= minimum_size) & (sizes <= maximum_size)
                & (purity >= majority) & (total > 1e-8))
    mixed = np.count_nonzero(votes > 1e-8, axis=1) > 1
    return sizes, winner, purity, eligible, mixed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--source-ply', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--neighbors', type=int, default=12)
    p.add_argument('--radius-factor', type=float, default=3.)
    p.add_argument('--max-radius', type=float, default=.006)
    p.add_argument('--normal-cos', type=float, default=.94)
    p.add_argument('--plane-ratio', type=float, default=.3)
    p.add_argument('--plane-slack', type=float, default=.0002)
    p.add_argument('--minimum-component', type=int, default=64)
    p.add_argument('--maximum-component', type=int, default=250000,
                   help='Larger components are rejected, not split or silently merged')
    p.add_argument('--majority', type=float, default=.7)
    p.add_argument('--motion-cluster-distance', type=float, default=.002)
    p.add_argument('--max-rebind-distance', type=float,
                   help='Defaults to the input binding attachment gate; farther proposals are retained unchanged')
    p.add_argument('--chunk-size', type=int, default=65536)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    positive = ('radius_factor', 'max_radius', 'motion_cluster_distance')
    if any(not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0 for key in positive):
        p.error('Distance/radius parameters must be finite and positive')
    if not 0 <= args.normal_cos <= 1 or not .5 < args.majority <= 1:
        p.error('normal-cos must be in [0,1] and majority in (.5,1]')
    if any(not np.isfinite(v) or v < 0 for v in (args.plane_ratio, args.plane_slack)):
        p.error('Plane thresholds must be finite and nonnegative')
    if args.neighbors < 2 or args.chunk_size < 1 or args.minimum_component < 2 or args.maximum_component < args.minimum_component:
        p.error('Invalid neighbor/chunk/component sizes')
    if args.max_rebind_distance is not None and (not np.isfinite(args.max_rebind_distance) or args.max_rebind_distance <= 0):
        p.error('max-rebind-distance must be finite and positive')
    from plyfile import PlyData
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    with np.load(args.parts, allow_pickle=False) as archive:
        parts = {key: archive[key] for key in archive.files}
    with np.load(args.binding, allow_pickle=False) as archive:
        binding = {key: archive[key] for key in archive.files}
    prior = json.loads(str(binding['metadata_json']))
    meta = json.loads(str(parts['metadata_json']))
    v = PlyData.read(args.source_ply, mmap=True)['vertex'].data
    xyz = np.column_stack([v[key] for key in ('x', 'y', 'z')])
    if len(xyz) != len(binding['labels']) or not np.isfinite(xyz).all():
        raise ValueError('Source point count or coordinates mismatch')
    if len(xyz) < 2 or len(xyz) >= np.iinfo(np.int32).max:
        raise ValueError('Unsupported point count')
    normals = disk_normals(np.column_stack([v[f'rot_{i}'] for i in range(4)]))
    scales = np.exp(np.column_stack([v[f'scale_{i}'] for i in range(3)]))
    if not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError('Invalid Gaussian scales')
    # This dataset uses the third local axis as the thin surface-normal axis.
    disk = scales[:, 2] <= .1 * np.minimum(scales[:, 0], scales[:, 1])
    radius = np.maximum(scales[:, 0], scales[:, 1])
    weights = opacity_weights(v['opacity'])
    original_labels = binding['labels'].copy()
    elastic = np.isin(original_labels, meta['deforming_objects'])
    excluded = elastic | ~disk
    weights[excluded] = 0
    matrices = parts['object_matrices'].astype(np.float64)
    delta = matrices @ np.linalg.inv(matrices[0])
    frame_ids = np.unique(np.linspace(0, len(delta) - 1, min(17, len(delta))).astype(int))
    features = delta[frame_ids, :, :3, :].transpose(1, 0, 2, 3).reshape(len(matrices[0]), -1)
    group = fclusterdata(features, args.motion_cluster_distance, criterion='distance', method='complete') - 1
    point_group = group[original_labels]
    tree = cKDTree(xyz)
    edge_rows, edge_cols = [], []
    for start in range(0, len(xyz), args.chunk_size):
        rows = np.arange(start, min(start + args.chunk_size, len(xyz)), dtype=np.int32)
        distances, neighbors = tree.query(xyz[rows], k=min(args.neighbors, len(xyz)), workers=args.workers)
        keep = tangent_edges(xyz, normals, radius, rows, neighbors, distances, excluded,
                             args.radius_factor, args.max_radius, args.normal_cos,
                             args.plane_ratio, args.plane_slack)
        local, slot = np.nonzero(keep)
        edge_rows.append(rows[local]); edge_cols.append(neighbors[local, slot].astype(np.int32))
        if start // args.chunk_size % 8 == 0:
            print('Built tangent adjacency', min(start + args.chunk_size, len(xyz)), '/', len(xyz), flush=True)
    del tree
    source = np.concatenate(edge_rows); target = np.concatenate(edge_cols)
    edge_count = len(source)
    del edge_rows, edge_cols
    graph = coo_matrix((np.ones(edge_count, dtype=np.bool_), (source, target)),
                       shape=(len(xyz), len(xyz))).tocsr()
    del source, target
    component_count, components = connected_components(graph, directed=False)
    del graph
    sizes, winner, purity, eligible, mixed = component_vote(
        components, point_group, weights, args.minimum_component, args.maximum_component, args.majority)
    proposed_group = winner[components]
    change = eligible[components] & ~excluded & (proposed_group != point_group)
    labels = original_labels.copy()
    nearest = binding['nearest_vertex'].copy()
    distance = binding['distance'].copy()
    surface = transform(parts['rest_points'], binding['registration'])
    limit = args.max_rebind_distance if args.max_rebind_distance is not None else prior['max_attachment_distance']
    accepted = np.zeros(len(xyz), dtype=bool)
    for g in np.unique(proposed_group[change]):
        rows = np.flatnonzero(change & (proposed_group == g))
        points = np.flatnonzero((group[parts['point_parts']] == g)
                               & ~np.isin(parts['point_parts'], meta['deforming_objects']))
        if not len(points):
            continue
        dist, near = cKDTree(surface[points]).query(xyz[rows], workers=args.workers)
        keep = dist <= limit
        rows = rows[keep]
        nearest[rows] = points[near[keep]]
        labels[rows] = parts['point_parts'][nearest[rows]]
        distance[rows] = dist[keep]
        accepted[rows] = True
    validate_correspondence(labels, nearest, parts, meta)
    if not (np.array_equal(labels[elastic], original_labels[elastic])
            and np.array_equal(nearest[elastic], binding['nearest_vertex'][elastic])
            and np.array_equal(distance[elastic], binding['distance'][elastic])):
        raise AssertionError('Elastic correspondences changed')
    moving = np.max(np.abs(delta - np.eye(4)), axis=(0, 2, 3)) > 1e-4
    moving[meta['deforming_objects']] = True
    maximum_distance = max(prior['max_attachment_distance'], limit)
    dynamic = moving[labels] & (distance <= maximum_distance)
    previous_dynamic = np.zeros(len(xyz), dtype=bool)
    previous_dynamic[binding['dynamic_indices']] = True
    dynamic[elastic] = previous_dynamic[elastic]
    dynamic_indices = np.flatnonzero(dynamic).astype(np.int32)
    counts = np.bincount(labels, minlength=len(moving))
    dynamic_counts = np.bincount(labels[dynamic_indices], minlength=len(moving))
    largest = np.argsort(sizes)[-20:][::-1]
    summary = {
        'component_count': int(component_count), 'directed_candidate_edges': edge_count,
        'size_percentiles': np.percentile(sizes, [50, 90, 99, 100]).tolist(),
        'mixed_components': int(mixed.sum()), 'eligible_components': int(eligible.sum()),
        'eligible_mixed_components': int((eligible & mixed).sum()),
        'oversized_components_retained': int((sizes > args.maximum_component).sum()),
        'oversized_rows_retained': int(sizes[sizes > args.maximum_component].sum()),
        'ambiguous_components_retained': int(((sizes >= args.minimum_component) & (purity < args.majority)).sum()),
        'proposed_changed_rows': int(change.sum()), 'accepted_changed_rows': int(accepted.sum()),
        'distance_rejected_rows': int((change & ~accepted).sum()),
        'elastic_rows_excluded': int(elastic.sum()), 'non_disk_rows_excluded': int((~disk).sum()),
        'largest_components': [{'component': int(i), 'size': int(sizes[i]), 'group': int(winner[i]),
                                'weighted_purity': float(purity[i]), 'eligible': bool(eligible[i])}
                               for i in largest]}
    thresholds = {key: getattr(args, key) for key in ('neighbors', 'radius_factor', 'max_radius', 'normal_cos',
                  'plane_ratio', 'plane_slack', 'minimum_component', 'maximum_component', 'majority',
                  'motion_cluster_distance', 'chunk_size')}
    thresholds['max_rebind_distance'] = limit
    report = {**prior, 'method': 'tangent-connected surface components; opacity-weighted motion-group consensus',
              'prior_binding': str(args.binding), 'source': str(args.source_ply), 'parts': str(args.parts),
              'surface_coherence': summary, 'surface_coherence_thresholds': thresholds,
              'motion_groups': int(group.max() + 1), 'changed_gaussians': int(accepted.sum()),
              'max_attachment_distance': maximum_distance, 'gaussian_count': len(xyz),
              'dynamic_count': len(dynamic_indices), 'static_count': len(xyz) - len(dynamic_indices),
              'unattached': int((distance > maximum_distance).sum()),
              'distance_percentiles': dict(zip(['p50', 'p90', 'p95', 'p99'],
                                               np.percentile(distance, [50, 90, 95, 99]).tolist())),
              'part_gaussians': {name: int(dynamic_counts[i]) for i, name in enumerate(meta['objects']) if moving[i]},
              'all_part_gaussians': {name: int(counts[i]) for i, name in enumerate(meta['objects'])},
              'appearance_and_rest_geometry_unchanged': True, 'event_approved': False,
              'status': 'INITIALIZATION_ONLY_NOT_APPROVED',
              'limitations': 'Geometric connected faces are not guaranteed physical parts. Touching coplanar parts '
                             'can merge, sharp corners can split one part. Oversized/ambiguous components and '
                             'elastic bands retain prior binding. No event approval; inspect and audit before use.'}
    binding.update(labels=labels.astype(np.int16), nearest_vertex=nearest.astype(np.int32),
                   distance=distance.astype(np.float32), dynamic_indices=dynamic_indices,
                   metadata_json=np.asarray(json.dumps(report)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **binding)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
