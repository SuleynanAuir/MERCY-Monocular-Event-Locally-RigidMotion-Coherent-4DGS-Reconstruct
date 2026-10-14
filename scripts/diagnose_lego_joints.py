"""Read-only joint candidates from source rigid motion and source surfaces.

This does not infer physical joints from nearby Gaussian splats. It solves the
full-timeline equation delta_g(t) c == delta_h(t) c in the source rest-world
coordinate system, then rejects fits unsupported by nearby source geometry.
An accepted edge remains a candidate, not proof of an actual mechanical joint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def fit_shared_pivot(first, second, rank_rtol=1e-5, rank_atol=1e-7):
    """Return the least-squares invariant point plus its unobservable subspace."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[1:] != (4, 4):
        raise ValueError('Joint fitting requires matching [frames,4,4] transforms')
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Non-finite transform')
    matrix = (first[:, :3, :3] - second[:, :3, :3]).reshape(-1, 3)
    target = (second[:, :3, 3] - first[:, :3, 3]).reshape(-1)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    threshold = max(rank_atol, rank_rtol * float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    point = vt[:rank].T @ ((u[:, :rank].T @ target) / singular[:rank])
    nullspace = vt[rank:].T.copy()
    # Stable signs make axis output reproducible across LAPACK implementations.
    for column in range(nullspace.shape[1]):
        if nullspace[np.argmax(abs(nullspace[:, column])), column] < 0:
            nullspace[:, column] *= -1
    residual = (matrix @ point - target).reshape(-1, 3)
    return {'point': point, 'nullspace': nullspace, 'rank': rank,
            'singular_values': singular, 'rank_threshold': threshold,
            'max_residual': float(np.linalg.norm(residual, axis=1).max()),
            'rms_residual': float(np.sqrt(np.square(residual).sum(1).mean()))}


def motion_groups(delta, eligible, rotation_tolerance, translation_tolerance):
    """Deterministic complete-link groups using every frame, including static."""
    groups = []
    for label in eligible:
        for members in groups:
            difference = delta[:, members] - delta[:, label, None]
            if (np.max(abs(difference[:, :, :3, :3])) <= rotation_tolerance
                    and np.max(np.linalg.norm(difference[:, :, :3, 3], axis=-1)) <= translation_tolerance):
                members.append(int(label))
                break
        else:
            groups.append([int(label)])
    return groups


class SurfaceSearch:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        self.tree = None
        # Small synthetic tests do not require a local SciPy installation.
        if len(self.points) > 256:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.points)

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.tree is not None:
            return self.tree.query(points)
        squared = np.square(points[:, None] - self.points[None]).sum(-1)
        rows = squared.argmin(1)
        return np.sqrt(squared[np.arange(len(points)), rows]), rows


def contact_midpoints(first, second, distance, max_samples):
    """Conservative sampled source-surface contacts, never Gaussian contacts."""
    midpoints, separations = [], []
    for source, target in ((first, second), (second, first)):
        lower, upper = target.points.min(0)-distance, target.points.max(0)+distance
        points = source.points[np.all((source.points >= lower) & (source.points <= upper), axis=1)]
        if len(points) > max_samples:
            points = points[np.linspace(0, len(points)-1, max_samples).astype(int)]
        if not len(points):
            continue
        gap, nearest = target.query(points)
        selected = gap <= distance
        midpoints.extend((.5*(points[selected]+target.points[nearest[selected]])).tolist())
        separations.extend(gap[selected].tolist())
    return np.asarray(midpoints, dtype=np.float64).reshape(-1, 3), separations


def graph_structure(group_count, edges, static_groups):
    """A spanning forest plus all omitted loop-closure edges; no loops hidden."""
    representative = list(range(group_count))

    def root(group):
        while representative[group] != group:
            representative[group] = representative[representative[group]]
            group = representative[group]
        return group

    tree, loops = [], []
    ordered = sorted(range(len(edges)), key=lambda i: (edges[i]['max_member_residual_bound'],
                                                       edges[i]['contact_min_distance'], i))
    for index in ordered:
        first, second = edges[index]['groups']
        a, b = root(first), root(second)
        if a == b:
            loops.append(index)
        else:
            representative[b] = a
            tree.append(index)
    components = {}
    for group in range(group_count):
        components.setdefault(root(group), []).append(group)
    result = []
    for members in components.values():
        anchors = sorted(set(members) & set(static_groups))
        result.append({'groups': members, 'static_groups': anchors,
                       'suggested_root': anchors[0] if anchors else min(members),
                       'unanchored': not bool(anchors)})
    return {'components': result, 'spanning_forest_edge_indices': tree,
            'loop_closure_edge_indices': loops,
            'cycle_rank': len(loops),
            'warning': 'A kinematic tree does not enforce the listed loop-closure edges.'}


def diagnose(parts, *, rotation_tolerance=1e-5, translation_tolerance=None,
             residual_tolerance=None, contact_distance=None, pivot_distance=None,
             max_surface_samples=6000, rank_rtol=1e-5, rank_atol=1e-7):
    matrices = np.asarray(parts['object_matrices'], dtype=np.float64)
    points = np.asarray(parts['rest_points'], dtype=np.float64)
    labels = np.asarray(parts['point_parts'])
    metadata = json.loads(str(parts['metadata_json']))
    names = metadata['objects']
    if (matrices.ndim != 4 or matrices.shape[2:] != (4, 4) or len(matrices) < 2
            or matrices.shape[1] != len(names) or points.ndim != 2 or points.shape[1:] != (3,)
            or not len(points) or labels.shape != (len(points),) or labels.dtype.kind not in 'iu'
            or labels.min() < 0 or labels.max() >= len(names)
            or not np.isfinite(matrices).all() or not np.isfinite(points).all()):
        raise ValueError('Invalid source archive geometry/matrices/metadata')
    if not np.allclose(matrices[:, :, 3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError('Source matrices must be affine')
    diagonal = float(np.linalg.norm(points.max(0)-points.min(0)))
    if diagonal <= 0:
        raise ValueError('Degenerate source geometry')
    translation_tolerance = diagonal*1e-6 if translation_tolerance is None else translation_tolerance
    residual_tolerance = max(diagonal*1e-5, 1e-7) if residual_tolerance is None else residual_tolerance
    contact_distance = diagonal*.01 if contact_distance is None else contact_distance
    pivot_distance = diagonal*.015 if pivot_distance is None else pivot_distance
    thresholds = {'rotation_tolerance': rotation_tolerance, 'translation_tolerance': translation_tolerance,
                  'residual_tolerance': residual_tolerance, 'contact_distance': contact_distance,
                  'pivot_surface_distance': pivot_distance, 'rank_rtol': rank_rtol, 'rank_atol': rank_atol}
    if any(not np.isfinite(v) or v <= 0 for v in thresholds.values()) or max_surface_samples < 1:
        raise ValueError('Thresholds/sample count must be finite and positive')
    delta = matrices @ np.linalg.inv(matrices[0])
    excluded, eligible = [], []
    deforming = set(metadata.get('deforming_objects', []))
    for label in range(len(names)):
        rotation = delta[:, label, :3, :3]
        if label in deforming:
            reason = 'deforming_source_object'
        elif not np.any(labels == label):
            reason = 'no_source_surface'
        elif (np.max(abs(rotation @ rotation.transpose(0, 2, 1)-np.eye(3))) > 1e-4
              or np.max(abs(np.linalg.det(rotation)-1)) > 1e-4):
            reason = 'nonrigid_object_delta'
        else:
            eligible.append(label)
            continue
        excluded.append({'object': label, 'name': names[label], 'reason': reason})
    groups = motion_groups(delta, eligible, rotation_tolerance, translation_tolerance)
    surfaces, group_records, static_groups = [], [], []
    for group, members in enumerate(groups):
        surface = points[np.isin(labels, members)]
        surfaces.append(SurfaceSearch(surface))
        transform = delta[:, members[0]]
        is_static = (np.max(abs(transform[:, :3, :3]-np.eye(3))) <= rotation_tolerance
                     and np.max(np.linalg.norm(transform[:, :3, 3], axis=-1)) <= translation_tolerance)
        if is_static:
            static_groups.append(group)
        group_records.append({'group': group, 'representative_object': members[0], 'objects': members,
                              'object_names': [names[i] for i in members], 'static': bool(is_static),
                              'source_surface_points': len(surface),
                              'rest_bounds': [surface.min(0).tolist(), surface.max(0).tolist()]})
    accepted, rejected = [], []
    for first in range(len(groups)):
        for second in range(first+1, len(groups)):
            g, h = groups[first], groups[second]
            fit = fit_shared_pivot(delta[:, g[0]], delta[:, h[0]], rank_rtol, rank_atol)
            record = {'groups': [first, second], 'rank': fit['rank'],
                      'singular_values': fit['singular_values'].tolist(), 'rank_threshold': fit['rank_threshold'],
                      'fit_max_residual': fit['max_residual'], 'fit_rms_residual': fit['rms_residual']}
            reason = None
            if fit['rank'] < 2:
                reason = 'insufficient_relative_rotational_excitation'
            elif fit['max_residual'] > residual_tolerance:
                reason = 'no_common_fixed_rest_pivot_over_full_timeline'
            if reason is None:
                contacts, gaps = contact_midpoints(surfaces[first], surfaces[second], contact_distance, max_surface_samples)
                record['sampled_contact_count'] = len(contacts)
                if not len(contacts):
                    reason = 'no_nearby_sampled_source_surface_contact'
            if reason is None:
                candidates = fit['point'] + (contacts-fit['point']) @ fit['nullspace'] @ fit['nullspace'].T
                chosen = int(np.argmin(np.square(candidates-contacts).sum(1)))
                pivot = candidates[chosen]
                source_distances = [float(surfaces[k].query(pivot)[0][0]) for k in (first, second)]
                # Select objects first: mixing an advanced object index and a
                # scalar coordinate index moves axes in NumPy unexpectedly.
                transforms = [delta[:, members] for members in (g, h)]
                trajectories = [np.einsum('tmij,j->tmi', value[..., :3, :3], pivot)
                                + value[..., :3, 3] for value in transforms]
                representative_residual = np.linalg.norm(trajectories[0][:, 0]-trajectories[1][:, 0], axis=-1)
                member_deviations = [float(np.linalg.norm(t-t[:, :1], axis=-1).max()) for t in trajectories]
                bound = float(representative_residual.max()) + sum(member_deviations)
                record.update(rest_pivot=pivot.tolist(), rest_axis=(fit['nullspace'][:, 0].tolist()
                              if fit['rank'] == 2 else None),
                              kind='revolute_axis_candidate' if fit['rank'] == 2 else 'shared_point_candidate',
                              contact_reference_point=contacts[chosen].tolist(), contact_min_distance=float(min(gaps)),
                              pivot_surface_distances=source_distances,
                              pivot_contact_region_distance=float(np.linalg.norm(pivot-contacts[chosen])),
                              max_residual=float(representative_residual.max()),
                              rms_residual=float(np.sqrt(np.square(representative_residual).mean())),
                              max_group_member_deviations=member_deviations,
                              max_member_residual_bound=bound)
                if max(source_distances) > pivot_distance:
                    reason = 'fitted_pivot_too_far_from_one_or_both_source_surfaces'
                elif bound > residual_tolerance:
                    reason = 'all_member_full_timeline_residual_bound_exceeds_tolerance'
            if reason is None:
                record['candidate_only'] = True
                accepted.append(record)
            else:
                record['rejection_reason'] = reason
                rejected.append(record)
    return {'format': 'source_rigid_joint_diagnosis_v1', 'coordinate_system': 'source rest-world; not registered Gaussian world',
            'frames_checked': len(matrices), 'source_bbox_diagonal': diagonal, 'thresholds': thresholds,
            'surface_samples_per_direction_max': max_surface_samples,
            'groups': group_records, 'excluded_objects': excluded, 'candidate_joints': accepted,
            'rejected_group_pairs': rejected, 'graph': graph_structure(len(groups), accepted, static_groups),
            'scope': 'Full sampled-timeline invariant-pivot evidence plus sampled source-surface support; not ground-truth mechanical topology. '
                     'A rejected fit is not a diagnosed joint; rejection may also reflect insufficient excitation, sparse surfaces, or source approximation. '
                     'No Gaussians, motion corrections, event thresholds or UI state are changed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rotation-tolerance', type=float, default=1e-5)
    parser.add_argument('--translation-tolerance', type=float)
    parser.add_argument('--residual-tolerance', type=float)
    parser.add_argument('--contact-distance', type=float)
    parser.add_argument('--pivot-distance', type=float)
    parser.add_argument('--max-surface-samples', type=int, default=6000)
    parser.add_argument('--rank-rtol', type=float, default=1e-5)
    parser.add_argument('--rank-atol', type=float, default=1e-7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.parts, allow_pickle=False) as parts:
        result = diagnose(parts, **{key: value for key, value in vars(args).items() if key not in ('parts', 'output')})
    digest = hashlib.sha256()
    with args.parts.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    result['source_parts'] = str(args.parts.resolve())
    result['source_parts_sha256'] = digest.hexdigest()
    serialized = json.dumps(result, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(serialized)
    print(json.dumps({'output': str(args.output), 'frames_checked': result['frames_checked'],
                      'groups': len(result['groups']), 'candidate_joints': len(result['candidate_joints']),
                      'components': len(result['graph']['components']), 'cycle_rank': result['graph']['cycle_rank']}, indent=2))


if __name__ == '__main__':
    main()
