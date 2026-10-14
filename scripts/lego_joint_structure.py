"""Prepare serializable hard-joint metadata from validated source joints."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.joint_refinement import joint_forest


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def build_structure(parts, binding, baseline, joint_report, joint_tolerance=1e-5):
    """Build source-space FK transformed into the canonical Gaussian world.

    The returned dict contains only JSON-serializable values. It deliberately
    contains all rigid groups, including a fixed root with no moving rows.
    """
    parts, binding, baseline, joint_report = map(Path, (parts, binding, baseline, joint_report))
    if not np.isfinite(joint_tolerance) or not 0 < joint_tolerance <= 1e-5:
        raise ValueError('Joint tolerance must be in (0,1e-5] Gaussian scene units')
    report = json.loads(joint_report.read_text())
    parts_hash = sha256(parts)
    if (report.get('format') != 'source_rigid_joint_diagnosis_v1'
            or report.get('source_parts_sha256') != parts_hash):
        raise ValueError('Joint report does not match the source parts archive')
    with np.load(parts, allow_pickle=False) as archive:
        matrices = archive['object_matrices'].astype(np.float64)
        points = archive['rest_points'].astype(np.float64)
        point_labels = archive['point_parts']
        metadata = json.loads(str(archive['metadata_json']))
    with np.load(binding, allow_pickle=False) as archive:
        labels = archive['labels']
        indices = archive['dynamic_indices']
        registration = archive['registration'].astype(np.float64)
    cached_indices = np.load(baseline/'dynamic_indices.npy', allow_pickle=False)
    poses = np.load(baseline/'poses.npy', mmap_mode='r', allow_pickle=False)
    if (not np.array_equal(indices, cached_indices) or poses.shape != (len(matrices), len(indices), 7)
            or report['frames_checked'] != len(matrices) or registration.shape != (4, 4)
            or not np.isfinite(registration).all() or not np.allclose(registration[3], [0, 0, 0, 1], atol=1e-8)):
        raise ValueError('Binding, baseline, registration and joint timelines disagree')
    scale = float(np.cbrt(np.linalg.det(registration[:3, :3])))
    alignment = registration[:3, :3]/scale if scale > 0 else np.zeros((3, 3))
    if scale <= 0 or not np.allclose(alignment @ alignment.T, np.eye(3), atol=1e-5):
        raise ValueError('Joint transfer requires a proper uniform similarity registration')
    groups, edges = report['groups'], report['candidate_joints']
    count = len(groups)
    if [g['group'] for g in groups] != list(range(count)) or not count:
        raise ValueError('Motion group IDs are not contiguous')
    object_group = np.full(len(matrices[0]), -1, dtype=np.int64)
    representatives = []
    for group, record in enumerate(groups):
        members = np.asarray(record['objects'], dtype=np.int64)
        if (not len(members) or np.any(members < 0) or np.any(members >= len(object_group))
                or np.any(object_group[members] != -1) or record['representative_object'] not in members
                or [metadata['objects'][i] for i in members] != record['object_names']):
            raise ValueError('Joint source-object groups are inconsistent')
        object_group[members] = group
        representatives.append(record['representative_object'])
    deforming = set(metadata.get('deforming_objects', []))
    if any(object_group[label] != -1 for label in deforming):
        raise ValueError('A deforming source object was put in a rigid group')
    dynamic_labels = labels[indices].astype(np.int64)
    if np.any(dynamic_labels < 0) or np.any(dynamic_labels >= len(object_group)):
        raise ValueError('Dynamic binding references an unknown source object')
    if any(label not in deforming and object_group[label] < 0 for label in np.unique(dynamic_labels)):
        raise ValueError('A moving rigid object has no validated source motion group')
    row_groups = object_group[dynamic_labels]
    joint_groups = [edge['groups'] for edge in edges]
    if any(edge['rank'] != 2 or edge.get('rest_axis') is None for edge in edges):
        raise ValueError('Hard axis correction only supports validated revolute-axis candidates')
    pivots = np.asarray([edge['rest_pivot'] for edge in edges], dtype=np.float64) @ registration[:3, :3].T + registration[:3, 3]
    axes = np.asarray([edge['rest_axis'] for edge in edges], dtype=np.float64) @ alignment.T
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    tree_edges = report['graph']['spanning_forest_edge_indices']
    adjacency = [[] for _ in groups]
    for edge in tree_edges:
        first, second = joint_groups[edge]
        adjacency[first].append((second, edge)); adjacency[second].append((first, edge))
    parents, parent_joints = [-2]*count, [-1]*count
    for component in report['graph']['components']:
        root = component['suggested_root']
        if not groups[root]['static'] or component['unanchored']:
            raise ValueError('Every hard-joint component needs a stationary source root')
        parents[root] = -1
        queue = [root]
        for parent in queue:
            for child, edge in adjacency[parent]:
                if parents[child] == -2:
                    parents[child], parent_joints[child] = parent, edge
                    queue.append(child)
    roots, order, loops, locked = joint_forest(parents, parent_joints, joint_groups)
    if loops != sorted(report['graph']['loop_closure_edge_indices']):
        raise ValueError('Joint report spanning tree and closure edges disagree')
    delta = registration @ (matrices @ np.linalg.inv(matrices[0])) @ np.linalg.inv(registration)
    source = delta[:, representatives].copy()
    rotation = source[:, :, :3, :3]
    if (np.max(abs(rotation @ rotation.transpose(0, 1, 3, 2)-np.eye(3))) > 1e-4
            or np.max(abs(np.linalg.det(rotation)-1)) > 1e-4):
        raise ValueError('Source motion contains a nonrigid group')
    source_q = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat().reshape(len(matrices), count, 4)[..., [3, 0, 1, 2]]
    for frame in range(1, len(source_q)):
        source_q[frame] *= np.where((source_q[frame]*source_q[frame-1]).sum(-1) < 0, -1., 1.)[:, None]
    source_q[0] = [1, 0, 0, 0]
    source_q[:, roots] = [1, 0, 0, 0]
    source_t = source[:, :, :3, 3].copy()
    source_t[0] = 0; source_t[:, roots] = 0
    rotation = Rotation.from_quat(source_q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(len(matrices), count, 3, 3)
    # A single well-observed source surface point per group avoids accidentally
    # freezing a rotating wheel whose centroid lies directly on its axle.
    envelopes = np.zeros((len(matrices), count), dtype=np.float64)
    for group, record in enumerate(groups):
        surface = points[np.isin(point_labels, record['objects'])]
        chosen = surface[np.linspace(0, len(surface)-1, min(64, len(surface))).astype(int)]
        chosen = chosen @ registration[:3, :3].T + registration[:3, 3]
        trajectory = np.einsum('tij,nj->tni', rotation[:, group], chosen)+source_t[:, group, None]
        movement = np.square(trajectory-trajectory[:1]).sum(-1)
        selected = movement.max(0).argmax()
        if movement[:, selected].max() > 1e-12:
            envelopes[:, group] = movement[:, selected]/movement[:, selected].max()
    envelopes[0] = 0
    envelopes[:, roots] = 0
    # Exact tree translations at zero residual; loop closure remains a measured
    # source precision constraint and is never quietly projected/soft-penalized.
    projected_t = np.zeros_like(source_t)
    for group in order:
        parent = parents[group]
        if parent != -1:
            pivot = pivots[parent_joints[group]]
            projected_t[:, group] = (np.einsum('tij,j->ti', rotation[:, parent], pivot)+projected_t[:, parent]
                                      - np.einsum('tij,j->ti', rotation[:, group], pivot))
    gaps, original_gaps = [], []
    for (first, second), pivot in zip(joint_groups, pivots):
        rotations_at_pivot = np.einsum('tij,j->ti', rotation[:, first]-rotation[:, second], pivot)
        gaps.append(np.linalg.norm(rotations_at_pivot+projected_t[:, first]-projected_t[:, second], axis=-1))
        original_gaps.append(np.linalg.norm(rotations_at_pivot+source_t[:, first]-source_t[:, second], axis=-1))
    max_gap = float(np.max(gaps)) if gaps else 0.
    if max_gap > joint_tolerance:
        raise ValueError(f'Source closed-loop residual {max_gap} exceeds hard limit {joint_tolerance}')
    # Convex bounding-box corner checks bound the zero-residual departure for
    # every rigid Gaussian, not just a small random point sample.
    rest = np.asarray(poses[0, :, :3], dtype=np.float64)
    max_deviation = 0.
    for label in np.unique(dynamic_labels):
        group = object_group[label]
        if group < 0:
            continue
        selected = rest[dynamic_labels == label]
        corners = np.asarray(list(itertools.product(*zip(selected.min(0), selected.max(0)))))
        change = rotation[:, group]-delta[:, label, :3, :3]
        shifts = projected_t[:, group]-delta[:, label, :3, 3]
        error = np.einsum('tij,nj->tni', change, corners)+shifts[:, None]
        max_deviation = max(max_deviation, float(np.linalg.norm(error, axis=-1).max()))
    if max_deviation > joint_tolerance:
        raise ValueError(f'Grouped rigid prior departure {max_deviation} exceeds limit {joint_tolerance}')
    return {'format': 'source_hard_joint_structure_v1', 'frame_count': len(matrices),
            'coordinate_system': 'canonical Gaussian world; wxyz quaternions; source-rest common pivots',
            'row_groups': row_groups.tolist(), 'parents': parents, 'parent_joints': parent_joints,
            'joint_groups': joint_groups, 'joint_pivots': pivots.tolist(), 'joint_axes': axes.tolist(),
            'source_quaternions': source_q.tolist(), 'source_translations': source_t.tolist(),
            'envelopes': envelopes.tolist(), 'locked_groups': locked, 'loop_edges': loops,
            'joint_tolerance': joint_tolerance, 'source_groups': groups,
            'diagnostics': {'source_joint_max_gap_scene': float(np.max(original_gaps)) if original_gaps else 0.,
                            'zero_residual_joint_max_gap_scene': max_gap,
                            'zero_residual_rigid_xyz_departure_bound_scene': max_deviation,
                            'zero_residual_departure_scope': 'All rigid rows bounded by source-object rest bounding-box corners; excludes float32 cache-rounding error.',
                            'source_registration_scale': scale, 'rigid_rows': int(np.sum(row_groups >= 0)),
                            'unchanged_deforming_rows': int(np.sum(row_groups < 0)),
                            'trainable_groups': sorted(set(range(count))-set(locked))},
            'input_sha256': {'parts': parts_hash, 'binding': sha256(binding), 'joint_report': sha256(joint_report),
                             'dynamic_indices': sha256(baseline/'dynamic_indices.npy'),
                             'baseline_cache_metadata': sha256(baseline/'cache.json')},
            'scope': 'Axis-only residuals on off-cycle branches. Root and every loop-path node/ancestor have zero residual. '
                     'All tree pivots are enforced by FK; every closure edge is checked against the stated source-precision tolerance. '
                     'Band rows keep baked motion; no independent translations, canonical appearance edits, or similarity conjugation.'}
