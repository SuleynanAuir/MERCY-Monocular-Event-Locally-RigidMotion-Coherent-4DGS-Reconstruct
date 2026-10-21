"""Recover and validate each baseline's own rigid motion, without reading a mesh."""
import numpy as np


def quaternion_product(a, b):
    return np.concatenate((a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True),
                           a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
                           + np.cross(a[..., 1:], b[..., 1:])), axis=-1)


def rotate(q, xyz):
    cross = 2 * np.cross(q[..., 1:], xyz)
    return xyz + q[..., :1] * cross + np.cross(q[..., 1:], cross)


def baseline_groups(labels, object_names, source_groups):
    """Use existing attachment labels only; geometry and motion come from GS."""
    mapping = {}
    for record in source_groups:
        for name in record['object_names']:
            if name in mapping:
                raise ValueError('Duplicate object in source motion groups')
            mapping[name] = record['group']
    labels = np.asarray(labels)
    if labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels >= len(object_names)):
        raise ValueError('Binding labels outside the source object list')
    lookup = np.array([mapping.get(name, -1) for name in object_names], np.int32)
    return lookup[labels]


def recover_rigid_motion(poses, row_groups, group_count, tolerance=1e-4):
    """Check ALL frames/rigid rows against representative SE(3), including rotations.

    A source's raw trajectory must never inherit another version's correction.
    Non-rigid rows have group -1 and are intentionally excluded.
    """
    row_groups = np.asarray(row_groups)
    if poses.ndim != 3 or poses.shape[1:] != (len(row_groups), 7):
        raise ValueError('Motion rows do not match binding')
    if np.any(row_groups < -1) or np.any(row_groups >= group_count):
        raise ValueError('Invalid motion group')
    rows = np.flatnonzero(row_groups >= 0)
    ids = row_groups[rows]
    representatives = {int(g): int(np.flatnonzero(row_groups == g)[0]) for g in np.unique(ids)}
    rest = np.asarray(poses[0], np.float64)
    rest_q = rest[:, 3:] / np.linalg.norm(rest[:, 3:], axis=1, keepdims=True)
    conjugate = rest_q * [1, -1, -1, -1]
    quaternions = np.zeros((len(poses), group_count, 4), np.float32)
    quaternions[..., 0] = 1
    translations = np.zeros((len(poses), group_count, 3), np.float32)
    max_position_error = max_rotation_error = 0.
    for index in range(len(poses)):
        frame = np.asarray(poses[index], np.float64)
        if not np.isfinite(frame).all() or np.any(np.linalg.norm(frame[:, 3:], axis=1) < 1e-8):
            raise ValueError('Invalid source pose')
        for group, representative in representatives.items():
            q = frame[representative, 3:]
            q = quaternion_product(q / np.linalg.norm(q), conjugate[representative])
            quaternions[index, group] = q
            translations[index, group] = frame[representative, :3] - rotate(q, rest[representative, :3])
        q = quaternions[index, ids].astype(np.float64)
        predicted = rotate(q, rest[rows, :3]) + translations[index, ids]
        position_error = float(np.linalg.norm(predicted - frame[rows, :3], axis=1).max(initial=0))
        predicted_q = quaternion_product(q, rest_q[rows])
        actual_q = frame[rows, 3:] / np.linalg.norm(frame[rows, 3:], axis=1, keepdims=True)
        rotation_error = float(np.minimum(np.linalg.norm(predicted_q - actual_q, axis=1),
                                         np.linalg.norm(predicted_q + actual_q, axis=1)).max(initial=0))
        max_position_error, max_rotation_error = max(max_position_error, position_error), max(max_rotation_error, rotation_error)
        if max(position_error, rotation_error) > tolerance:
            raise ValueError(f'Source motion is not rigid within its groups at frame {index}: '
                             f'position={position_error}, rotation={rotation_error}')
    return quaternions, translations, dict(checked_frames=len(poses), checked_rigid_rows=len(rows),
                                          max_position_error=max_position_error, max_rotation_error=max_rotation_error)
