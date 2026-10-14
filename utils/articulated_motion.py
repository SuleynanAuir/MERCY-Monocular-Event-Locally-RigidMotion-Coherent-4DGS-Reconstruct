"""Bake source articulated mesh animation onto geometrically attached Gaussians.

Rigid parts use exact rigid transforms. The two elastic bands use local rigid
surface fits; this preserves splat shape but approximates their stretching.
"""
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def apply_transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def quaternion_product(left, right):
    """Hamilton product in scalar-first order, broadcasting over Gaussian rows."""
    aw, av, bw, bv = left[..., :1], left[..., 1:], right[..., :1], right[..., 1:]
    return np.concatenate((aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)), axis=-1)


def wxyz(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[..., [3, 0, 1, 2]]


class ArticulatedBaker:
    def __init__(self, vertices, parts_path, binding_path):
        with np.load(parts_path, allow_pickle=False) as archive:
            self.parts = {key: archive[key] for key in archive.files}
        with np.load(binding_path, allow_pickle=False) as archive:
            binding = {key: archive[key] for key in archive.files}
        self.source = json.loads(str(self.parts['metadata_json']))
        self.binding = json.loads(str(binding['metadata_json']))
        self.indices = binding['dynamic_indices']
        self.labels = binding['labels'][self.indices]
        self.nearest = binding['nearest_vertex'][self.indices]
        self.a = binding['registration']
        self.inverse = np.linalg.inv(self.a)
        self.xyz = np.column_stack([vertices[k][self.indices] for k in ('x', 'y', 'z')])
        self.rotation = np.column_stack([vertices[f'rot_{i}'][self.indices] for i in range(4)])
        matrices = self.parts['object_matrices'].astype(np.float64)
        self.delta = self.a @ (matrices @ np.linalg.inv(matrices[0])) @ self.inverse
        self.delta[0] = np.eye(4)
        self.groups = {label: np.flatnonzero(self.labels == label) for label in np.unique(self.labels)}
        self.bands = {}
        self.max_rigidity_error = 0.
        for label, rows in self.groups.items():
            if label in self.source['deforming_objects']:
                rest = self.parts[f'deformed_{label}'][0].astype(np.float64)
                neighbors = cKDTree(rest).query(rest, k=min(16, len(rest)))[1]
                centered = rest[neighbors] - rest[neighbors].mean(1, keepdims=True)
                self.bands[label] = (neighbors, centered,
                                     self.nearest[rows] - self.parts['part_offsets'][label])
            else:
                rot = self.delta[:, label, :3, :3]
                error = float(np.max(abs(rot @ rot.transpose(0, 2, 1) - np.eye(3))))
                self.max_rigidity_error = max(error, self.max_rigidity_error)
                if error > 1e-4 or np.any(np.linalg.det(rot) < .9999):
                    raise ValueError(f'Part {label} is not rigid: {error}')

    def frame(self, index):
        if not 0 <= index < len(self.delta):
            raise IndexError(index)
        result = np.column_stack((self.xyz, self.rotation)).astype(np.float32)
        if index == 0:
            return result
        for label, rows in self.groups.items():
            if label not in self.bands:
                delta = self.delta[index, label]
                result[rows, :3] = apply_transform(self.xyz[rows], delta)
                result[rows, 3:] = quaternion_product(wxyz(delta[:3, :3]), self.rotation[rows])
                continue
            neighbors, centered, nearest = self.bands[label]
            rest = self.parts[f'deformed_{label}'][0].astype(np.float64)
            current = self.parts[f'deformed_{label}'][index].astype(np.float64)
            local = current[neighbors] - current[neighbors].mean(1, keepdims=True)
            u, _, vt = np.linalg.svd(local.transpose(0, 2, 1) @ centered)
            sign = np.ones((len(u), 3)); sign[:, -1] = np.linalg.det(u @ vt)
            rotation = (u * sign[:, None, :]) @ vt
            local_xyz = apply_transform(self.xyz[rows], self.inverse)
            deformed = np.einsum('nij,nj->ni', rotation[nearest], local_xyz - rest[nearest]) + current[nearest]
            result[rows, :3] = apply_transform(deformed, self.a)
            world_rotation = self.a[:3, :3] @ rotation[nearest] @ self.inverse[:3, :3]
            result[rows, 3:] = quaternion_product(wxyz(world_rotation), self.rotation[rows])
        if not np.isfinite(result).all():
            raise ValueError(f'Non-finite pose at frame {index}')
        return result
