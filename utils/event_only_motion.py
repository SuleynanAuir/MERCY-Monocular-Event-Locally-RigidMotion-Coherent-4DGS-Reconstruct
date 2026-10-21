"""Explicit-input deformation graphs; no external motion or part labels.

The default graph is built only from G0. An optional hard rigid grouping may
be inferred from G0, raw events, and the camera, with separate provenance.
Neither model is Blender parts or a uniquely identified physical solution.
Single-view event supervision does not uniquely determine its hidden motion.
Quaternions throughout this module use the 3DGS ``(w, x, y, z)`` convention.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re

import torch
from torch import nn


HARD_RIGID_GROUP_KIND = "event_inferred_hard_rigid_groups_v1"


def _hard_rigid_group_metadata(graph, indices, edges, node_count):
    """Validate the opt-in hard-group contract without changing legacy graphs.

    Membership is inferred, not a semantic/physical part annotation. Centers
    refer to G0 members, but their grouping also depends on events and camera.
    The checkpoint additionally binds these hashes to its input provenance.
    """
    metadata = graph["metadata"]
    required = {"kind": HARD_RIGID_GROUP_KIND,
                "binding_source": "explicit_G0_raw_events_and_camera",
                "node_source": "G0_group_member_centers"}
    if (any(metadata.get(key) != value for key, value in required.items()) or
            metadata.get("blender_parts") is not False):
        raise ValueError("Hard rigid groups require explicit event-inferred binding provenance")
    if "version" in metadata and (type(metadata["version"]) is not int or metadata["version"] != 1):
        raise ValueError("Unsupported hard rigid group metadata version")
    for key in ("physical_uniqueness_guaranteed", "unseen_motion_identifiable"):
        if metadata.get(key, False) is not False:
            raise ValueError("Hard rigid groups cannot claim uniquely identified physical/hidden motion")
    hashes = metadata.get("input_sha256")
    if (not isinstance(hashes, Mapping) or
            set(hashes) != {"g0", "events", "intrinsics", "extrinsics"} or
            any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in hashes.values())):
        raise ValueError("Hard rigid groups require four explicit input SHA256 values")
    # Check original weights before float32 conversion/normalization, so a
    # near-one soft weight cannot be rounded into a valid hard assignment.
    original_weights = torch.as_tensor(graph["knn_weights"])
    if (indices.shape[1] != 1 or original_weights.shape != indices.shape or
            not torch.equal(original_weights, torch.ones_like(original_weights))):
        raise ValueError("Hard rigid groups require k=1 and weights exactly one")
    if tuple(edges.shape) != (0, 2):
        raise ValueError("Hard rigid groups require empty edges with shape (0, 2)")
    if torch.unique(indices[:, 0]).numel() != node_count:
        raise ValueError("Every hard rigid group must have at least one G0 member")
    return {**metadata, "version": 1, "input_sha256": dict(hashes),
            "physical_uniqueness_guaranteed": False, "unseen_motion_identifiable": False}


def build_graph(xyz_numpy, nodes=192, neighbors=4, seed=0):
    """Build a deterministic graph exclusively from the supplied G0 positions.

    Voxel candidates followed by farthest-point sampling avoid an N-by-K
    distance matrix. cKDTree supplies bounded N-by-neighbors interpolation
    arrays. ``nodes`` is an upper bound when the source has too few distinct
    positions. Input ordering, including every Gaussian row, is preserved.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    for value, name in ((nodes, "nodes"), (neighbors, "neighbors")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    xyz = np.asarray(xyz_numpy, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("G0 positions must be a nonempty finite N-by-3 array")
    lower, upper = xyz.min(axis=0), xyz.max(axis=0)
    extent = upper - lower
    scene_scale = float(np.linalg.norm(extent))
    if not math.isfinite(scene_scale):
        raise ValueError("G0 extent is not finite")
    scene_scale = scene_scale if scene_scale > 0 else 1.0
    target = min(nodes, len(xyz))
    candidate_limit = max(2048, target * 32)
    cell = scene_scale / max(1, target * 8) ** (1 / 3)
    candidates = None
    for _ in range(8):
        # At most N-by-3 scratch storage, never N-by-node_count.
        voxel = np.floor((xyz - lower) / cell).astype(np.int64)
        _, representatives = np.unique(voxel, axis=0, return_index=True)
        candidates = xyz[representatives]
        if len(candidates) >= target or not np.any(extent):
            break
        cell *= 0.5
    if len(candidates) > candidate_limit:
        # np.unique orders voxels lexicographically; stratification retains
        # candidates throughout that spatial ordering without a random crop.
        candidate_ids = np.linspace(0, len(candidates) - 1, candidate_limit, dtype=np.int64)
        candidates = candidates[candidate_ids]
    count = min(target, len(candidates))
    random = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = random.integers(len(candidates))
    nearest_squared = np.full(len(candidates), np.inf)
    for index in range(count):
        if index:
            selected[index] = int(np.argmax(nearest_squared))
        delta = candidates - candidates[selected[index]]
        nearest_squared = np.minimum(nearest_squared, np.einsum("ij,ij->i", delta, delta))
        nearest_squared[selected[:index + 1]] = -1
    centers = candidates[selected]
    tree = cKDTree(centers)
    neighbor_count = min(neighbors, count)
    query_count = min(neighbor_count + 1, count)
    # Query in chunks to bound temporary float64 distance storage as well.
    knn_indices = np.empty((len(xyz), neighbor_count), dtype=np.int32)
    knn_weights = np.empty((len(xyz), neighbor_count), dtype=np.float32)
    for start in range(0, len(xyz), 65536):
        stop = min(start + 65536, len(xyz))
        distance, indices = tree.query(xyz[start:stop], k=query_count)
        distance = np.asarray(distance).reshape(stop - start, query_count)
        indices = np.asarray(indices).reshape(stop - start, query_count)
        if query_count > neighbor_count:
            radius = distance[:, neighbor_count:neighbor_count + 1]
        else:
            radius = distance[:, -1:] * 1.001
        radius = np.maximum(radius, scene_scale * 1e-8)
        weights = np.maximum(1 - distance[:, :neighbor_count] / radius, 0) ** 2
        total = weights.sum(axis=1, keepdims=True)
        # Equidistant/tied neighbors can all lie exactly on the support radius.
        tied = total[:, 0] <= np.finfo(np.float64).eps
        weights[tied] = 1
        total[tied] = neighbor_count
        knn_indices[start:stop] = indices[:, :neighbor_count]
        knn_weights[start:stop] = weights / total
    if count > 1:
        _, adjacent = tree.query(centers, k=min(9, count))
        edge_set = {tuple(sorted((index, int(other))))
                    for index, row in enumerate(adjacent) for other in row if index != other}
        edges = np.asarray(sorted(edge_set), dtype=np.int64).reshape(-1, 2)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    return {
        "centers": centers.astype(np.float32), "knn_indices": knn_indices,
        "knn_weights": knn_weights, "edges": edges, "scene_scale": scene_scale,
        "metadata": {"kind": "g0_only_latent_deformation_graph", "version": 1,
                     "node_source": "explicit_G0_positions_only", "seed": seed,
                     "source_gaussians": len(xyz), "nodes": count, "neighbors": neighbor_count,
                     "blender_parts": False, "physical_uniqueness_guaranteed": False,
                     "unseen_motion_identifiable": False},
    }


def _normalize_quaternion(quaternion):
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    safe = quaternion / norm.clamp_min(torch.finfo(quaternion.dtype).eps)
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1
    return torch.where(norm > torch.finfo(quaternion.dtype).eps, safe, identity)


def _axis_angle_quaternion(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    # sin(angle/2)/angle has a well-defined nonzero derivative w.r.t. the
    # vector at zero. sinc avoids the zero/zero and NaN backward of a where-
    # guarded raw division, so zero initialization remains trainable.
    scale = 0.5 * torch.sinc(angle / (2 * math.pi))
    return torch.cat((torch.cos(angle * 0.5), vector * scale), dim=-1)


def _quaternion_multiply(left, right):
    scalar = left[..., :1] * right[..., :1] - (left[..., 1:] * right[..., 1:]).sum(dim=-1, keepdim=True)
    vector = (left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:] +
              torch.cross(left[..., 1:], right[..., 1:], dim=-1))
    return torch.cat((scalar, vector), dim=-1)


def _rotate(quaternion, vector):
    imaginary = quaternion[..., 1:]
    doubled = 2 * torch.cross(imaginary, vector, dim=-1)
    return vector + quaternion[..., :1] * doubled + torch.cross(imaginary, doubled, dim=-1)


class EventOnlyMotion(nn.Module):
    """Piecewise-linear knot motion with an exactly fixed first G0 frame.

    ``rotation_vectors`` and ``translations`` contain only T-1 trainable knots;
    the first knot at time zero is structurally absent from the parameters.
    ``deform(..., indices=rows)`` means the supplied xyz/rot already contain
    those G0 rows; indices are their original row IDs, not an extra slice.
    """

    def __init__(self, graph, times, chunk_size=65536):
        super().__init__()
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        centers = torch.as_tensor(graph["centers"], dtype=torch.float32).clone()
        indices = torch.as_tensor(graph["knn_indices"])
        weights = torch.as_tensor(graph["knn_weights"], dtype=torch.float32).clone()
        edges = torch.as_tensor(graph["edges"])
        scale = float(graph["scene_scale"])
        if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or not torch.isfinite(centers).all():
            raise ValueError("Graph centers must be finite K-by-3 positions")
        if (indices.ndim != 2 or indices.shape[1] < 1 or not len(indices) or
                indices.dtype not in (torch.int32, torch.int64) or weights.shape != indices.shape or
                not torch.isfinite(weights).all() or torch.any(weights < 0) or
                torch.any(indices < 0) or torch.any(indices >= len(centers)) or
                not torch.allclose(weights.sum(dim=1), torch.ones(len(weights)), atol=1e-5, rtol=1e-5)):
            raise ValueError("Graph neighbor indices/weights are invalid")
        if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype not in (torch.int32, torch.int64) or
                torch.any(edges < 0) or torch.any(edges >= len(centers)) or
                torch.any(edges[:, 0] == edges[:, 1])):
            raise ValueError("Graph edges are invalid")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scene_scale must be finite and positive")
        supplied_metadata = graph.get("metadata")
        hard_metadata = None
        if (isinstance(supplied_metadata, Mapping) and
                supplied_metadata.get("kind") == HARD_RIGID_GROUP_KIND):
            hard_metadata = _hard_rigid_group_metadata(graph, indices, edges, len(centers))
        knots = torch.as_tensor(times, dtype=torch.float64).clone()
        if (knots.ndim != 1 or len(knots) < 2 or not torch.isfinite(knots).all() or
                knots[0] != 0 or torch.any(knots[1:] <= knots[:-1])):
            raise ValueError("Times must be finite, strictly increasing, and begin at zero")
        self.register_buffer("centers", centers)
        self.register_buffer("knn_indices", indices.to(dtype=torch.long).clone())
        self.register_buffer("knn_weights", weights / weights.sum(dim=1, keepdim=True))
        self.register_buffer("edges", edges.to(dtype=torch.long).clone())
        self.register_buffer("times", knots)
        self.register_buffer("scene_scale", torch.tensor(scale, dtype=torch.float32))
        self.rotation_vectors = nn.Parameter(torch.zeros((len(knots) - 1, len(centers), 3)))
        self.translations = nn.Parameter(torch.zeros_like(self.rotation_vectors))
        self.chunk_size = chunk_size
        self.metadata = {"kind": "g0_only_latent_deformation_graph", "version": 1,
                         "node_source": "explicit_G0_positions_only", "blender_parts": False,
                         "physical_uniqueness_guaranteed": False, "unseen_motion_identifiable": False,
                         "quaternion_order": "wxyz", "interpolation": "linear_axis_angle_and_translation",
                         "anchor": "fixed_identity_at_time_zero", "nodes": len(centers),
                         "source_gaussians": len(indices), "knots": len(knots)}
        if hard_metadata is not None:
            self.metadata = {**hard_metadata, "quaternion_order": "wxyz",
                             "interpolation": "linear_axis_angle_and_translation",
                             "anchor": "fixed_identity_at_time_zero", "nodes": len(centers),
                             "source_gaussians": len(indices), "knots": len(knots)}

    def _all_knots(self):
        zero = self.translations.new_zeros((1, len(self.centers), 3))
        return torch.cat((zero, self.translations), dim=0), torch.cat((zero, self.rotation_vectors), dim=0)

    def at_time(self, time):
        """Return (K-by-3 node translations, K-by-4 wxyz quaternions)."""
        value = torch.as_tensor(time, dtype=self.times.dtype, device=self.times.device)
        if value.ndim != 0 or not torch.isfinite(value) or value < self.times[0] or value > self.times[-1]:
            raise ValueError("Requested time must be finite and inside the knot interval")
        right = torch.searchsorted(self.times, value, right=True).clamp(1, len(self.times) - 1)
        left = right - 1
        alpha = ((value - self.times[left]) / (self.times[right] - self.times[left])).to(self.translations.dtype)
        translations, rotations = self._all_knots()
        translation = translations[left] * (1 - alpha) + translations[right] * alpha
        rotation = rotations[left] * (1 - alpha) + rotations[right] * alpha
        return translation, _axis_angle_quaternion(rotation)

    def deform(self, xyz, rot, time, indices=None):
        """Deform Gaussian centers and orientations without changing row order."""
        if (xyz.ndim != 2 or xyz.shape[1] != 3 or rot.shape != (len(xyz), 4) or
                xyz.device != self.centers.device or rot.device != xyz.device or
                not xyz.is_floating_point() or not rot.is_floating_point()):
            raise ValueError("xyz/rot must be N-by-3/N-by-4 floating tensors on the motion device")
        if indices is None:
            if len(xyz) != len(self.knn_indices):
                raise ValueError("Subset xyz requires explicit original G0 row indices")
            neighbor_ids, weights = self.knn_indices, self.knn_weights
        else:
            rows = torch.as_tensor(indices, device=self.knn_indices.device)
            if (rows.ndim != 1 or len(rows) != len(xyz) or rows.dtype not in (torch.int32, torch.int64) or
                    torch.any(rows < 0) or torch.any(rows >= len(self.knn_indices))):
                raise ValueError("indices must identify each supplied row in the original G0")
            neighbor_ids = self.knn_indices[rows.long()]
            weights = self.knn_weights[rows.long()]
        translations, quaternions = self.at_time(time)
        # Exact source preservation, including quaternion signs and values.
        if torch.as_tensor(time).detach().item() == 0:
            zero = translations.sum() * 0
            return xyz + zero, rot + zero
        output_xyz, output_rot = [], []
        for start in range(0, len(xyz), self.chunk_size):
            end = min(start + self.chunk_size, len(xyz))
            neighbors = neighbor_ids[start:end]
            blend = weights[start:end].to(dtype=xyz.dtype)
            centers = self.centers[neighbors].to(dtype=xyz.dtype)
            node_q = quaternions[neighbors].to(dtype=xyz.dtype)
            local = xyz[start:end, None, :] - centers
            posed = _rotate(node_q, local) + centers + translations[neighbors].to(dtype=xyz.dtype)
            output_xyz.append((posed * blend[..., None]).sum(dim=1))
            # q and -q are the same rotation. Align each neighborhood before
            # averaging to prevent cancellation and NaNs around pi rotations.
            reference = node_q[:, :1]
            sign = torch.where((node_q * reference).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
            blended_q = _normalize_quaternion((node_q * sign * blend[..., None]).sum(dim=1))
            source_q = _normalize_quaternion(rot[start:end].to(dtype=xyz.dtype))
            output_rot.append(_normalize_quaternion(_quaternion_multiply(blended_q, source_q)))
        if not len(xyz):
            return xyz, rot
        return torch.cat(output_xyz, dim=0), torch.cat(output_rot, dim=0)

    def forward(self, xyz, rot, time, indices=None):
        return self.deform(xyz, rot, time, indices=indices)

    def spatial_arap(self, time):
        """Symmetric edge ARAP energy, normalized by G0 bounding-box scale."""
        translations, rotations = self.at_time(time)
        if not len(self.edges):
            return (translations.sum() + rotations.sum()) * 0
        first, second = self.edges[:, 0], self.edges[:, 1]
        rest = self.centers[second] - self.centers[first]
        moved = rest + translations[second] - translations[first]
        first_error = (_rotate(rotations[first], rest) - moved) / self.scene_scale
        second_error = (_rotate(rotations[second], rest) - moved) / self.scene_scale
        return 0.5 * (first_error.square().mean() + second_error.square().mean())

    def temporal_smoothness(self, velocity_weight=0.0):
        """Nonuniform-time acceleration regularity, with optional velocity cost.

        Translation is normalized by G0 scale. This is a latent smoothness
        prior, not a proof of physically correct or identifiable hidden motion.
        """
        if not math.isfinite(float(velocity_weight)) or velocity_weight < 0:
            raise ValueError("velocity_weight must be finite and nonnegative")
        translations, rotations = self._all_knots()
        delta = (self.times[1:] - self.times[:-1]).to(translations.dtype)[:, None, None]
        velocity_t = (translations[1:] - translations[:-1]) / delta / self.scene_scale
        velocity_r = (rotations[1:] - rotations[:-1]) / delta
        result = (velocity_t.square().mean() + velocity_r.square().mean()) * velocity_weight
        if len(self.times) > 2:
            interval = (delta[1:] + delta[:-1]) * 0.5
            acceleration_t = (velocity_t[1:] - velocity_t[:-1]) / interval
            acceleration_r = (velocity_r[1:] - velocity_r[:-1]) / interval
            result = result + acceleration_t.square().mean() + acceleration_r.square().mean()
        return result
