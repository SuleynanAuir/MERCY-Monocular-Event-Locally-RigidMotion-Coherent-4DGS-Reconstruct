"""Array-only, bounded spatial-prior binding of event-inferred rigid seeds.

No files, models, camera paths, or legacy motion are read. Input poses are
world-to-world transforms of the supplied G0, not camera-space PnP matrices.
Nearest-seed propagation is a spatial prior, never a semantic part annotation.
The separate unknown node has an identity *initialization*, not measured
static motion. Callers must apply ``recommended_frozen_groups`` themselves.
"""

from collections.abc import Mapping
import json
import math
import re

import numpy as np


HARD_RIGID_GROUP_KIND = "event_inferred_hard_rigid_groups_v1"


def _positions(value, name, allow_empty=False):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] != 3 or
            (not allow_empty and not len(result)) or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite {'possibly empty ' if allow_empty else 'nonempty '}N-by-3 positions")
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _hashes(value):
    if (not isinstance(value, Mapping) or
            set(value) != {"g0", "events", "intrinsics", "extrinsics"} or
            any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
                for item in value.values())):
        raise ValueError("Exactly four explicit input SHA256 values are required")
    return dict(value)


def _validate_transform(matrix, name):
    if (not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError(f"{name} must be a finite proper rigid 4-by-4 transform")


def _principal_rotation_vector(matrix):
    # The largest eigenvector is (x,y,z,w). This is stable at pi, unlike the
    # usual division by sin(angle) in a trace-only Rodrigues conversion.
    r = matrix
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
        [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
        [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ], dtype=np.float64) / 3
    _, eigenvectors = np.linalg.eigh(k)
    quaternion = eigenvectors[:, -1]
    if quaternion[3] < 0:
        quaternion = -quaternion
    sine = np.linalg.norm(quaternion[:3])
    if sine < 1e-12:
        return np.zeros(3)
    angle = 2 * math.atan2(float(sine), float(max(0, quaternion[3])))
    return quaternion[:3] * (angle / sine)


def _continuous_rotation_vector(matrix, previous):
    vector = _principal_rotation_vector(matrix)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        old_angle = float(np.linalg.norm(previous))
        if old_angle < 1e-10:
            return vector
        turns = int(np.rint(old_angle / (2 * math.pi)))
        return previous / old_angle * (2 * math.pi * turns)
    axis = vector / angle
    nearest_turn = int(np.rint((np.dot(previous, axis) - angle) / (2 * math.pi)))
    candidates = [axis * (angle + 2 * math.pi * turn)
                  for turn in range(nearest_turn - 1, nearest_turn + 2)]
    return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - previous)))


def _rotation_matrix(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def _nearest_distances(query, seeds, chunk_size):
    """Small synthetic inputs need only NumPy; real G0 uses bounded cKDTree."""
    if len(query) * len(seeds) <= 2_000_000:
        result = np.empty(len(query), dtype=np.float64)
        block = min(chunk_size, max(1, 2_000_000 // max(1, len(seeds))))
        for start in range(0, len(query), block):
            delta = query[start:start + block, None, :] - seeds[None, :, :]
            result[start:start + block] = np.sqrt(np.min(np.einsum("nsi,nsi->ns", delta, delta), axis=1))
        return result, "bounded_numpy_small_input"
    try:
        from scipy.spatial import cKDTree
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("Large hard binding requires a working scipy.spatial.cKDTree; no quadratic fallback") from error
    tree = cKDTree(seeds)
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        result[start:start + chunk_size] = tree.query(query[start:start + chunk_size], k=1)[0]
    return result, "scipy_cKDTree"


def _distance_quantiles(value):
    finite = np.asarray(value)[np.isfinite(value)]
    if not len(finite):
        return None
    return {f"p{percent}": float(number) for percent, number in
            zip((0, 50, 90, 95, 99, 100), np.percentile(finite, (0, 50, 90, 95, 99, 100)))}


def _validated_assignment_prior(prior, xyz, seeds, labels, supported_groups):
    """Validate an explicit geometry prior without opening any stored paths."""
    from utils.event_only_geodesic_binding import (ASSIGNMENT_PRIOR, HASH_POLICY,
                                                   geodesic_binding_input_hashes)
    if not isinstance(prior, Mapping) or not isinstance(prior.get("metadata"), Mapping):
        raise ValueError("assignment_prior must be a geodesic result with metadata")
    try:
        metadata = json.loads(json.dumps(dict(prior["metadata"]), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("assignment_prior metadata must contain only finite JSON values") from error
    if (metadata.get("assignment_prior") != ASSIGNMENT_PRIOR or
            type(metadata.get("version")) is not int or metadata["version"] != 1 or
            metadata.get("hash_policy") != HASH_POLICY or
            metadata.get("binding_is_spatial_prior_not_parts") is not True or
            metadata.get("automatic_radius_expansion") is not False or
            metadata.get("input_G0_row_order_preserved") is not True or
            metadata.get("physical_uniqueness_guaranteed") is not False or
            metadata.get("unseen_motion_identifiable") is not False or
            type(metadata.get("observations_added")) is not int or metadata["observations_added"] != 0):
        raise ValueError("Unsupported or falsely observational assignment_prior metadata kind/policy")
    expected_hashes = geodesic_binding_input_hashes(xyz, seeds, labels)
    if any(metadata.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("assignment_prior G0/trusted seed input SHA256 mismatch")
    for name in ("cell_size", "max_euclidean_distance", "max_path_length", "max_edge_length",
                 "effective_max_edge_length", "max_seed_attachment_distance"):
        value = metadata.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value <= 0):
            raise ValueError(f"assignment_prior {name} must be a finite positive budget")
    if (type(metadata.get("max_nodes")) is not int or not 1 <= metadata["max_nodes"] <= 100000 or
            type(metadata.get("neighbors")) is not int or not 1 <= metadata["neighbors"] <= 8 or
            metadata.get("node_cap_policy") != "reject_no_automatic_coarsening" or
            metadata.get("path_cost_includes_seed_and_G0_attachments") is not True):
        raise ValueError("assignment_prior graph budget policy is invalid")
    if (metadata["effective_max_edge_length"] > min(metadata["max_edge_length"], math.sqrt(3) * metadata["cell_size"]) or
            metadata["max_seed_attachment_distance"] > min(metadata["effective_max_edge_length"],
                metadata["max_euclidean_distance"], metadata["max_path_length"])):
        raise ValueError("assignment_prior exceeds its declared edge/attachment budgets")
    threshold = metadata.get("min_relative_margin")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("assignment_prior relative margin threshold is invalid")
    assigned = np.asarray(prior.get("source_labels"))
    path = np.asarray(prior.get("nearest_geodesic"), dtype=np.float64)
    margin = np.asarray(prior.get("margin"), dtype=np.float64)
    if (assigned.shape != (len(xyz),) or not np.issubdtype(assigned.dtype, np.integer) or
            np.any(assigned < -1) or np.any(assigned > np.iinfo(np.int32).max) or
            path.shape != assigned.shape or margin.shape != assigned.shape):
        raise ValueError("assignment_prior arrays must retain the complete G0 row order/length")
    known = assigned >= 0
    if np.any(~np.isin(assigned[known], list(supported_groups))):
        raise ValueError("assignment_prior assigns an unsupported source group")
    if (np.isnan(path).any() or np.any(path < 0) or
            np.any(path[np.isfinite(path)] > metadata["max_path_length"]) or
            not np.isfinite(path[known]).all()):
        raise ValueError("assignment_prior assigned paths must be finite and inside path budget")
    if (not np.isfinite(margin).all() or np.any(margin < 0) or np.any(margin > 1) or
            np.any(margin[known] < threshold) or np.any(margin[known] <= 0) or
            np.any(margin[~np.isfinite(path)] != 0)):
        raise ValueError("assignment_prior competition margin is invalid or below threshold")
    return assigned.astype(np.int32, copy=True), path.copy(), margin.copy(), metadata


def build_hard_binding(g0_xyz, seed_xyz, seed_labels, pose_times, poses, pose_valid,
                       input_sha256, *, max_binding_distance, min_support=8,
                       min_valid_poses=2, unknown_policy="identity", static_label=0,
                       seed_confidence=None, min_seed_confidence=0.0,
                       ambiguity_distance=0.0, chunk_size=65536, assignment_prior=None):
    """Return a hard graph, initial knot arrays, and honest support diagnostics.

    ``seed_labels[S]`` is -1 (unknown) or a source pose column in [0,G).
    ``poses[T,G,4,4]`` contains G0-world-relative SE3; invalid entries may be
    NaN but are never used. ``pose_valid`` must be an actual Boolean array.
    Dynamic groups require distinct confident seeds and non-t0 valid poses.
    The static source label needs seed support, but its identity pose flags
    are a prior, not observations. All dynamic t0 flags are also nonobserved.

    Every G0 row is kept, assigned to its nearest supported group's seed only
    inside the explicit distance limit. Equal/near-equal competing groups
    within ``ambiguity_distance`` remain unknown. Unknowns go to one separate
    -1 source-label node, or cause an error with ``unknown_policy='reject'``.
    A node is not created for an unsupported or zero-member source group.

    Optionally ``assignment_prior`` is the complete result of
    ``build_geodesic_binding`` on the same ordered G0 and seed arrays. Its
    kind, geometry hashes, supported labels, finite path budgets and margins
    are checked, plus the chosen group's distance to its confident seeds.
    Prior unknowns stay unknown; they are never filled by nearest-seed fallback.
    The prior changes only membership, never pose observation flags. With no
    prior supplied, the original nearest-seed behavior is unchanged.

    Missing dynamic knots interpolate continuous equivalent rotation vectors
    and center-based translations from the fixed t0 identity and genuine
    valid knots. After the last valid knot they hold the last initialization.
    This fills parameters, never observations, and does not resolve angular
    aliasing when true motion exceeds pi between available observations.

    Returned ``rotation_vectors``/``translations`` have shape [T-1,K,3] for
    copying into EventOnlyMotion parameters. ``supervision_seed_groups[S]``
    maps eligible source seeds to output node IDs (-1 means no known group).
    ``recommended_frozen_groups[K]`` marks static-prior/unknown nodes; this
    array does not itself freeze any caller's optimizer or prove stationarity.
    """
    xyz = _positions(g0_xyz, "G0")
    seeds = _positions(seed_xyz, "seed_xyz", allow_empty=True)
    labels = np.asarray(seed_labels)
    if (labels.shape != (len(seeds),) or not np.issubdtype(labels.dtype, np.integer) or
            np.any(labels < -1)):
        raise ValueError("seed_labels must be an integer S-vector with labels >= -1")
    times = np.asarray(pose_times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("pose_times must be finite, strictly increasing, and start at zero")
    transforms = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(pose_valid)
    if (transforms.ndim != 4 or transforms.shape[0] != len(times) or transforms.shape[2:] != (4, 4) or
            valid.dtype != np.bool_ or valid.shape != transforms.shape[:2]):
        raise ValueError("poses must be T-by-G-by-4-by-4 and pose_valid a Boolean T-by-G array")
    group_count = transforms.shape[1]
    if np.any(labels >= group_count):
        raise ValueError("seed_labels reference a missing pose column")
    support_minimum = _positive_integer(min_support, "min_support")
    valid_minimum = _positive_integer(min_valid_poses, "min_valid_poses")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    if (isinstance(max_binding_distance, (bool, np.bool_)) or
            not math.isfinite(float(max_binding_distance)) or float(max_binding_distance) <= 0):
        raise ValueError("max_binding_distance must be an explicit finite positive distance")
    radius = float(max_binding_distance)
    if not math.isfinite(float(ambiguity_distance)) or ambiguity_distance < 0:
        raise ValueError("ambiguity_distance must be finite and nonnegative")
    if unknown_policy not in {"identity", "reject"}:
        raise ValueError("unknown_policy must be identity or reject")
    if (static_label is not None and
            (isinstance(static_label, (bool, np.bool_)) or not isinstance(static_label, (int, np.integer)) or
             static_label < 0)):
        raise ValueError("static_label must be a nonnegative integer or None")
    static_label = None if static_label is None else int(static_label)
    if not math.isfinite(float(min_seed_confidence)) or not 0 <= min_seed_confidence <= 1:
        raise ValueError("min_seed_confidence must be in [0, 1]")
    confidence = np.ones(len(seeds)) if seed_confidence is None else np.asarray(seed_confidence, dtype=np.float64)
    if (confidence.shape != (len(seeds),) or not np.isfinite(confidence).all() or
            np.any(confidence < 0) or np.any(confidence > 1)):
        raise ValueError("seed_confidence must be a finite S-vector in [0, 1]")
    hashes = _hashes(input_sha256)
    for time_index, group in np.argwhere(valid):
        matrix = transforms[time_index, group]
        _validate_transform(matrix, f"poses[{time_index},{group}]")
        if ((time_index == 0 or group == static_label) and
                not np.allclose(matrix, np.eye(4), atol=1e-5, rtol=0)):
            raise ValueError("t0 and static-prior valid poses must be identity, not measured displacement")

    eligible_seed = (labels >= 0) & (confidence >= min_seed_confidence)
    group_seeds = {}
    source_diagnostics = []
    for group in range(group_count):
        rows = np.flatnonzero(eligible_seed & (labels == group))
        unique = np.unique(seeds[rows], axis=0)
        actual_valid = int(valid[1:, group].sum()) if group != static_label else 0
        rejection = None
        if len(unique) < support_minimum:
            rejection = "insufficient_distinct_confident_seeds"
        elif group != static_label and actual_valid < valid_minimum:
            rejection = "insufficient_non_t0_observed_poses"
        if rejection is None:
            group_seeds[group] = unique
        source_diagnostics.append({"source_label": group, "confident_seed_count": int(len(rows)),
                                   "distinct_seed_count": int(len(unique)),
                                   "observed_non_t0_pose_count": actual_valid,
                                   "static_identity_prior": group == static_label,
                                   "rejection_reason": rejection})

    nearest = np.full(len(xyz), np.inf)
    runner_up = np.full(len(xyz), np.inf)
    assigned_source = np.full(len(xyz), -1, dtype=np.int32)
    backends = set()
    for group, points in group_seeds.items():
        distance, backend = _nearest_distances(xyz, points, chunk_size)
        backends.add(backend)
        closer = distance < nearest
        runner_up = np.where(closer, nearest, np.minimum(runner_up, distance))
        nearest = np.minimum(nearest, distance)
        assigned_source[closer] = group
    inside = nearest <= radius
    ambiguous = np.zeros(len(xyz), dtype=bool)
    comparable = inside & np.isfinite(runner_up)
    ambiguous[comparable] = runner_up[comparable] - nearest[comparable] <= ambiguity_distance
    assigned_source[~inside | ambiguous] = -1
    prior_metadata = None
    if assignment_prior is not None:
        assigned_source, prior_paths, prior_margin, prior_metadata = _validated_assignment_prior(
            assignment_prior, xyz, seeds, labels, group_seeds)
        # Verify the selected group, not merely the globally nearest group:
        # a bad prior must not move distant body points with a nearby bucket.
        for group in np.unique(assigned_source[assigned_source >= 0]):
            members = assigned_source == group
            selected_distance, backend = _nearest_distances(xyz[members], group_seeds[int(group)], chunk_size)
            backends.add(backend)
            limit = min(radius, prior_metadata["max_euclidean_distance"])
            if np.any(selected_distance > limit):
                raise ValueError("assignment_prior selected-group distance exceeds the explicit Euclidean/binding budget")
            tolerance = 64 * np.finfo(np.float64).eps * max(1., radius, prior_metadata["max_path_length"])
            if np.any(prior_paths[members] + tolerance < selected_distance):
                raise ValueError("assignment_prior path is shorter than its selected-group Euclidean lower bound")
            nearest[members] = selected_distance
        inside = nearest <= radius
        ambiguous = ((assigned_source < 0) & np.isfinite(prior_paths) &
                     ((prior_margin < prior_metadata["min_relative_margin"]) | (prior_margin <= 0)))
    unknown = assigned_source < 0
    if unknown.any() and unknown_policy == "reject":
        if prior_metadata is not None:
            raise ValueError(f"{int(unknown.sum())} G0 Gaussians remain unknown under the bounded geometric assignment prior")
        raise ValueError(f"{int(unknown.sum())} G0 Gaussians lack an unambiguous supported seed within max_binding_distance")

    retained = [group for group in sorted(group_seeds) if np.any(assigned_source == group)]
    for item in source_diagnostics:
        if item["rejection_reason"] is None and item["source_label"] not in retained:
            item["rejection_reason"] = "no_bound_G0_members"
    source_labels = retained + ([-1] if unknown.any() else [])
    output_labels = np.empty(len(xyz), dtype=np.int32)
    centers = np.empty((len(source_labels), 3), dtype=np.float64)
    seed_groups = np.full(len(seeds), -1, dtype=np.int32)
    observed = np.zeros((len(times), len(source_labels)), dtype=bool)
    rotation_vectors = np.zeros((len(times), len(source_labels), 3), dtype=np.float64)
    translations = np.zeros_like(rotation_vectors)
    initial_poses = np.tile(np.eye(4), (len(times), len(source_labels), 1, 1))
    frozen = np.zeros(len(source_labels), dtype=bool)
    output_diagnostics = []
    for node, group in enumerate(source_labels):
        members = assigned_source == group
        output_labels[members] = node
        centers[node] = xyz[members].mean(axis=0)
        role = "unobserved_unknown_identity_prior" if group == -1 else (
            "seed_supported_static_identity_prior" if group == static_label else "event_inferred_dynamic_group")
        frozen[node] = group in (-1, static_label)
        interpolated, held = [], []
        if group >= 0:
            seed_groups[eligible_seed & (labels == group)] = node
        if group >= 0 and group != static_label:
            observed[1:, node] = valid[1:, group]
            known = np.r_[0, np.flatnonzero(observed[1:, node]) + 1]
            previous = np.zeros(3)
            knot_rotations, knot_translations = [previous], [np.zeros(3)]
            for index in known[1:]:
                matrix = transforms[index, group]
                previous = _continuous_rotation_vector(matrix[:3, :3], previous)
                knot_rotations.append(previous)
                knot_translations.append(matrix[:3, 3] + (matrix[:3, :3] - np.eye(3)) @ centers[node])
            knot_rotations, knot_translations = np.asarray(knot_rotations), np.asarray(knot_translations)
            for dimension in range(3):
                rotation_vectors[:, node, dimension] = np.interp(times, times[known], knot_rotations[:, dimension])
                translations[:, node, dimension] = np.interp(times, times[known], knot_translations[:, dimension])
            interpolated = [int(index) for index in range(1, int(known[-1])) if not observed[index, node]]
            held = list(range(int(known[-1]) + 1, len(times)))
            for index in range(len(times)):
                rotation = _rotation_matrix(rotation_vectors[index, node])
                initial_poses[index, node, :3, :3] = rotation
                initial_poses[index, node, :3, 3] = translations[index, node] - (rotation - np.eye(3)) @ centers[node]
        output_diagnostics.append({
            "node": node, "source_label": int(group), "role": role,
            "member_count": int(members.sum()), "nearest_seed_distance_quantiles": _distance_quantiles(nearest[members]),
            "pose_observed_indices": np.flatnonzero(observed[:, node]).tolist(),
            "interpolated_initialization_indices": interpolated, "held_tail_initialization_indices": held,
            "recommended_frozen": bool(frozen[node]),
        })

    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    scale = scale if scale > 0 else 1.0
    if not math.isfinite(scale) or scale > np.finfo(np.float32).max:
        raise ValueError("G0 extent does not fit the finite float32 motion graph")
    arrays = [centers, rotation_vectors[1:], translations[1:]]
    if any(not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float32).max) for value in arrays):
        raise ValueError("Hard binding initialization does not fit finite float32 parameters")
    diagnostics = {
        "source_groups": source_diagnostics, "output_groups": output_diagnostics,
        "gaussian_count": len(xyz), "known_bound_gaussians": int((~unknown).sum()),
        "unknown_gaussians": int(unknown.sum()),
        "static_prior_gaussians": int((assigned_source == static_label).sum()) if static_label is not None else 0,
        "outside_binding_radius_gaussians": int((~inside).sum()), "ambiguous_gaussians": int(ambiguous.sum()),
        "retained_known_group_count": len(retained),
        "retained_dynamic_group_count": sum(group != static_label for group in retained),
        "unknown_group_count": int(unknown.any()),
        "distance_backends": sorted(backends), "binding_is_spatial_prior_not_parts": True,
        "unknown_is_observed_static": False, "physical_motion_validated": False,
    }
    metadata = {
        "kind": HARD_RIGID_GROUP_KIND, "version": 1,
        "binding_source": "explicit_G0_raw_events_and_camera", "node_source": "G0_group_member_centers",
        "blender_parts": False, "input_sha256": hashes,
        "physical_uniqueness_guaranteed": False, "unseen_motion_identifiable": False,
        "method": "bounded_nearest_supported_seed_spatial_prior_v1", "center_rule": "mean_of_all_bound_G0_members",
        "binding_is_spatial_prior_not_parts": True, "max_binding_distance": radius,
        "ambiguity_distance": float(ambiguity_distance), "min_support": support_minimum,
        "min_valid_poses": valid_minimum, "min_seed_confidence": float(min_seed_confidence),
        "unknown_policy": unknown_policy, "unknown_source_label": -1, "unknown_is_observed_static": False,
        "static_source_label": static_label, "static_prior_pose_validity": static_label in retained,
        "group_source_labels": source_labels, "pose_observed": observed.tolist(),
        "recommended_frozen_groups": frozen.tolist(),
        "missing_pose_initialization": "linear_continuous_axis_angle_and_center_translation_then_hold_last",
        "rotation_branch_policy": "nearest_equivalent_rotvec_branch_not_observed_turn_count",
        "t0_identity_is_observation": False, "diagnostics": diagnostics,
    }
    if prior_metadata is not None:
        metadata["method"] = "bounded_G0_geodesic_assignment_prior_v1"
        metadata["assignment_prior"] = prior_metadata["assignment_prior"]
        metadata["assignment_prior_metadata"] = prior_metadata
        metadata["assignment_prior_input_hashes_verified"] = True
        metadata["assignment_prior_is_observation"] = False
        diagnostics["assignment_prior_used"] = True
        diagnostics["assignment_prior_unknown_gaussians"] = int(unknown.sum())
    graph = {"centers": centers.astype(np.float32), "knn_indices": output_labels[:, None],
             "knn_weights": np.ones((len(xyz), 1), dtype=np.float32),
             "edges": np.empty((0, 2), dtype=np.int64), "scene_scale": scale, "metadata": metadata}
    return {"graph": graph, "times": times.copy(),
            "rotation_vectors": rotation_vectors[1:].astype(np.float32),
            "translations": translations[1:].astype(np.float32), "initial_world_poses": initial_poses,
            "pose_observed": observed, "gaussian_group_ids": output_labels,
            "group_source_labels": np.asarray(source_labels, dtype=np.int32),
            "supervision_seed_groups": seed_groups, "recommended_frozen_groups": frozen,
            "diagnostics": diagnostics}
