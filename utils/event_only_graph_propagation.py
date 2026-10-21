"""Bounded G0-only spatial motion initialization, not semantic rigid parts.

Visible flow constraints are hard anchors. Hidden nodes are never given motion
by image-plane overlap: support must arrive through short, local G0 3D edges.
This is an initialization prior, not evidence that hidden motion is correct.
"""

from __future__ import annotations

import heapq
import math


def propagate_graph_motion(graph_centers_np, provisional_translations_np,
                           initially_visible_mask_np, accepted_masks_np, edges=None, *,
                           neighbors=6, max_hops=3, radius_multiplier=3.,
                           edge_length_multiplier=2.5, arap_iterations=8,
                           static_anchor_mask_np=None):
    """Propagate trusted visible motion to bounded, graph-supported hidden nodes.

    Arrays contain T-1 parameter knots; the fixed G0/t0 knot is not supplied.
    ``accepted_masks`` at hidden nodes are ignored. Visible nodes without a
    currently accepted update retain their previous anchor, initially zero,
    in the default conservative mode. With an explicit static anchor mask,
    only its nodes are forced to zero; never-tracked visible nodes are unknown
    and may receive the same bounded 3D propagation as hidden nodes. Previously
    trusted visible nodes retain their last observation as a hard prior anchor.
    Unknown nodes without a nearby trusted nonzero anchor remain zero.

    Spatially compatible local anchor fits seed an ARAP solve; otherwise a
    hard-anchor Laplacian extension supplies its initial state. No files,
    cameras, image coordinates, models, or labels are accepted or loaded.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    for value, name, minimum in ((neighbors, "neighbors", 1), (max_hops, "max_hops", 1),
                                 (arap_iterations, "arap_iterations", 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for value, name in ((radius_multiplier, "radius_multiplier"),
                        (edge_length_multiplier, "edge_length_multiplier")):
        if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    centers = np.asarray(graph_centers_np, dtype=np.float64)
    provisional = np.asarray(provisional_translations_np, dtype=np.float64)
    visible = np.asarray(initially_visible_mask_np)
    accepted = np.asarray(accepted_masks_np)
    if centers.ndim != 2 or centers.shape[1] != 3 or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("G0 centers must be finite N-by-3 positions")
    count = len(centers)
    if (provisional.ndim != 3 or provisional.shape[1:] != (count, 3) or
            not len(provisional) or not np.isfinite(provisional).all()):
        raise ValueError("Provisional translations must be finite (T-1)-by-N-by-3")
    if (visible.dtype != np.bool_ or visible.shape != (count,) or accepted.dtype != np.bool_ or
            accepted.shape != provisional.shape[:2]):
        raise ValueError("Visibility and accepted masks must be boolean and match the graph/knots")
    explicit_static = static_anchor_mask_np is not None
    static = np.zeros(count, dtype=bool) if not explicit_static else np.asarray(static_anchor_mask_np)
    if static.dtype != np.bool_ or static.shape != (count,):
        raise ValueError("Static anchor mask must be a boolean N-vector")
    scale = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-8)
    if not math.isfinite(scale):
        raise ValueError("G0 graph scale is not finite")
    tree = cKDTree(centers)
    if count > 1:
        local_distances = tree.query(centers, k=min(4, count))[0][:, 1:]
        positive = local_distances[local_distances > scale * 1e-10]
        fallback = float(np.median(positive)) if len(positive) else scale
        local_scale = np.median(local_distances, axis=1)
        local_scale = np.where(local_scale > scale * 1e-10, local_scale, fallback)
    else:
        local_scale = np.array([scale])
    if edges is None:
        if count > 1:
            _, nearest = tree.query(centers, k=min(neighbors + 1, count))
            pairs = sorted({tuple(sorted((node, int(other)))) for node, row in enumerate(nearest)
                            for other in row if node != other})
            edge_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        else:
            edge_array = np.empty((0, 2), dtype=np.int64)
    else:
        value = np.asarray(edges)
        if (value.ndim != 2 or value.shape[1] != 2 or value.dtype.kind not in "iu" or
                np.any(value < 0) or np.any(value >= count) or np.any(value[:, 0] == value[:, 1])):
            raise ValueError("Edges must contain valid distinct integer node pairs")
        edge_array = np.unique(np.sort(value.astype(np.int64), axis=1), axis=0)
    input_edges = len(edge_array)
    lengths = np.linalg.norm(centers[edge_array[:, 0]] - centers[edge_array[:, 1]], axis=1)
    keep = ((lengths > scale * 1e-10) &
            (lengths <= edge_length_multiplier * np.minimum(local_scale[edge_array[:, 0]], local_scale[edge_array[:, 1]])))
    edge_array, lengths = edge_array[keep], lengths[keep]
    first, second = edge_array[:, 0], edge_array[:, 1]
    characteristic = np.maximum(local_scale[first], local_scale[second])
    weights = np.exp(-.5 * (lengths / characteristic) ** 2) / np.maximum(lengths / scale, 1e-6) ** 2
    adjacency = [[] for _ in range(count)]
    for index, (left, right) in enumerate(edge_array):
        adjacency[left].append((int(right), float(lengths[index]), float(weights[index])))
        adjacency[right].append((int(left), float(lengths[index]), float(weights[index])))
    matrix = coo_matrix((np.concatenate((weights, weights)),
                        (np.concatenate((first, second)), np.concatenate((second, first)))),
                       shape=(count, count)).tocsr()
    laplacian = coo_matrix((np.asarray(matrix.sum(axis=1)).reshape(-1),
                           (np.arange(count), np.arange(count))), shape=(count, count)).tocsr() - matrix

    def nearby_nodes(node):
        """Bound both hop count and accumulated 3D edge length."""
        budget = radius_multiplier * local_scale[node]
        queue = [(0., 0, node)]
        best = {(node, 0): 0.}
        anchors = {}
        while queue:
            distance, hops, current = heapq.heappop(queue)
            if distance > best.get((current, hops), float("inf")):
                continue
            anchors[current] = min(distance, anchors.get(current, float("inf")))
            if hops >= max_hops:
                continue
            for other, edge_distance, _ in adjacency[current]:
                proposed = distance + edge_distance
                if proposed <= budget and proposed < best.get((other, hops + 1), float("inf")):
                    best[other, hops + 1] = proposed
                    heapq.heappush(queue, (proposed, hops + 1, other))
        ids = np.asarray(sorted(anchors), dtype=np.int64)
        return ids, np.asarray([anchors[index] for index in ids], dtype=np.float64)

    potentially_unknown = ~static if explicit_static else ~visible
    neighborhoods = {int(node): nearby_nodes(int(node)) for node in np.flatnonzero(potentially_unknown)}

    def proper_fit(source, target, weight):
        covariance = (source * weight[:, None]).T @ target
        source_covariance = (source * weight[:, None]).T @ source
        eigenvalues = np.linalg.eigvalsh(source_covariance)
        if eigenvalues[-2] <= max(eigenvalues[-1] * 1e-5, scale**2 * 1e-12):
            return None
        left, _, right_t = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[-1, -1] = 1 if np.linalg.det(right_t.T @ left.T) >= 0 else -1
        return right_t.T @ correction @ left.T

    def local_rotations(positions):
        result = np.repeat(np.eye(3)[None], count, axis=0)
        for node, neighbors_and_weights in enumerate(adjacency):
            if len(neighbors_and_weights) < 2:
                continue
            ids = np.array([item[0] for item in neighbors_and_weights], dtype=np.int64)
            weight = np.array([item[2] for item in neighbors_and_weights], dtype=np.float64)
            weight /= weight.sum()
            rotation = proper_fit(centers[node] - centers[ids], positions[node] - positions[ids], weight)
            if rotation is not None:
                result[node] = rotation
        return result

    output = np.zeros_like(provisional, dtype=np.float32)
    output_rotations = np.zeros_like(output)
    anchor_translation = np.zeros((count, 3), dtype=np.float64)
    ever_trusted = np.zeros(count, dtype=bool)
    previous_rotation = np.zeros((count, 3), dtype=np.float64)
    reports = []
    for knot in range(len(provisional)):
        trusted = visible & accepted[knot] & ~static
        anchor_translation[trusted] = provisional[knot, trusted]
        anchor_translation[static] = 0
        ever_trusted |= trusted
        anchored = (static | ever_trusted) if explicit_static else visible
        moving_support = ever_trusted & (np.linalg.norm(anchor_translation, axis=1) > scale * 1e-7)
        eligible = np.zeros(count, dtype=bool)
        anchor_neighborhoods = {}
        for node, (ids, distances) in neighborhoods.items():
            if anchored[node]:
                continue
            choose = anchored[ids]
            ids, distances = ids[choose], distances[choose]
            anchor_neighborhoods[node] = (ids, distances)
            eligible[node] = bool(np.any(moving_support[ids]))
        unknown = np.flatnonzero(eligible)
        fixed = np.flatnonzero(~eligible)
        translations = np.zeros((count, 3), dtype=np.float64)
        translations[anchored] = anchor_translation[anchored]
        positions = centers + translations
        rigid_seed_count = 0
        if len(unknown):
            solver = splu(laplacian[unknown][:, unknown].tocsc())
            coupling = laplacian[unknown][:, fixed]
            translations[unknown] = solver.solve(-coupling @ translations[fixed])
            positions = centers + translations
            for node in unknown:
                ids, distances = anchor_neighborhoods[int(node)]
                if len(ids) < 3:
                    continue
                weight = np.exp(-.5 * (distances / max(local_scale[node], scale * 1e-8)) ** 2)
                weight /= weight.sum()
                source_centroid = (centers[ids] * weight[:, None]).sum(axis=0)
                target_centroid = ((centers[ids] + anchor_translation[ids]) * weight[:, None]).sum(axis=0)
                source = centers[ids] - source_centroid
                target = centers[ids] + anchor_translation[ids] - target_centroid
                rotation = proper_fit(source, target, weight)
                if rotation is None:
                    continue
                residual = source @ rotation.T - target
                error = np.sqrt((weight * (residual * residual).sum(axis=1)).sum())
                spread = np.sqrt((weight * (source * source).sum(axis=1)).sum())
                if error / max(spread, scale * 1e-8) <= .10:
                    positions[node] = (centers[node] - source_centroid) @ rotation.T + target_centroid
                    rigid_seed_count += 1
            for _ in range(arap_iterations):
                rotations = local_rotations(positions)
                rest = centers[first] - centers[second]
                average = .5 * (rotations[first] + rotations[second])
                contribution = weights[:, None] * np.einsum("eij,ej->ei", average, rest)
                right_hand = np.zeros_like(centers)
                np.add.at(right_hand, first, contribution)
                np.add.at(right_hand, second, -contribution)
                positions[unknown] = solver.solve(right_hand[unknown] - coupling @ positions[fixed])
            translations[unknown] = positions[unknown] - centers[unknown]
        # Assign hard anchors again, so round-off never drifts a static anchor.
        translations[anchored] = anchor_translation[anchored]
        translations[~anchored & ~eligible] = 0
        translations[static] = 0
        rotations = local_rotations(centers + translations)
        vectors = Rotation.from_matrix(rotations).as_rotvec()
        for node in np.flatnonzero(eligible | trusted):
            angle = np.linalg.norm(vectors[node])
            if angle > 1e-8:
                axis = vectors[node] / angle
                turns = int(round(float(np.dot(previous_rotation[node] - vectors[node], axis)) / (2 * math.pi)))
                candidates = [vectors[node] + axis * 2 * math.pi * turn for turn in (turns - 1, turns, turns + 1)]
                vectors[node] = min(candidates, key=lambda value: np.linalg.norm(value - previous_rotation[node]))
        vectors[anchored & ~trusted] = previous_rotation[anchored & ~trusted]
        vectors[~anchored & ~eligible] = 0
        vectors[static] = 0
        previous_rotation = vectors.copy()
        output[knot] = translations
        output_rotations[knot] = vectors
        reports.append({"knot_index": knot, "trusted_visible_updates": int(trusted.sum()),
                        "held_visible_anchors": int((anchored & visible & ~trusted & ~static).sum()),
                        "explicit_static_anchors": int(static.sum()),
                        "supported_hidden_nodes": int((eligible & ~visible).sum()),
                        "supported_unreliable_visible_nodes": int((eligible & visible).sum()),
                        "unsupported_hidden_nodes": int((~visible & ~eligible).sum()),
                        "local_rigid_seeds": rigid_seed_count,
                        "hidden_accepted_masks_ignored": int((~visible & accepted[knot]).sum()),
                        "static_accepted_masks_ignored": int((static & accepted[knot]).sum()),
                        "max_hidden_translation": float(np.linalg.norm(translations[~visible], axis=1).max()) if (~visible).any() else 0.,
                        "visible_anchor_max_error": float(np.abs(translations[anchored & visible] - anchor_translation[anchored & visible]).max()) if (anchored & visible).any() else 0.,
                        "static_anchor_max_error": float(np.abs(translations[static]).max()) if static.any() else 0.})
    return {"translations": output, "rotation_vectors": output_rotations,
            "debug": {"knots": reports, "input_edges": input_edges,
                      "retained_edges": len(edge_array), "rejected_long_or_zero_edges": input_edges - len(edge_array)},
            "metadata": {"kind": "bounded_G0_graph_hard_anchor_ARAP_initialization", "version": 2,
                         "spatial_source": "G0_centers_and_G0_edges_only", "uses_image_projection": False,
                         "uses_semantic_or_blender_parts": False, "trusted_visible_anchors": "exact",
                         "unaccepted_visible_anchors": "held_past_observation_else_unknown" if explicit_static else "previous_value_or_initial_zero",
                         "static_mask_policy": "caller_supplied_mask_forced_zero" if explicit_static else "conservative_all_visible_are_anchors",
                         "static_mask_authenticity_verified": False,
                         "unsupported_hidden_nodes": "zero", "max_hops": max_hops,
                         "radius_multiplier": float(radius_multiplier),
                         "edge_length_multiplier": float(edge_length_multiplier),
                         "arap_iterations": arap_iterations, "initial_t0_omitted": True,
                         "hidden_motion_validated": False, "physical_uniqueness_guaranteed": False,
                         "limitation": "nearby_G0_nodes_are_not_known_rigid_parts;_cross_part_leakage_is_still_possible"}}
