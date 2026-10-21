"""Bounded occupied-G0 geometry propagation, never a physical part label.

Array inputs only. There is no file access, motion inference, observation
creation, automatic radius expansion, or implicit voxel-size coarsening.
Large sparse graph operations require SciPy; no dense graph fallback exists.
"""

import hashlib
import math
from numbers import Integral, Real

import numpy as np


ASSIGNMENT_PRIOR = "bounded_occupied_G0_geodesic_v1"
HASH_POLICY = "tag_and_shape_then_float64_le_xyz_int64_le_labels_input_row_order_trusted_labels_ge0_v1"


def _inputs(g0_xyz, seed_xyz, seed_labels):
    xyz = np.asarray(g0_xyz, dtype=np.float64)
    seeds = np.asarray(seed_xyz, dtype=np.float64)
    labels = np.asarray(seed_labels)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("g0_xyz must be nonempty finite N-by-3 positions")
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError("seed_xyz must be finite S-by-3 positions, possibly empty")
    if (labels.shape != (len(seeds),) or not np.issubdtype(labels.dtype, np.integer) or
            np.any(labels < -1) or np.any(labels > np.iinfo(np.int32).max)):
        raise ValueError("seed_labels must be an integer S-vector in [-1, int32_max]")
    return xyz, seeds, labels.astype(np.int64, copy=False)


def _update_array_hash(hasher, tag, value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    hasher.update(tag.encode("ascii") + b"\0")
    hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
    if array.size:
        hasher.update(memoryview(array).cast("B"))


def geodesic_binding_input_hashes(g0_xyz, seed_xyz, seed_labels):
    """Canonical input identities; G0 row order and trusted seed order matter.

    Unknown seeds (label -1) are excluded from the trusted-seed hash. A caller
    may use this helper to check a supplied assignment prior without rebuilding
    its graph. Hash agreement records identity, not physical correctness.
    """
    xyz, seeds, labels = _inputs(g0_xyz, seed_xyz, seed_labels)
    g0_hash, seed_hash = hashlib.sha256(), hashlib.sha256()
    _update_array_hash(g0_hash, "g0_xyz_v1", xyz, "<f8")
    trusted = labels >= 0
    _update_array_hash(seed_hash, "trusted_seed_xyz_v1", seeds[trusted], "<f8")
    _update_array_hash(seed_hash, "trusted_seed_labels_v1", labels[trusted], "<i8")
    return {"g0_xyz_sha256": g0_hash.hexdigest(), "trusted_seed_sha256": seed_hash.hexdigest()}


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be an explicit finite positive number")
    return float(value)


def _integer(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(value)


def _scipy_backend():
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("Geodesic binding requires working scipy sparse/cKDTree; no dense fallback") from error
    return cKDTree, coo_matrix, dijkstra


def _voxel_coordinates(points, origin, cell_size):
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.floor((points - origin) / cell_size)
    if not np.isfinite(values).all() or np.any(np.abs(values) >= np.iinfo(np.int64).max // 4):
        raise ValueError("Coordinates/cell_size exceed the supported int64 voxel range")
    return values.astype(np.int64)


def _finite_quantiles(value):
    values = np.asarray(value)[np.isfinite(value)]
    if not len(values):
        return None
    return {f"p{percent}": float(number) for percent, number in
            zip((0, 50, 90, 95, 99, 100), np.percentile(values, (0, 50, 90, 95, 99, 100)))}


def build_geodesic_binding(g0_xyz, seed_xyz, seed_labels, *, cell_size,
                           max_euclidean_distance, max_path_length, max_edge_length,
                           max_nodes, neighbors=8, min_relative_margin=0.15):
    """Assign bounded geometry-prior labels to every G0 row, or -1 unknown.

    The origin is G0.min(0). Occupied voxels are lexicographically ordered and
    represented by their first actual G0 input row. An excessive voxel count
    raises ValueError; callers must explicitly choose any coarser resolution.
    Edges require mutual kNN (k<=8), neighboring voxel indices (Chebyshev<=1),
    and length<=min(max_edge_length,sqrt(3)*cell_size). No graph edge crosses an
    entire empty voxel layer. Disconnected components are not assigned labels
    merely because another point in the component has a seed.

    A seed uses its occupied voxel, if any, otherwise its nearest adjacent
    voxel representative. Attachment distance is capped by all applicable
    budgets and sqrt(3)*cell_size. Different labels attaching to the same
    node make that node an unknown barrier. Other-label seed nodes also act
    as barriers during each group's multi-source Dijkstra traversal.

    Shortest-path cost includes seed->representative and representative->G0
    attachments. Each row additionally needs a nearest *attached* same-group
    seed within max_euclidean_distance. The competition margin is
    (second-best - best)/second-best among budget-admissible different labels;
    it is 1 with only one admissible group and 0 with none. Exact ties are
    unknown even with min_relative_margin=0. No observations are added.

    Returns source_labels[int32,N], nearest_geodesic[float64,N] (inf when no
    admissible path), margin[float64,N], JSON-safe metadata and diagnostics.
    Distances/margins are geometric criteria, not calibrated probabilities.
    """
    xyz, seeds, labels = _inputs(g0_xyz, seed_xyz, seed_labels)
    cell_size = _positive_real(cell_size, "cell_size")
    euclidean_limit = _positive_real(max_euclidean_distance, "max_euclidean_distance")
    path_limit = _positive_real(max_path_length, "max_path_length")
    edge_limit = _positive_real(max_edge_length, "max_edge_length")
    max_nodes = _integer(max_nodes, "max_nodes", 1, 100000)
    neighbors = _integer(neighbors, "neighbors", 1, 8)
    if (isinstance(min_relative_margin, (bool, np.bool_)) or not isinstance(min_relative_margin, Real) or
            not math.isfinite(float(min_relative_margin)) or not 0 <= min_relative_margin <= 1):
        raise ValueError("min_relative_margin must be finite in [0, 1]")
    margin_limit = float(min_relative_margin)
    hashes = geodesic_binding_input_hashes(xyz, seeds, labels)
    origin = xyz.min(axis=0)
    coordinates = _voxel_coordinates(xyz, origin, cell_size)
    voxels, first_rows, inverse = np.unique(coordinates, axis=0, return_index=True, return_inverse=True)
    node_count = len(voxels)
    if node_count > max_nodes:
        raise ValueError(f"Occupied node count {node_count} exceeds explicit max_nodes={max_nodes}; no automatic coarsening")
    node_xyz = xyz[first_rows]
    within_voxel_distance = np.linalg.norm(xyz - node_xyz[inverse], axis=1)
    local_edge_limit = min(edge_limit, math.sqrt(3) * cell_size)
    attachment_limit = min(local_edge_limit, euclidean_limit, path_limit)
    owner = np.full(len(xyz), -1, dtype=np.int32)
    best = np.full(len(xyz), np.inf)
    second = np.full(len(xyz), np.inf)
    margin = np.zeros(len(xyz), dtype=np.float64)
    conflict = np.zeros(node_count, dtype=bool)
    attached = np.zeros(len(seeds), dtype=bool)
    seed_nodes = np.full(len(seeds), -1, dtype=np.int64)
    seed_distance = np.full(len(seeds), np.inf)
    edge_count = 0
    group_reports = []
    trusted_rows = np.flatnonzero(labels >= 0)
    if len(trusted_rows):
        cKDTree, coo_matrix, dijkstra = _scipy_backend()
        tree = cKDTree(node_xyz)
        query_count = min(node_count, neighbors + 1)
        _, adjacent = tree.query(node_xyz, k=query_count)
        adjacent = np.asarray(adjacent).reshape(node_count, query_count)
        source = np.repeat(np.arange(node_count), query_count)
        target = adjacent.reshape(-1).astype(np.int64)
        keep = source != target
        source, target = source[keep], target[keep]
        # Sparse integer codes avoid N-by-N adjacency or Python set per node.
        codes, reverse = source * node_count + target, target * node_count + source
        mutual = np.isin(codes, reverse, assume_unique=True) & (source < target)
        source, target = source[mutual], target[mutual]
        lengths = np.linalg.norm(node_xyz[source] - node_xyz[target], axis=1)
        local = ((np.abs(voxels[source] - voxels[target]).max(axis=1) <= 1) &
                 (lengths <= local_edge_limit))
        source, target, lengths = source[local], target[local], lengths[local]
        edge_count = len(lengths)

        seed_voxels = _voxel_coordinates(seeds[trusted_rows], origin, cell_size)
        lookup = {tuple(value): index for index, value in enumerate(voxels)}
        _, nearest_node = tree.query(seeds[trusted_rows], k=1)
        for offset, row in enumerate(trusted_rows):
            node = lookup.get(tuple(seed_voxels[offset]), int(nearest_node[offset]))
            distance = float(np.linalg.norm(seeds[row] - node_xyz[node]))
            if distance > attachment_limit or np.max(np.abs(seed_voxels[offset] - voxels[node])) > 1:
                continue
            attached[row], seed_nodes[row], seed_distance[row] = True, node, distance
        anchor_label = np.full(node_count, -1, dtype=np.int64)
        for row in np.flatnonzero(attached):
            node = seed_nodes[row]
            if anchor_label[node] >= 0 and anchor_label[node] != labels[row]:
                conflict[node] = True
            elif anchor_label[node] < 0:
                anchor_label[node] = labels[row]
        # Conflicting sources are not valid propagation starts for either side.
        attached &= ~conflict[np.maximum(seed_nodes, 0)]
        for group in np.unique(labels[trusted_rows]):
            rows = np.flatnonzero(attached & (labels == group))
            report = {"source_label": int(group), "trusted_seed_count": int((labels == group).sum()),
                      "attached_seed_count": int(len(rows)), "reachable_nodes": 0,
                      "budget_admissible_gaussians": 0}
            group_reports.append(report)
            if not len(rows):
                continue
            blocked = conflict | ((anchor_label >= 0) & (anchor_label != group))
            usable = ~blocked[source] & ~blocked[target]
            left, right, costs = source[usable], target[usable], lengths[usable]
            # One private virtual source per group, never all groups' virtual
            # nodes together: other groups must not create zero-cost shortcuts.
            attached_nodes = np.unique(seed_nodes[rows])
            attachment_cost = np.full(node_count, np.inf)
            np.minimum.at(attachment_cost, seed_nodes[rows], seed_distance[rows])
            virtual = np.full(len(attached_nodes), node_count, dtype=np.int64)
            edge_rows = np.r_[left, right, virtual, attached_nodes]
            edge_columns = np.r_[right, left, attached_nodes, virtual]
            edge_values = np.r_[costs, costs, attachment_cost[attached_nodes], attachment_cost[attached_nodes]]
            graph = coo_matrix((edge_values, (edge_rows, edge_columns)),
                               shape=(node_count + 1, node_count + 1)).tocsr()
            # Explicit zero-valued attachment edges are meaningful; do not
            # eliminate sparse zeros or exact-on-G0 seeds become disconnected.
            distance = np.asarray(dijkstra(graph, directed=True, indices=node_count,
                                          limit=path_limit))[:node_count]
            report["reachable_nodes"] = int(np.isfinite(distance).sum())
            total = distance[inverse] + within_voxel_distance
            seed_tree = cKDTree(seeds[rows])
            eligible = np.zeros(len(xyz), dtype=bool)
            for start in range(0, len(xyz), 65536):
                stop = min(start + 65536, len(xyz))
                straight = seed_tree.query(xyz[start:stop], k=1)[0]
                eligible[start:stop] = ((total[start:stop] <= path_limit) &
                                        (straight <= euclidean_limit) & ~blocked[inverse[start:stop]])
            report["budget_admissible_gaussians"] = int(eligible.sum())
            candidate = np.where(eligible, total, np.inf)
            closer = candidate < best
            second = np.where(closer, best, np.minimum(second, candidate))
            best = np.minimum(best, candidate)
            owner[closer] = group

    reachable = np.isfinite(best)
    competing = reachable & np.isfinite(second)
    margin[reachable & ~competing] = 1
    denominator = np.maximum(second[competing], np.finfo(np.float64).tiny)
    margin[competing] = np.clip((second[competing] - best[competing]) / denominator, 0, 1)
    tied = np.zeros(len(xyz), dtype=bool)
    tolerance = np.finfo(np.float64).eps * 32 * np.maximum(second[competing], cell_size)
    tied[competing] = second[competing] - best[competing] <= tolerance
    ambiguous = reachable & ((margin < margin_limit) | tied)
    owner[~reachable | ambiguous | conflict[inverse]] = -1
    for report in group_reports:
        group_members = owner == report["source_label"]
        report["assigned_gaussians"] = int(group_members.sum())
        report["assigned_path_quantiles"] = _finite_quantiles(best[group_members])
    diagnostics = {
        "gaussian_count": len(xyz), "occupied_nodes": node_count, "undirected_edges": edge_count,
        "trusted_seed_count": len(trusted_rows), "attached_seed_count": int(attached.sum()),
        "unattached_or_conflicted_seed_count": int(len(trusted_rows) - attached.sum()),
        "conflicted_nodes": int(conflict.sum()), "conflicted_gaussians": int(conflict[inverse].sum()),
        "unreachable_or_over_budget_gaussians": int((~reachable).sum()),
        "ambiguous_gaussians": int(ambiguous.sum()), "unknown_gaussians": int((owner < 0).sum()),
        "assigned_gaussians": int((owner >= 0).sum()), "groups": group_reports,
        "nearest_admissible_geodesic_quantiles": _finite_quantiles(best),
    }
    metadata = {
        "assignment_prior": ASSIGNMENT_PRIOR, "version": 1, **hashes, "hash_policy": HASH_POLICY,
        "binding_is_spatial_prior_not_parts": True, "physical_uniqueness_guaranteed": False,
        "unseen_motion_identifiable": False, "observations_added": 0,
        "unknown_label": -1, "input_G0_row_order_preserved": True,
        "cell_size": cell_size, "grid_origin": origin.tolist(), "max_nodes": max_nodes,
        "node_cap_policy": "reject_no_automatic_coarsening",
        "representative_policy": "first_actual_G0_row_per_lexicographically_ordered_occupied_voxel",
        "neighbors": neighbors, "edge_policy": "mutual_kNN_and_adjacent_occupied_voxels_and_dual_length_cap",
        "max_edge_length": edge_limit, "effective_max_edge_length": local_edge_limit,
        "max_seed_attachment_distance": attachment_limit,
        "max_euclidean_distance": euclidean_limit, "max_path_length": path_limit,
        "path_cost_includes_seed_and_G0_attachments": True,
        "min_relative_margin": margin_limit,
        "margin_definition": "(second_best-best)/second_best_among_budget_admissible_different_labels",
        "same_node_seed_conflict_policy": "unknown_barrier",
        "other_label_seed_nodes_are_barriers": True, "automatic_radius_expansion": False,
        "diagnostics": diagnostics,
    }
    return {"source_labels": owner, "nearest_geodesic": best, "margin": margin,
            "metadata": metadata, "diagnostics": diagnostics}
