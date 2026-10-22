"""Small synthetic occupied-geometry tests; never read G0/event/model files.

Cloud uses real SciPy. When the local SciPy binary cannot import, only these
small tests substitute brute-force neighbors and heap Dijkstra; production
never has this fallback. The test backend is reported in setUpClass.
"""

import heapq
import json
import math
import sys
import unittest
from unittest import mock

import numpy as np

from utils import event_only_geodesic_binding as binding


class _SmallTree:
    def __init__(self, values):
        self.values = np.asarray(values)
        if len(self.values) > 512:
            raise AssertionError("Synthetic tree double is restricted to <=512 nodes")

    def query(self, values, k=1):
        query = np.asarray(values)
        distance = np.linalg.norm(query[:, None] - self.values[None], axis=-1)
        indices = np.argsort(distance, axis=1, kind="stable")[:, :k]
        result = np.take_along_axis(distance, indices, axis=1)
        return (result[:, 0], indices[:, 0]) if k == 1 else (result, indices)


class _SmallSparse:
    def __init__(self, values, shape):
        weights, (rows, columns) = values
        self.shape = shape
        self.edges = {}
        for row, column, weight in zip(rows, columns, weights):
            key = int(row), int(column)
            self.edges[key] = self.edges.get(key, 0.) + float(weight)

    def tocsr(self):
        return self


def _small_dijkstra(graph, *, directed, indices, limit):
    if not directed or graph.shape[0] > 513:
        raise AssertionError("Synthetic Dijkstra double only supports small directed tests")
    neighbors = [[] for _ in range(graph.shape[0])]
    for (left, right), weight in graph.edges.items():
        neighbors[left].append((right, weight))
    distance = np.full(graph.shape[0], np.inf)
    distance[indices] = 0
    queue = [(0., indices)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance != distance[current]:
            continue
        for other, weight in neighbors[current]:
            candidate = current_distance + weight
            if candidate <= limit and candidate < distance[other]:
                distance[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return distance


def line(count=9):
    return np.column_stack((np.arange(count) * .125, np.zeros(count), np.zeros(count)))


def options(**changes):
    result = dict(cell_size=.125, max_euclidean_distance=2., max_path_length=2.,
                  max_edge_length=.14, max_nodes=100, neighbors=2, min_relative_margin=.15)
    result.update(changes)
    return result


class GeodesicBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            binding._scipy_backend()
        except RuntimeError:
            cls.use_small_test_double = True
            print("\nGeodesic tests: small synthetic backend double (local SciPy unavailable); cloud SciPy verification still required.", file=sys.stderr)
        else:
            cls.use_small_test_double = False
            print("\nGeodesic tests: real SciPy sparse/cKDTree backend.", file=sys.stderr)

    def setUp(self):
        if self.use_small_test_double:
            patcher = mock.patch.object(binding, "_scipy_backend", return_value=(_SmallTree, _SmallSparse, _small_dijkstra))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_connected_thin_shell_reaches_back_by_a_bounded_nonstraight_path(self):
        front = np.array([[0., 0, step * .125] for step in range(9)])
        rim = np.array([[step * .125, 0, 1.] for step in range(1, 9)])
        back = np.array([[1., 0, step * .125] for step in range(7, -1, -1)])
        shell = np.concatenate((front, rim, back))
        result = binding.build_geodesic_binding(shell, shell[:1], np.array([4]),
                                                **options(max_path_length=3.1, max_euclidean_distance=1.5))
        np.testing.assert_array_equal(result["source_labels"], 4)
        self.assertGreater(result["nearest_geodesic"][-1], 2.5)
        self.assertLessEqual(result["nearest_geodesic"][-1], 3.1)
        shorter = binding.build_geodesic_binding(shell, shell[:1], np.array([4]),
                                                 **options(max_path_length=2., max_euclidean_distance=1.5))
        self.assertEqual(shorter["source_labels"][-1], -1)

    def test_close_disconnected_sheet_cannot_jump_an_explicit_gap(self):
        first = line(5)
        second = first + [0, .1875, 0]
        xyz = np.concatenate((first, second))
        result = binding.build_geodesic_binding(xyz, first[:1], np.array([2]),
                                                **options(max_euclidean_distance=1., neighbors=4))
        np.testing.assert_array_equal(result["source_labels"][:5], 2)
        np.testing.assert_array_equal(result["source_labels"][5:], -1)
        self.assertLess(np.linalg.norm(second[0] - first[0]), 1.)

    def test_large_declared_edge_cap_does_not_cross_empty_voxel_layers(self):
        xyz = np.array([[0., 0, 0], [.375, 0, 0]])
        result = binding.build_geodesic_binding(xyz, xyz[:1], np.array([1]),
                                                **options(max_edge_length=99.))
        np.testing.assert_array_equal(result["source_labels"], [1, -1])
        self.assertEqual(result["diagnostics"]["undirected_edges"], 0)
        self.assertAlmostEqual(result["metadata"]["effective_max_edge_length"], math.sqrt(3) * .125)

    def test_two_groups_compete_and_exact_tie_is_unknown_even_at_zero_threshold(self):
        xyz = line()
        for threshold in (0., .15):
            result = binding.build_geodesic_binding(xyz, xyz[[0, -1]], np.array([3, 9]),
                                                    **options(min_relative_margin=threshold))
            with self.subTest(threshold=threshold):
                np.testing.assert_array_equal(result["source_labels"], [3, 3, 3, 3, -1, 9, 9, 9, 9])
                self.assertEqual(result["margin"][4], 0)
                self.assertAlmostEqual(result["margin"][3], .4)
        stricter = binding.build_geodesic_binding(xyz, xyz[[0, -1]], np.array([3, 9]),
                                                  **options(min_relative_margin=.6))
        np.testing.assert_array_equal(stricter["source_labels"][3:6], -1)

    def test_same_node_conflicting_seed_labels_are_unknown_barriers(self):
        xyz = np.array([[0., 0, 0], [.03125, 0, 0], [.125, 0, 0], [.25, 0, 0]])
        result = binding.build_geodesic_binding(xyz, xyz[:2], np.array([1, 7]), **options())
        np.testing.assert_array_equal(result["source_labels"], -1)
        self.assertEqual(result["diagnostics"]["conflicted_nodes"], 1)
        self.assertEqual(result["diagnostics"]["conflicted_gaussians"], 2)
        self.assertEqual(result["diagnostics"]["attached_seed_count"], 0)
        self.assertTrue(np.isinf(result["nearest_geodesic"]).all())

    def test_seed_and_G0_attachments_both_consume_path_budget(self):
        xyz = np.array([[0., 0, 0], [.0625, 0, 0], [.125, 0, 0]])
        seeds = np.array([[.03125, 0, 0]])
        result = binding.build_geodesic_binding(xyz, seeds, np.array([2]), **options(max_path_length=.08))
        np.testing.assert_array_equal(result["source_labels"], [2, -1, -1])
        self.assertAlmostEqual(result["nearest_geodesic"][0], .03125)
        self.assertTrue(result["metadata"]["path_cost_includes_seed_and_G0_attachments"])

    def test_exact_on_G0_seed_keeps_zero_cost_sparse_attachment_edge(self):
        xyz = line(3)
        result = binding.build_geodesic_binding(xyz, xyz[:1], np.array([0]), **options(max_path_length=.01))
        np.testing.assert_array_equal(result["source_labels"], [0, -1, -1])
        self.assertEqual(result["nearest_geodesic"][0], 0)

    def test_other_label_anchors_are_barriers_not_virtual_source_shortcuts(self):
        xyz = line()
        result = binding.build_geodesic_binding(xyz, xyz[[0, 2, 7]], np.array([1, 2, 2]),
                                                **options(min_relative_margin=.9))
        self.assertEqual(result["source_labels"][-1], 2)
        self.assertEqual(result["margin"][-1], 1)
        self.assertAlmostEqual(result["nearest_geodesic"][-1], .125)

    def test_euclidean_budget_is_enforced_separately_from_geodesic_budget(self):
        xyz = line()
        result = binding.build_geodesic_binding(xyz, xyz[:1], np.array([5]),
                                                **options(max_euclidean_distance=.4, max_path_length=2.))
        np.testing.assert_array_equal(result["source_labels"][:4], 5)
        np.testing.assert_array_equal(result["source_labels"][4:], -1)
        self.assertFalse(result["metadata"]["automatic_radius_expansion"])

    def test_far_seed_is_not_snapped_unboundedly_to_an_occupied_component(self):
        xyz = line()
        result = binding.build_geodesic_binding(xyz, np.array([[50., 0, 0]]), np.array([1]), **options())
        np.testing.assert_array_equal(result["source_labels"], -1)
        self.assertEqual(result["diagnostics"]["attached_seed_count"], 0)

    def test_node_cap_explicitly_rejects_before_graph_build_without_coarsening(self):
        xyz = line()
        with mock.patch.object(binding, "_scipy_backend") as backend:
            with self.assertRaisesRegex(ValueError, "exceeds.*no automatic coarsening"):
                binding.build_geodesic_binding(xyz, xyz[:1], np.array([1]), **options(max_nodes=4))
            backend.assert_not_called()

    def test_full_G0_row_order_duplicates_and_repeated_execution_are_preserved(self):
        xyz = np.concatenate((line(), line()[[0, 4, 8]]))
        seeds, labels = xyz[[0, 8]], np.array([3, 9])
        first = binding.build_geodesic_binding(xyz, seeds, labels, **options())
        second = binding.build_geodesic_binding(xyz, seeds, labels, **options())
        for key in ("source_labels", "nearest_geodesic", "margin"):
            np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(first[key].shape, (len(xyz),))
        np.testing.assert_array_equal(first["source_labels"][-3:], first["source_labels"][[0, 4, 8]])
        order = np.arange(len(xyz))[::-1]
        permuted = binding.build_geodesic_binding(xyz[order], seeds, labels, **options())
        np.testing.assert_array_equal(permuted["source_labels"], first["source_labels"][order])
        self.assertEqual(first["source_labels"].dtype, np.int32)

    def test_empty_trusted_seed_set_is_unknown_and_metadata_is_finite_JSON(self):
        xyz = line()
        with mock.patch.object(binding, "_scipy_backend") as backend:
            result = binding.build_geodesic_binding(xyz, np.empty((0, 3)), np.empty(0, dtype=np.int32), **options())
            backend.assert_not_called()
        np.testing.assert_array_equal(result["source_labels"], -1)
        np.testing.assert_array_equal(result["margin"], 0)
        self.assertTrue(np.isinf(result["nearest_geodesic"]).all())
        self.assertIsNone(result["diagnostics"]["nearest_admissible_geodesic_quantiles"])
        json.dumps(result["metadata"], allow_nan=False)
        json.dumps(result["diagnostics"], allow_nan=False)
        self.assertEqual(result["metadata"]["observations_added"], 0)
        self.assertTrue(result["metadata"]["binding_is_spatial_prior_not_parts"])

    def test_hashes_bind_G0_row_order_and_trusted_seed_geometry_and_labels(self):
        xyz, seeds, labels = line(), line()[[0, -1]], np.array([3, 9])
        expected = binding.geodesic_binding_input_hashes(xyz, seeds, labels)
        actual = binding.build_geodesic_binding(xyz, seeds, labels, **options())["metadata"]
        for key, digest in expected.items():
            self.assertEqual(actual[key], digest)
            self.assertEqual(len(digest), 64)
        reverse = binding.geodesic_binding_input_hashes(xyz[::-1], seeds, labels)
        self.assertNotEqual(reverse["g0_xyz_sha256"], expected["g0_xyz_sha256"])
        changed = binding.geodesic_binding_input_hashes(xyz, seeds, np.array([3, 8]))
        self.assertNotEqual(changed["trusted_seed_sha256"], expected["trusted_seed_sha256"])
        unknown_added = binding.geodesic_binding_input_hashes(xyz, np.concatenate((seeds, [[99, 0, 0]])),
                                                             np.r_[labels, -1])
        self.assertEqual(unknown_added, expected)
        canonical = binding.geodesic_binding_input_hashes(xyz.astype(np.float32), seeds.astype(np.float32), labels.astype(np.int32))
        self.assertEqual(canonical, expected)

    def test_invalid_budgets_labels_and_nonfinite_arrays_are_rejected(self):
        xyz, seeds, labels = line(), line()[:1], np.array([1])
        cases = [{"cell_size": 0}, {"cell_size": math.nan}, {"max_euclidean_distance": math.inf},
                 {"max_path_length": -1}, {"max_edge_length": True}, {"max_nodes": 100001},
                 {"max_nodes": True}, {"neighbors": 9}, {"min_relative_margin": -1},
                 {"min_relative_margin": math.nan}]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                binding.build_geodesic_binding(xyz, seeds, labels, **options(**overrides))
        for bad_labels in (np.array([-2]), np.array([.5]), np.array([2**40])):
            with self.subTest(labels=bad_labels), self.assertRaises(ValueError):
                binding.build_geodesic_binding(xyz, seeds, bad_labels, **options())
        invalid = xyz.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            binding.build_geodesic_binding(invalid, seeds, labels, **options())


if __name__ == "__main__":
    unittest.main()
