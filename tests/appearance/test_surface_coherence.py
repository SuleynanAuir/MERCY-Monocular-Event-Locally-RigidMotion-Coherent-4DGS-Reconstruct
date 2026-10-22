import unittest

import numpy as np

from scripts.appearance.cohere_lego_surfaces import component_vote, disk_normals, opacity_weights, tangent_edges


class SurfaceCoherenceTests(unittest.TestCase):
    def test_normals_and_infinite_opacity(self):
        np.testing.assert_allclose(disk_normals([[1, 0, 0, 0], [2, 0, 0, 0]]), [[0, 0, 1], [0, 0, 1]])
        np.testing.assert_allclose(opacity_weights([-np.inf, 0, np.inf]), [0, .5, 1])
        with self.assertRaises(ValueError):
            disk_normals([[0, 0, 0, 0]])

    def test_tangent_graph_rejects_parallel_depth_layers_and_bands(self):
        xyz = np.array([[0, 0, 0], [.001, 0, 0], [0, 0, .001], [.002, 0, 0]])
        normals = np.tile([0, 0, 1], (4, 1))
        radius = np.full(4, .001)
        neighbors = np.array([[0, 1, 2, 3]])
        distance = np.linalg.norm(xyz[neighbors] - xyz[0], axis=-1)
        keep = tangent_edges(xyz, normals, radius, np.array([0]), neighbors, distance,
                             np.array([False, False, False, True]), 3, .006, .94, .3, .0002)
        np.testing.assert_array_equal(keep, [[False, True, False, False]])

    def test_component_majority_rejects_ambiguous_and_oversized(self):
        components = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        groups = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1])
        weights = np.array([1, 1, .1, 1, 1, 1, 1, 1, 1])
        sizes, winner, purity, eligible, mixed = component_vote(components, groups, weights, 2, 3, .7)
        np.testing.assert_array_equal(sizes, [3, 2, 4])
        np.testing.assert_array_equal(winner, [0, 0, 1])
        np.testing.assert_array_equal(eligible, [True, False, False])
        np.testing.assert_array_equal(mixed, [True, True, False])
        self.assertGreater(purity[0], .9)


if __name__ == '__main__':
    unittest.main()
